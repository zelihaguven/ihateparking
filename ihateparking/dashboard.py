"""FastAPI dashboard for the mock CARLA perception/planning loop."""
from __future__ import annotations

from dataclasses import asdict
import argparse
import math
import os
from pathlib import Path
import threading
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .ai_perception import ObstacleEstimate, ParkingSlotEstimate
from .controller import ControllerConfig, PlanTracker
from .dynamic_safety import DynamicReplanner, SafetyAction, SafetyConfig, SafetyDecision
from .geometry import Pose
from .mock_carla_server import MockCarlaWorld
from .planner import HybridAStar, PlanResult, PlannerConfig
from .scenario_generator import ScenarioSpec
from .stack_contracts import AutonomyMode, Gear, PerceptionFrame


ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"


class ResetRequest(BaseModel):
    seed: int = Field(default=100, ge=0, le=2_147_483_647)
    kind: Literal[
        "perpendicular_forward", "perpendicular_reverse", "parallel_forward",
        "parallel_reverse", "angled_forward", "angled_reverse",
    ] = "perpendicular_reverse"
    dynamic_actor_count: int = Field(default=0, ge=0, le=6)


class TickRequest(BaseModel):
    dt: float = Field(default=0.10, ge=0.01, le=0.20)


class ManualStepRequest(BaseModel):
    speed_mps: float = Field(default=0.35, ge=0.0, le=0.70)
    steering_angle_rad: float = Field(default=0.0, ge=-0.56, le=0.56)
    gear: Literal["drive", "reverse", "park"] = "drive"
    dt: float = Field(default=0.10, ge=0.01, le=0.20)


class MpcDemoRequest(BaseModel):
    horizon_steps: int = Field(default=4, ge=3, le=10)
    max_iterations: int = Field(default=12, ge=5, le=60)
    dt: float = Field(default=0.18, ge=0.10, le=0.25)
    extra_clearance_m: float = Field(default=0.05, ge=0.0, le=0.10)
    warm_start: bool = True
    derivative_mode: Literal["finite_difference", "casadi_ad"] = "finite_difference"


class ParkingSearchRequest(BaseModel):
    seed: int = Field(default=100, ge=0, le=2_147_483_647)
    case: Literal[
        "clear_slot", "cars_too_close", "slot_too_short",
        "occupied_slot", "low_confidence", "no_verified_path",
    ] = "clear_slot"


DASHBOARD_PLANNER_EXPANSIONS = 12_000


class DashboardSession:
    GOAL_POSITION_TOLERANCE_M = .25
    GOAL_YAW_TOLERANCE_RAD = math.radians(7.0)

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir or os.environ.get("IHATEPARKING_OUTPUT_DIR", "artifacts/dashboard"))
        self.world = MockCarlaWorld(self.output_dir / "mock_carla")
        # The default search ends exactly on the goal via a Reeds-Shepp shot;
        # the expansion cap bounds the worst-case reset latency.
        # A tolerance-only fallback plan must still end inside the capture
        # tolerance that tick() uses to declare the vehicle parked.
        self.initial_planner_config = PlannerConfig(
            max_expansions=DASHBOARD_PLANNER_EXPANSIONS,
            goal_position_tolerance=self.GOAL_POSITION_TOLERANCE_M,
            goal_yaw_tolerance=self.GOAL_YAW_TOLERANCE_RAD,
        )
        self.tracking_config = ControllerConfig(dt=.10, speed=.5)
        self.safety = DynamicReplanner(
            planner_config=PlannerConfig(
                xy_resolution=.8,
                yaw_bins=12,
                primitive_length=.6,
                primitive_sample=.3,
                steering_samples=3,
                goal_position_tolerance=.7,
                goal_yaw_tolerance=math.radians(28.0),
                max_expansions=80,
            ),
            safety_config=SafetyConfig(),
        )
        self.current_plan: PlanResult | None = None
        self.tracker: PlanTracker | None = None
        self.plan_exhausted = False
        self._pending_plans = 0
        self._lock = threading.RLock()
        self.completed = False
        self.last_safety = SafetyDecision(SafetyAction.PROCEED, True, "waiting for reset", 999.0, 0.45, None)
        self.event_log: list[dict[str, Any]] = []
        self.last_action = "proceed"
        self.reset()

    def reset(self, seed: int = 100, kind: str = "perpendicular_reverse", dynamic_actor_count: int = 0) -> dict[str, Any]:
        with self._lock:
            snapshot = self.world.reset(seed, kind, dynamic_actor_count)
            scenario = self._require_scenario()
            start = self.world.pose
            self.current_plan = None
            self.tracker = None
            self._pending_plans += 1
        # The search can take seconds; run it outside the lock so /api/state
        # stays responsive. tick() holds the vehicle still meanwhile.
        try:
            plan = HybridAStar(scenario.to_world(), self.initial_planner_config).plan(start=start)
        finally:
            with self._lock:
                self._pending_plans -= 1
        with self._lock:
            if self.world.scenario is not scenario:
                # A newer reset replaced the scene while this plan was computed.
                return self.state(self.world.snapshot())
            self._adopt_plan(plan)
            self.safety.replan_count = 0
            self.completed = False
            self.last_safety = SafetyDecision(SafetyAction.PROCEED, True, "initial plan ready", 999.0, .45, None)
            self.event_log = [{"frame": snapshot["frame"], "time_s": snapshot["timestamp_s"], "level": "info", "message": "scenario reset and initial plan generated"}]
            self.last_action = "proceed"
            return self.state(snapshot)

    def tick(self, dt: float = .10) -> dict[str, Any]:
        with self._lock:
            snapshot = self.world.snapshot()
            if self._pending_plans:
                return self.state(snapshot)
            scenario = self._require_scenario()
            # Drive the plan to its end (it finishes on the exact goal) before
            # judging the pose; stopping at the first tolerance entry parks crooked.
            plan_done = self.tracker is None or self.tracker.finished
            if self.completed or (plan_done and scenario.to_world().at_goal(self.world.pose, self.GOAL_POSITION_TOLERANCE_M, self.GOAL_YAW_TOLERANCE_RAD)):
                self.completed = True
                self.world.speed_mps = 0.0
                self.world.gear = Gear.PARK
                self.last_safety = SafetyDecision(SafetyAction.PROCEED, True, "goal pose captured; vehicle parked", 999.0, .45, None)
                self._event("complete", "goal pose captured; vehicle parked")
                return self.state(self.world.snapshot())
            perception = self._perception_from_dict(snapshot["perception"])
            assessment = self.safety.assess_and_replan(
                scenario.to_world(),
                self.world.pose,
                abs(self.world.speed_mps),
                self.world.gear,
                perception,
                self.current_plan,
            )
            self.last_safety = assessment.decision
            if assessment.replanned and assessment.plan is not None:
                self._adopt_plan(assessment.plan)
                self._event("replan", assessment.plan.message)
            if assessment.decision.action is SafetyAction.BRAKE:
                self.world.step(0.0, 0.0, Gear.PARK.value, dt)
                self._event("brake", assessment.decision.reason)
            else:
                gear, steering, speed = self._control_from_plan()
                self.world.step(speed, steering, gear.value, dt)
                if assessment.decision.action is SafetyAction.REPLAN:
                    self._event("replan", assessment.decision.reason)
            return self.state(self.world.snapshot())

    def manual_step(self, speed_mps: float, steering_angle_rad: float, gear: str, dt: float = .10) -> dict[str, Any]:
        with self._lock:
            snapshot = self.world.snapshot()
            scenario = self._require_scenario()
            perception = self._perception_from_dict(snapshot["perception"])
            self.last_safety = self.safety.monitor.evaluate(scenario.to_world(), self.world.pose, abs(self.world.speed_mps), self.world.gear, perception)
            if self.last_safety.should_brake:
                speed_mps, steering_angle_rad, gear = 0.0, 0.0, Gear.PARK.value
                self._event("brake", self.last_safety.reason)
            self.world.step(speed_mps, steering_angle_rad, gear, dt)
            return self.state(self.world.snapshot())

    def current_state(self) -> dict[str, Any]:
        with self._lock:
            return self.state(self.world.snapshot())

    def _require_scenario(self) -> ScenarioSpec:
        scenario = self.world.scenario
        if scenario is None:
            raise RuntimeError("dashboard session has not been reset")
        return scenario

    def state(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        scenario = self._require_scenario()
        return {
            "snapshot": snapshot,
            "scene": {
                "bounds": list(scenario.bounds),
                "slot": asdict(scenario.target_slot),
                "static_obstacles": [asdict(item) for item in scenario.static_obstacles],
                "dynamic_actors": [item.to_dict() for item in scenario.dynamic_actors],
            },
            "plan": self._plan_dict(self.current_plan),
            "safety": self._safety_dict(self.last_safety),
            "replan_count": self.safety.replan_count,
            "autonomy_mode": AutonomyMode.COMPLETE.value if self.completed else (AutonomyMode.SAFE_STOP.value if self.last_safety.should_brake else (AutonomyMode.PLAN_MANOEUVRE.value if self.last_safety.should_replan else AutonomyMode.EXECUTE.value)),
            "events": list(self.event_log[-30:]),
        }

    def _adopt_plan(self, plan: PlanResult) -> None:
        self.current_plan = plan
        scenario = self._require_scenario()
        self.tracker = PlanTracker(plan, scenario.vehicle.wheelbase, scenario.vehicle.max_steering, self.tracking_config)
        self.plan_exhausted = False

    def _control_from_plan(self) -> tuple[Gear, float, float]:
        """Next segment-aware tracking command; hold still once the plan ends."""
        command = self.tracker.command(self.world.pose) if self.tracker is not None else None
        if command is None:
            if not self.plan_exhausted and self.current_plan is not None and self.current_plan.success:
                self.plan_exhausted = True
                self._event("info", "plan executed; final pose is outside the capture tolerance, holding still")
            return Gear.PARK, 0.0, 0.0
        gear = Gear.REVERSE if command.gear < 0 else Gear.DRIVE
        return gear, command.steering, command.speed

    @staticmethod
    def _perception_from_dict(data: dict[str, Any]) -> PerceptionFrame:
        slots = tuple(ParkingSlotEstimate(
            slot_id=item["slot_id"], pose=Pose(**item["pose"]), length_m=item["length_m"], width_m=item["width_m"], confidence=item["confidence"], uncertainty_m=item["uncertainty_m"],
        ) for item in data.get("slots", []))
        obstacles = tuple(ObstacleEstimate(
            track_id=item["track_id"], kind=item["kind"], pose=Pose(**item["pose"]), length_m=item["length_m"], width_m=item["width_m"], confidence=item["confidence"], uncertainty_m=item["uncertainty_m"], dynamic=item.get("dynamic", False), velocity_mps=item.get("velocity_mps", 0.0),
        ) for item in data.get("obstacles", []))
        return PerceptionFrame(
            timestamp_s=data["timestamp_s"], frame_id=data["frame_id"], slots=slots, obstacles=obstacles,
            drivable_space_confidence=data.get("drivable_space_confidence", 0.0), out_of_distribution_score=data.get("out_of_distribution_score", 1.0), model_name=data.get("model_name", "unknown"), extra=data.get("extra", {}),
        )

    @staticmethod
    def _plan_dict(plan: PlanResult | None) -> dict[str, Any] | None:
        if plan is None:
            return None
        return {"success": plan.success, "message": plan.message, "expanded": plan.expanded, "cost": plan.cost, "path_length": plan.path_length, "poses": [asdict(pose) for pose in plan.poses], "gears": plan.gears, "steerings": plan.steerings}

    @staticmethod
    def _safety_dict(decision: SafetyDecision) -> dict[str, Any]:
        return {"action": decision.action.value, "safe": decision.safe, "reason": decision.reason, "minimum_clearance_m": decision.minimum_clearance_m, "braking_distance_m": decision.braking_distance_m, "time_to_collision_s": decision.time_to_collision_s, "predicted_collisions": list(decision.predicted_collisions), "emergency_stop": decision.emergency_stop}

    def _event(self, level: str, message: str) -> None:
        event = {"frame": self.world.frame, "time_s": round(self.world.timestamp_s, 3), "level": level, "message": message}
        if not self.event_log or self.event_log[-1]["message"] != message or level != self.last_action:
            self.event_log.append(event)
        self.last_action = level


app = FastAPI(title="ihateparking Dynamic Safety Console", version="0.3")
session = DashboardSession()
_heavy_simulation_lock = threading.Lock()
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
app.mount("/assets", StaticFiles(directory=str(ROOT / "assets")), name="assets")
app.mount("/artifacts", StaticFiles(directory=str(ROOT / "artifacts")), name="artifacts")


@app.get("/manus-routes.json", include_in_schema=False)
def routes_manifest() -> FileResponse:
    return FileResponse(ROOT / "manus-routes.json", media_type="application/json")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(WEB_DIR / "dashboard.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard_alias() -> FileResponse:
    return FileResponse(WEB_DIR / "dashboard.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "ihateparking-dashboard", "backend": "FastAPI", "mock_carla": True}


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    return session.current_state()


@app.post("/api/reset")
async def reset(body: ResetRequest) -> dict[str, Any]:
    return await run_in_threadpool(session.reset, body.seed, body.kind, body.dynamic_actor_count)


@app.post("/api/tick")
async def tick(body: TickRequest) -> dict[str, Any]:
    return await run_in_threadpool(session.tick, body.dt)


@app.post("/api/step")
async def step(body: ManualStepRequest) -> dict[str, Any]:
    return await run_in_threadpool(session.manual_step, body.speed_mps, body.steering_angle_rad, body.gear, body.dt)


@app.post("/api/mpc/run")
async def run_mpc_demo(body: MpcDemoRequest) -> dict[str, Any]:
    if not _heavy_simulation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Another compute-intensive simulation is running on this worker; retry after it finishes.",
        )

    def execute() -> dict[str, Any]:
        try:
            from scripts.demo_constrained_mpc import (
                MPCConfig,
                comparison_payload,
                run_comparison,
            )
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="The optional MPC demo dependencies are not installed on this server.",
            ) from exc
        config = MPCConfig(
            horizon_steps=body.horizon_steps,
            max_iterations=body.max_iterations,
            dt=body.dt,
            extra_clearance_m=body.extra_clearance_m,
            warm_start=body.warm_start,
            derivative_mode=body.derivative_mode,
        )
        try:
            return comparison_payload(run_comparison(config))
        except RuntimeError as exc:
            if "SciPy is required" in str(exc) or "CasADi is required" in str(exc):
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            raise

    try:
        return await run_in_threadpool(execute)
    finally:
        _heavy_simulation_lock.release()


@app.post("/api/parking-search/run")
async def run_parking_search_demo(body: ParkingSearchRequest) -> dict[str, Any]:
    if not _heavy_simulation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Another compute-intensive simulation is running on this worker; retry after it finishes.",
        )
    try:
        from ihateparking.parking_search import run_parking_search

        return await run_in_threadpool(run_parking_search, body.seed, body.case)
    finally:
        _heavy_simulation_lock.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the ihateparking dynamic safety dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "3000")))
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
