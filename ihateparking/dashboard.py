"""FastAPI dashboard for the mock CARLA perception/planning loop."""
from __future__ import annotations

from dataclasses import asdict
import argparse
import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .ai_perception import ObstacleEstimate, ParkingSlotEstimate
from .dynamic_safety import DynamicReplanner, SafetyAction, SafetyConfig, SafetyDecision
from .geometry import Pose, distance
from .mock_carla_server import MockCarlaWorld
from .planner import HybridAStar, PlanResult, PlannerConfig
from .stack_contracts import AutonomyMode, Gear, PerceptionFrame


ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"


class DashboardSession:
    def __init__(self, output_dir: str | Path = "artifacts/dashboard"):
        self.output_dir = Path(output_dir)
        self.world = MockCarlaWorld(self.output_dir / "mock_carla")
        self.safety = DynamicReplanner(
            planner_config=PlannerConfig(max_expansions=20_000),
            safety_config=SafetyConfig(),
        )
        self.current_plan: PlanResult | None = None
        self.last_safety = SafetyDecision(SafetyAction.PROCEED, True, "waiting for reset", 999.0, 0.45, None)
        self.event_log: list[dict[str, Any]] = []
        self.last_action = "proceed"
        self.reset()

    def reset(self, seed: int = 100, kind: str = "perpendicular_reverse", dynamic_actor_count: int = 2) -> dict[str, Any]:
        snapshot = self.world.reset(seed, kind, dynamic_actor_count)
        self.current_plan = HybridAStar(self.world.scenario.to_world(), PlannerConfig(max_expansions=20_000)).plan(start=self.world.pose)
        self.safety.replan_count = 0
        self.last_safety = SafetyDecision(SafetyAction.PROCEED, True, "initial plan ready", 999.0, .45, None)
        self.event_log = [{"frame": snapshot["frame"], "time_s": snapshot["timestamp_s"], "level": "info", "message": "scenario reset and initial plan generated"}]
        self.last_action = "proceed"
        return self.state(snapshot)

    def tick(self, dt: float = .10) -> dict[str, Any]:
        snapshot = self.world.snapshot()
        perception = self._perception_from_dict(snapshot["perception"])
        assessment = self.safety.assess_and_replan(
            self.world.scenario.to_world(),
            self.world.pose,
            abs(self.world.speed_mps),
            self.world.gear,
            perception,
            self.current_plan,
        )
        self.last_safety = assessment.decision
        if assessment.replanned and assessment.plan is not None:
            self.current_plan = assessment.plan
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
        snapshot = self.world.snapshot()
        perception = self._perception_from_dict(snapshot["perception"])
        self.last_safety = self.safety.monitor.evaluate(self.world.scenario.to_world(), self.world.pose, abs(self.world.speed_mps), self.world.gear, perception)
        if self.last_safety.should_brake:
            speed_mps, steering_angle_rad, gear = 0.0, 0.0, Gear.PARK.value
            self._event("brake", self.last_safety.reason)
        self.world.step(speed_mps, steering_angle_rad, gear, dt)
        return self.state(self.world.snapshot())

    def state(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        scenario = self.world.scenario
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
            "autonomy_mode": AutonomyMode.SAFE_STOP.value if self.last_safety.should_brake else (AutonomyMode.PLAN_MANOEUVRE.value if self.last_safety.should_replan else AutonomyMode.EXECUTE.value),
            "events": list(self.event_log[-30:]),
        }

    def _control_from_plan(self) -> tuple[Gear, float, float]:
        if self.current_plan is None or not self.current_plan.success or len(self.current_plan.poses) < 2:
            return Gear.PARK, 0.0, 0.0
        pose = self.world.pose
        closest = min(range(len(self.current_plan.poses)), key=lambda index: distance(pose, self.current_plan.poses[index]))
        target_index = min(len(self.current_plan.poses) - 1, closest + 2)
        raw_gear = self.current_plan.gears[target_index]
        gear = Gear.REVERSE if raw_gear < 0 else Gear.DRIVE
        target = self.current_plan.poses[target_index]
        from .controller import pure_pursuit
        steering = pure_pursuit(pose, target, raw_gear, self.world.scenario.vehicle.wheelbase, .6, self.world.scenario.vehicle.max_steering)
        return gear, steering, .35

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


app = FastAPI(title="ihateparking Dynamic Safety Console", version="0.2")
session = DashboardSession()
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


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
    return session.state(session.world.snapshot())


@app.post("/api/reset")
async def reset(request: Request) -> dict[str, Any]:
    body = await request.json()
    return session.reset(int(body.get("seed", 100)), str(body.get("kind", "perpendicular_reverse")), int(body.get("dynamic_actor_count", 2)))


@app.post("/api/tick")
async def tick(request: Request) -> dict[str, Any]:
    body = await request.json()
    return session.tick(float(body.get("dt", .10)))


@app.post("/api/step")
async def step(request: Request) -> dict[str, Any]:
    body = await request.json()
    return session.manual_step(float(body.get("speed_mps", .35)), float(body.get("steering_angle_rad", 0.0)), str(body.get("gear", "drive")), float(body.get("dt", .10)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the ihateparking dynamic safety dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
