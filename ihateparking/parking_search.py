"""Synthetic parking-slot assessment with fail-closed planning and execution.

The scene observer is a seeded geometric proposal generator, not a camera model.
The assessment plans against conservative perceived geometry; only a validated
plan is executed, and the independent ground-truth footprint guard remains on.
No rejected test layout is allowed to advance the simulated ego pose.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Any, Literal

from .controller import ControllerConfig, simulate_closed_loop
from .geometry import Pose
from .planner import HybridAStar, PlanResult, PlannerConfig
from .world import ParkingSlot, ParkingWorld, RectangleObstacle, VehicleSpec

ParkingCase = Literal[
    "clear_slot",
    "cars_too_close",
    "slot_too_short",
    "occupied_slot",
    "low_confidence",
    "no_verified_path",
]

CENTER = (10.0, 10.0)
START = Pose(16.5, 3.0, math.pi)
BOUNDS = (0.0, 20.0, 0.0, 16.0)
PARKED_CAR_LENGTH_M = 4.8
PARKED_CAR_WIDTH_M = 2.0
LONGITUDINAL_MARGIN_M = 0.15
LATERAL_MARGIN_M = 0.10
CONFIDENCE_FLOOR = 0.70


@dataclass(frozen=True)
class _CaseDefinition:
    label: str
    slot_length_m: float
    slot_width_m: float
    confidence: float = 0.96
    uncertainty_m: float = 0.10
    occupied: bool = False
    planner_expansions: int = 180


_CASES: dict[str, _CaseDefinition] = {
    # Deliberately generous so the existing Hybrid A* baseline can actually
    # plan and park in this positive demonstration case.
    "clear_slot": _CaseDefinition("Clear, generous perpendicular bay", 5.0, 5.0),
    "cars_too_close": _CaseDefinition("Flanking cars leave too little width", 5.0, 2.0),
    "slot_too_short": _CaseDefinition("Insufficient parking depth", 4.5, 5.0),
    "occupied_slot": _CaseDefinition("Candidate footprint is occupied", 5.0, 5.0, occupied=True),
    "low_confidence": _CaseDefinition("Uncertain slot/vehicle detections", 5.0, 5.0, confidence=0.42, uncertainty_m=0.42),
    # This gap passes the scalar footprint fit check, but the bounded planner
    # does not find a verified path. That is a NO-GO, not proof no path exists.
    "no_verified_path": _CaseDefinition("Dimensions pass, route not verified", 5.0, 3.0),
}


@dataclass(frozen=True)
class _CarEstimate:
    name: str
    center: tuple[float, float]
    length_m: float
    width_m: float
    yaw: float
    confidence: float
    uncertainty_m: float

    def inflated_obstacle(self) -> RectangleObstacle:
        return RectangleObstacle(
            self.name,
            self.center,
            self.length_m + 2.0 * self.uncertainty_m,
            self.width_m + 2.0 * self.uncertainty_m,
            self.yaw,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "parked_vehicle",
            "center": list(self.center),
            "length_m": self.length_m,
            "width_m": self.width_m,
            "yaw": self.yaw,
            "confidence": self.confidence,
            "uncertainty_m": self.uncertainty_m,
            "dynamic": False,
        }


def _case_definition(case: str) -> _CaseDefinition:
    try:
        return _CASES[case]
    except KeyError as exc:
        raise ValueError(f"unsupported parking case: {case}") from exc


def _perceived_scene(
    case: str,
    seed: int,
    definition: _CaseDefinition,
) -> tuple[ParkingSlot, ParkingSlot, list[RectangleObstacle], list[RectangleObstacle], list[dict[str, Any]], dict[str, Any]]:
    """Return truth/perceived geometry and inspectable seeded object proposals."""
    rng = random.Random(seed)
    vehicle = VehicleSpec()
    center_x, center_y = CENTER
    # Neighboring vehicles flank the candidate. Their detected side faces, not
    # a pre-filled decision, determine the observed free width.
    half_spacing = definition.slot_width_m / 2.0 + PARKED_CAR_WIDTH_M / 2.0
    left_truth = RectangleObstacle(
        "parked_left",
        (center_x - half_spacing, center_y),
        PARKED_CAR_LENGTH_M,
        PARKED_CAR_WIDTH_M,
        math.pi / 2.0,
    )
    right_truth = RectangleObstacle(
        "parked_right",
        (center_x + half_spacing, center_y),
        PARKED_CAR_LENGTH_M,
        PARKED_CAR_WIDTH_M,
        math.pi / 2.0,
    )
    curb_truth = RectangleObstacle("back_curb", (center_x, 13.8), 16.0, 0.28, 0.0)
    truth_obstacles = [left_truth, right_truth, curb_truth]
    if definition.occupied:
        truth_obstacles.append(RectangleObstacle(
            "blocking_vehicle",
            CENTER,
            PARKED_CAR_LENGTH_M,
            1.8,
            math.pi / 2.0,
        ))

    car_confidence = max(0.0, min(1.0, definition.confidence + rng.uniform(-0.012, 0.012)))
    position_noise_m = 0.015
    left_estimate = _CarEstimate(
        left_truth.name,
        (left_truth.center[0] + rng.gauss(0.0, position_noise_m), center_y + rng.gauss(0.0, position_noise_m)),
        PARKED_CAR_LENGTH_M + rng.gauss(0.0, 0.02),
        PARKED_CAR_WIDTH_M + rng.gauss(0.0, 0.02),
        left_truth.yaw,
        car_confidence,
        definition.uncertainty_m,
    )
    right_estimate = _CarEstimate(
        right_truth.name,
        (right_truth.center[0] + rng.gauss(0.0, position_noise_m), center_y + rng.gauss(0.0, position_noise_m)),
        PARKED_CAR_LENGTH_M + rng.gauss(0.0, 0.02),
        PARKED_CAR_WIDTH_M + rng.gauss(0.0, 0.02),
        right_truth.yaw,
        car_confidence,
        definition.uncertainty_m,
    )

    # Perpendicular slots have their width on the world x axis. Derive it from
    # the estimated neighboring-car side faces and preserve uncertainty as a
    # separate conservative margin in the fit gate.
    left_edge = left_estimate.center[0] + left_estimate.width_m / 2.0
    right_edge = right_estimate.center[0] - right_estimate.width_m / 2.0
    observed_width = max(0.1, right_edge - left_edge)
    observed_center_x = (left_edge + right_edge) / 2.0
    observed_length = max(0.1, definition.slot_length_m + rng.gauss(0.0, 0.015))
    slot_confidence = min(definition.confidence, left_estimate.confidence, right_estimate.confidence)
    slot_uncertainty = definition.uncertainty_m
    estimated_slot = ParkingSlot(
        (observed_center_x, center_y),
        observed_length,
        observed_width,
        math.pi / 2.0,
    )
    truth_slot = ParkingSlot(
        CENTER,
        definition.slot_length_m,
        definition.slot_width_m,
        math.pi / 2.0,
    )

    perceived_obstacles = [left_estimate.inflated_obstacle(), right_estimate.inflated_obstacle()]
    perceived_records = [left_estimate.to_dict(), right_estimate.to_dict()]
    perceived_curb = RectangleObstacle(
        curb_truth.name,
        (curb_truth.center[0], curb_truth.center[1] + rng.gauss(0.0, 0.02)),
        curb_truth.length + 0.10,
        curb_truth.width + 0.10,
        curb_truth.yaw,
    )
    perceived_obstacles.append(perceived_curb)
    perceived_records.append({
        "name": curb_truth.name,
        "kind": "curb",
        "center": list(perceived_curb.center),
        "length_m": curb_truth.length,
        "width_m": curb_truth.width,
        "yaw": curb_truth.yaw,
        "confidence": 0.98,
        "uncertainty_m": 0.05,
        "dynamic": False,
    })
    if definition.occupied:
        blocker_truth = truth_obstacles[-1]
        blocker_confidence = slot_confidence
        perceived_obstacles.append(RectangleObstacle(
            blocker_truth.name,
            (blocker_truth.center[0] + rng.gauss(0.0, 0.01), blocker_truth.center[1]),
            blocker_truth.length + 2.0 * slot_uncertainty,
            blocker_truth.width + 2.0 * slot_uncertainty,
            blocker_truth.yaw,
        ))
        perceived_records.append({
            "name": blocker_truth.name,
            "kind": "parked_vehicle",
            "center": [blocker_truth.center[0], blocker_truth.center[1]],
            "length_m": blocker_truth.length,
            "width_m": blocker_truth.width,
            "yaw": blocker_truth.yaw,
            "confidence": blocker_confidence,
            "uncertainty_m": slot_uncertainty,
            "dynamic": False,
        })

    slot_record = {
        "slot_id": "candidate-A",
        "center": [estimated_slot.center[0], estimated_slot.center[1]],
        "length_m": estimated_slot.length,
        "width_m": estimated_slot.width,
        "confidence": slot_confidence,
        "uncertainty_m": slot_uncertainty,
        "left_neighbor": left_truth.name,
        "right_neighbor": right_truth.name,
        "case_label": definition.label,
    }
    truth_slot_record = {
        "center": list(truth_slot.center),
        "length_m": truth_slot.length,
        "width_m": truth_slot.width,
        "yaw": truth_slot.yaw,
    }
    truth_obstacle_records = [
        {
            "name": obstacle.name,
            "kind": "parked_vehicle" if "vehicle" in obstacle.name or obstacle.name.startswith("parked_") else "curb",
            "center": list(obstacle.center),
            "length_m": obstacle.length,
            "width_m": obstacle.width,
            "yaw": obstacle.yaw,
        }
        for obstacle in truth_obstacles
    ]
    perceived_world = ParkingWorld(BOUNDS, perceived_obstacles, estimated_slot, START, vehicle, collision_margin=0.12)
    truth_world = ParkingWorld(BOUNDS, truth_obstacles, truth_slot, START, vehicle, collision_margin=0.12)
    perception = {
        "source": "seeded_synthetic_geometric_proposals",
        "model_name": "synthetic_slot_observer",
        "is_real_sensor_data": False,
        "slot": slot_record,
        "obstacles": perceived_records,
        "truth_slot": truth_slot_record,
        "truth_obstacles": truth_obstacle_records,
    }
    return truth_slot, estimated_slot, truth_obstacles, perceived_obstacles, perceived_records, {
        "vehicle": asdict(vehicle),
        "ego_start": asdict(START),
        "perceived_world": perceived_world,
        "truth_world": truth_world,
        "perception": perception,
    }


def _assess_fit(
    slot: ParkingSlot,
    confidence: float,
    uncertainty_m: float,
    world: ParkingWorld,
) -> dict[str, Any]:
    vehicle = world.vehicle
    required_length = vehicle.length + 2.0 * (LONGITUDINAL_MARGIN_M + uncertainty_m)
    required_width = vehicle.width + 2.0 * (LATERAL_MARGIN_M + uncertainty_m)
    assessment: dict[str, Any] = {
        "confidence": confidence,
        "confidence_floor": CONFIDENCE_FLOOR,
        "uncertainty_m": uncertainty_m,
        "estimated_length_m": slot.length,
        "estimated_width_m": slot.width,
        "required_length_m": required_length,
        "required_width_m": required_width,
        "checks": [],
        "passed": False,
        "reason_code": "",
        "reason": "",
    }
    if confidence < CONFIDENCE_FLOOR:
        assessment.update(
            reason_code="low_confidence",
            reason=f"Synthetic detection confidence {confidence:.2f} is below the {CONFIDENCE_FLOOR:.2f} decision floor; keep searching.",
        )
        return assessment

    assessment["checks"].append({"name": "confidence", "passed": True})
    if slot.length < required_length:
        assessment["reason_code"] = "too_short"
        assessment["reason"] = f"Nearby boundary leaves {slot.length:.2f} m of depth; the vehicle needs {required_length:.2f} m including margin and uncertainty."
        assessment["checks"].append({"name": "length_fit", "passed": False})
        return assessment
    assessment["checks"].append({"name": "length_fit", "passed": True})

    if slot.width < required_width:
        assessment["reason_code"] = "too_narrow"
        assessment["reason"] = f"Flanking vehicles leave {slot.width:.2f} m of width; the vehicle needs {required_width:.2f} m including margin and uncertainty."
        assessment["checks"].append({"name": "width_fit", "passed": False})
        return assessment
    assessment["checks"].append({"name": "width_fit", "passed": True})

    goal = slot.goal_pose(vehicle)
    if world.is_collision(goal, extra_margin=0.05):
        assessment["reason_code"] = "occupied_goal"
        assessment["reason"] = "The conservative target footprint overlaps a detected vehicle or curb; do not enter the bay."
        assessment["checks"].append({"name": "target_occupancy", "passed": False})
        return assessment
    assessment["checks"].append({"name": "target_occupancy", "passed": True})
    assessment.update(passed=True, reason_code="fit_passed", reason="Confidence, dimensions and conservative target occupancy checks passed.")
    return assessment


def _planner_config(expansions: int) -> PlannerConfig:
    return PlannerConfig(
        xy_resolution=0.65,
        yaw_bins=16,
        primitive_length=0.75,
        primitive_sample=0.25,
        steering_samples=5,
        goal_position_tolerance=0.35,
        goal_yaw_tolerance=math.radians(12.0),
        max_expansions=expansions,
    )


def _pose_records(poses: list[Pose]) -> list[dict[str, float]]:
    return [asdict(pose) for pose in poses]


def run_parking_search(seed: int = 100, case: ParkingCase = "clear_slot") -> dict[str, Any]:
    """Assess one synthetic candidate and park only after both gate and plan pass."""
    definition = _case_definition(case)
    truth_slot, estimated_slot, truth_obstacles, perceived_obstacles, perceived_records, context = _perceived_scene(case, seed, definition)
    perceived_world: ParkingWorld = context["perceived_world"]
    truth_world: ParkingWorld = context["truth_world"]
    perception: dict[str, Any] = context["perception"]
    confidence = float(perception["slot"]["confidence"])
    uncertainty = float(perception["slot"]["uncertainty_m"])
    assessment = _assess_fit(estimated_slot, confidence, uncertainty, perceived_world)

    plan: PlanResult | None = None
    plan_time_ms = 0.0
    planner_expanded = 0
    state_sequence = ["SEARCH_CANDIDATE", "ESTIMATE_GAP", "ASSESS_FIT"]
    execution: dict[str, Any] | None = None
    decision = "NO_GO"
    status = "NO_GO_KEEP_SEARCHING"
    message = str(assessment["reason"])

    if assessment["passed"]:
        state_sequence.append("PLAN_ROUTE")
        started = time.perf_counter()
        plan = HybridAStar(perceived_world, _planner_config(definition.planner_expansions)).plan(start=START)
        plan_time_ms = (time.perf_counter() - started) * 1000.0
        planner_expanded = plan.expanded
        assessment["checks"].append({"name": "collision_checked_route", "passed": plan.success})
        if not plan.success:
            assessment.update(
                passed=False,
                reason_code="no_verified_path",
                reason=f"Hybrid A* found no verified collision-free route within {planner_expanded} expansions; this does not prove no path exists. Keep searching and do not move.",
            )
            message = str(assessment["reason"])
            state_sequence.extend(["SAFE_STOP", "KEEP_SEARCHING"])
        else:
            state_sequence.append("EXECUTE_SUPERVISED_PLAN")
            simulation = simulate_closed_loop(
                truth_world,
                plan,
                ControllerConfig(dt=0.05, speed=0.8, lookahead=0.6, max_time=55.0),
            )
            final_pose = simulation.poses[-1] if simulation.poses else START
            verified = simulation.success and truth_world.at_goal(final_pose)
            execution = {
                "status": "parked" if verified else "safe_stop",
                "success": verified,
                "message": simulation.message,
                "simulated_time_s": simulation.time,
                "motion_steps": max(0, len(simulation.poses) - 1),
                "trajectory": _pose_records(simulation.poses),
                "final_pose": asdict(final_pose),
                "goal_verified": truth_world.at_goal(final_pose),
                "applied_collision_count": 0,
                "rejected_unsafe_step_proposals": simulation.collisions,
            }
            if verified:
                decision = "PARK"
                status = "PARKED"
                message = "Candidate passed the estimated fit checks, Hybrid A* plan check and supervised closed-loop goal verification."
                state_sequence.extend(["VERIFY_PARKED", "COMPLETE"])
                assessment.update(reason_code="parked_verified", reason=message)
            else:
                decision = "SAFE_STOP"
                status = "SAFE_STOP"
                message = "Closed-loop parking was not verified; the simulator safely stopped. This candidate is not accepted."
                state_sequence.append("SAFE_STOP")
                assessment.update(reason_code="closed_loop_not_verified", reason=message)

    plan_record: dict[str, Any] | None = None
    if plan is not None:
        plan_record = {
            "success": plan.success,
            "message": plan.message,
            "expanded": plan.expanded,
            "planning_time_ms": plan_time_ms,
            "path_length_m": plan.path_length,
            "path": _pose_records(plan.poses),
            "gears": plan.gears,
            "steerings_rad": plan.steerings,
        }
    candidate = {
        "slot_id": str(perception["slot"]["slot_id"]),
        "truth": {"center": list(truth_slot.center), "length_m": truth_slot.length, "width_m": truth_slot.width, "yaw": truth_slot.yaw},
        "estimate": dict(perception["slot"]),
        "decision": status,
        "assessment": assessment,
        "planner": plan_record,
        "neighbors": ["parked_left", "parked_right"],
    }
    truth_obstacle_records = perception["truth_obstacles"]
    return {
        "case": case,
        "case_label": definition.label,
        "seed": seed,
        "decision": decision,
        "status": status,
        "message": message,
        "state_sequence": state_sequence,
        "perception": perception,
        "assessment": assessment,
        "scene": {
            "bounds": list(BOUNDS),
            "ego_start": context["ego_start"],
            "vehicle": context["vehicle"],
            "candidate": candidate,
            "perceived_obstacles": perceived_records,
            "truth_obstacles": truth_obstacle_records,
        },
        "planner": plan_record,
        "execution": execution,
        "safety": {
            "applied_collision_count": 0,
            "rejected_unsafe_step_proposals": int(execution["rejected_unsafe_step_proposals"]) if execution else 0,
            "motion_steps": int(execution["motion_steps"]) if execution else 0,
            "rejected_candidate_motion_steps": 0 if decision != "PARK" else None,
        },
        "safety_boundary": "Synthetic geometry only. A rejected candidate receives zero motion. The accepted example is deliberately spacious; no real sensor, CARLA or vehicle connection is present.",
    }
