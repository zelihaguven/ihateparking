"""Deterministic pedestrian crossing stress test for the safety supervisor."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from collections import Counter

from .ai_perception import ObstacleEstimate, ParkingSlotEstimate
from .dynamic_safety import DynamicReplanner, SafetyAction, SafetyConfig
from .geometry import Pose
from .mock_carla_server import MockCarlaWorld
from .planner import HybridAStar, PlannerConfig
from .scenario_generator import DynamicActorSpec, ScenarioKind
from .stack_contracts import Gear, PerceptionFrame


def perception_from_snapshot(data: dict) -> PerceptionFrame:
    return PerceptionFrame(
        timestamp_s=data["timestamp_s"],
        frame_id=data["frame_id"],
        slots=tuple(ParkingSlotEstimate(
            slot_id=item["slot_id"], pose=Pose(**item["pose"]),
            length_m=item["length_m"], width_m=item["width_m"],
            confidence=item["confidence"], uncertainty_m=item["uncertainty_m"],
        ) for item in data.get("slots", [])),
        obstacles=tuple(ObstacleEstimate(
            track_id=item["track_id"], kind=item["kind"], pose=Pose(**item["pose"]),
            length_m=item["length_m"], width_m=item["width_m"],
            confidence=item["confidence"], uncertainty_m=item["uncertainty_m"],
            dynamic=item.get("dynamic", False), velocity_mps=item.get("velocity_mps", 0.0),
        ) for item in data.get("obstacles", [])),
        drivable_space_confidence=data.get("drivable_space_confidence", 0.0),
        out_of_distribution_score=data.get("out_of_distribution_score", 1.0),
        model_name=data.get("model_name", "unknown"),
        extra=data.get("extra", {}),
    )


def _pedestrian_crossing_scenario(world: MockCarlaWorld, count: int) -> None:
    """Place pedestrians on a deterministic crossing path near, not inside, ego."""
    actors = tuple(DynamicActorSpec(
        actor_id=f"pedestrian_{index:02d}",
        kind="pedestrian",
        # Cross just outside the ego footprint. The monitor should still be
        # conservative, but the benchmark must not manufacture contact.
        start_pose=Pose(18.5, .9 + index * .25, math.pi / 2.0),
        velocity_mps=.55 + index * .08,
        path="cross_slot_aisle",
        spawn_time_s=.4 + index * .25,
    ) for index in range(count))
    world.scenario = replace(world.scenario, dynamic_actors=actors)


def run_stress_test(output: str | Path = "artifacts/pedestrian_stress.json", frames: int = 90, pedestrian_count: int = 4) -> dict:
    world = MockCarlaWorld(Path(output).with_suffix("") / "mock_carla")
    world.reset(
        seed=2026,
        kind=ScenarioKind.PARALLEL_REVERSE.value,
        dynamic_actor_count=pedestrian_count,
        dynamic_actor_kinds=("pedestrian",),
    )
    _pedestrian_crossing_scenario(world, pedestrian_count)

    planner_config = PlannerConfig(
        xy_resolution=.8, yaw_bins=12, primitive_length=.6,
        primitive_sample=.3, steering_samples=3,
        goal_position_tolerance=.7, goal_yaw_tolerance=math.radians(28.0),
        max_expansions=80,
    )
    current_plan = HybridAStar(world.scenario.to_world(), planner_config).plan(start=world.pose)
    supervisor = DynamicReplanner(planner_config=planner_config, safety_config=SafetyConfig())
    action_counts: Counter[str] = Counter()
    snapshots: list[dict] = []
    minimum_clearance = math.inf
    first_brake_frame: int | None = None
    first_replan_frame: int | None = None
    replan_count = 0

    for _ in range(frames):
        snapshot = world.snapshot()
        snapshots.append(snapshot)
        perception = perception_from_snapshot(snapshot["perception"])
        result = supervisor.assess_and_replan(
            world.scenario.to_world(), world.pose, abs(world.speed_mps), world.gear,
            perception, current_plan,
        )
        action = result.decision.action.value
        action_counts[action] += 1
        minimum_clearance = min(minimum_clearance, result.decision.minimum_clearance_m)
        if result.replanned and result.plan is not None:
            current_plan = result.plan
            replan_count += 1
            first_replan_frame = snapshot["frame"] if first_replan_frame is None else first_replan_frame
        if result.decision.action is SafetyAction.BRAKE:
            first_brake_frame = snapshot["frame"] if first_brake_frame is None else first_brake_frame
            world.step(0.0, 0.0, Gear.PARK.value, .10)
        else:
            # Deliberately advance ego toward the crossing zone; safety must
            # override this command when a pedestrian enters the envelope.
            world.step(.35, 0.0, Gear.DRIVE.value, .10)

    final = world.snapshot()
    snapshots.append(final)
    report = {
        "benchmark": "pedestrian_dynamic_obstacle_stress_test",
        "scenario_id": world.scenario.scenario_id,
        "scenario_kind": world.scenario.kind.value,
        "pedestrian_count": pedestrian_count,
        "frames": len(snapshots),
        "ground_truth_pedestrian_frames": sum(any(item["kind"] == "pedestrian" for item in snap["ground_truth"]["obstacles"]) for snap in snapshots),
        "neural_dynamic_detection_frames": sum(any(item.get("dynamic", False) for item in snap["perception"]["obstacles"]) for snap in snapshots),
        "safety_action_counts": dict(action_counts),
        "first_replan_frame": first_replan_frame,
        "first_brake_frame": first_brake_frame,
        "replan_count": replan_count,
        "minimum_predicted_clearance_m": None if math.isinf(minimum_clearance) else minimum_clearance,
        "contact_events": final["state"]["collisions"],
        "final_speed_mps": final["state"]["speed_mps"],
        "ground_truth_source": final["ground_truth_source"],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic pedestrian crossing safety stress test")
    parser.add_argument("--output", default="artifacts/pedestrian_stress.json")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--pedestrians", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_stress_test(args.output, args.frames, args.pedestrians), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
