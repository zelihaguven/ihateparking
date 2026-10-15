"""Replayable GroundTruthBackend benchmark and first sensor-flow test.

This runner is intentionally CARLA-independent. It validates the stream and AI
contracts before a simulator is installed; the same PerceptionFrame contract is
used later by live CARLA sensor callbacks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from .ai_perception import GroundTruthBackend, NeuralPerceptionAdapter, RawObstacle, RawSlot
from .geometry import Pose
from .scenario_generator import ScenarioConfig, ScenarioGenerator, ScenarioKind, ScenarioSpec
from .stack_contracts import SensorFrame


@dataclass(frozen=True)
class GroundTruthSensorStream:
    scenario: ScenarioSpec
    frame_count: int = 5

    def __iter__(self) -> Iterator[SensorFrame]:
        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        hz = max(.1, self.scenario.sensors.perception_hz)
        for index in range(self.frame_count):
            timestamp = index / hz
            yield SensorFrame(
                timestamp_s=timestamp,
                frame_id=f"{self.scenario.scenario_id}:{index:06d}",
                rgb_uri=f"groundtruth://{self.scenario.scenario_id}/rgb_front/{index:06d}",
                depth_uri=f"groundtruth://{self.scenario.scenario_id}/depth_front/{index:06d}" if self.scenario.sensors.depth_front else None,
                lidar_uri=f"groundtruth://{self.scenario.scenario_id}/lidar/{index:06d}" if self.scenario.sensors.lidar else None,
                odometry={"x": self.scenario.start_pose.x, "y": self.scenario.start_pose.y, "yaw": self.scenario.start_pose.yaw},
            )


def backend_for_scenario(scenario: ScenarioSpec, dynamic_poses: Mapping[str, Pose] | None = None) -> GroundTruthBackend:
    dynamic_poses = dynamic_poses or {}
    slots = (RawSlot(
        slot_id=f"{scenario.scenario_id}:target",
        pose=scenario.target_slot.goal_pose(scenario.vehicle),
        length_m=scenario.target_slot.length,
        width_m=scenario.target_slot.width,
        confidence=1.0,
        uncertainty_m=0.0,
    ),)
    obstacles = [RawObstacle(
        track_id=f"{scenario.scenario_id}:{obstacle.name}",
        kind="static_obstacle",
        pose=Pose(obstacle.center[0], obstacle.center[1], obstacle.yaw),
        length_m=obstacle.length,
        width_m=obstacle.width,
        confidence=1.0,
        uncertainty_m=0.0,
    ) for obstacle in scenario.static_obstacles]
    obstacles.extend(RawObstacle(
        track_id=actor.actor_id,
        kind=actor.kind,
        pose=dynamic_poses.get(actor.actor_id, actor.start_pose),
        length_m=4.5 if actor.kind == "vehicle" else 0.7,
        width_m=2.0 if actor.kind == "vehicle" else 0.7,
        confidence=1.0,
        uncertainty_m=0.0,
        dynamic=True,
        velocity_mps=actor.velocity_mps,
    ) for actor in scenario.dynamic_actors)
    return GroundTruthBackend(slots, obstacles)


def run_ground_truth_benchmark(scenarios: Iterable[ScenarioSpec], frame_count: int = 5) -> dict:
    records = []
    total_frames = 0
    valid_frames = 0
    for scenario in scenarios:
        model = NeuralPerceptionAdapter(backend_for_scenario(scenario))
        frames = list(GroundTruthSensorStream(scenario, frame_count=frame_count))
        frame_records = []
        for sensor_frame in frames:
            perception = model.infer(sensor_frame)
            valid = perception.timestamp_s == sensor_frame.timestamp_s and perception.frame_id == sensor_frame.frame_id
            valid_frames += int(valid)
            total_frames += 1
            frame_records.append({
                "frame_id": sensor_frame.frame_id,
                "timestamp_s": sensor_frame.timestamp_s,
                "rgb_uri": sensor_frame.rgb_uri,
                "depth_uri": sensor_frame.depth_uri,
                "lidar_uri": sensor_frame.lidar_uri,
                "slot_count": len(perception.slots),
                "obstacle_count": len(perception.obstacles),
                "model_name": perception.model_name,
                "valid_contract": valid,
            })
        records.append({
            "scenario_id": scenario.scenario_id,
            "kind": scenario.kind.value,
            "expected_gear_sequence": list(scenario.expected_gear_sequence),
            "frames": frame_records,
        })
    return {
        "benchmark": "ground_truth_sensor_flow",
        "backend": "ground_truth_replay",
        "scenario_count": len(records),
        "frame_count": total_frames,
        "valid_contract_frames": valid_frames,
        "all_frames_valid": total_frames == valid_frames,
        "scenarios": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CARLA-independent GroundTruthBackend sensor-flow benchmark")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", default="artifacts/ground_truth_benchmark.json")
    args = parser.parse_args()
    scenarios = ScenarioGenerator().generate_suite(args.count, seed=args.seed)
    report = run_ground_truth_benchmark(scenarios, frame_count=args.frames)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("scenario_count", "frame_count", "valid_contract_frames", "all_frames_valid")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
