"""Seeded parking scenario generation for CARLA and offline replay.

The generator is simulator-agnostic: it emits a serializable scenario contract
and a compatible 2-D ParkingWorld. A CARLA adapter can consume the same record
without coupling scenario design to the CARLA Python package.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
import random
from typing import Iterable

from .geometry import Pose
from .world import ParkingSlot, ParkingWorld, RectangleObstacle, VehicleSpec


class ScenarioKind(str, Enum):
    PERPENDICULAR_FORWARD = "perpendicular_forward"
    PERPENDICULAR_REVERSE = "perpendicular_reverse"
    PARALLEL_REVERSE = "parallel_reverse"
    ANGLED_FORWARD = "angled_forward"


@dataclass(frozen=True)
class WeatherSpec:
    preset: str = "clear_noon"
    rain: float = 0.0
    fog: float = 0.0
    wetness: float = 0.0
    sun_altitude_deg: float = 55.0


@dataclass(frozen=True)
class SensorSuiteSpec:
    rgb_front: bool = True
    rgb_rear: bool = True
    depth_front: bool = True
    lidar: bool = True
    imu: bool = True
    wheel_odometry: bool = True
    camera_width: int = 1280
    camera_height: int = 720
    camera_fov_deg: float = 100.0
    perception_hz: float = 12.0
    dropout_probability: float = 0.0
    pixel_noise_std: float = 0.0


@dataclass(frozen=True)
class DynamicActorSpec:
    actor_id: str
    kind: str
    start_pose: Pose
    velocity_mps: float
    path: str
    spawn_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "actor_id": self.actor_id,
            "kind": self.kind,
            "start_pose": asdict(self.start_pose),
            "velocity_mps": self.velocity_mps,
            "path": self.path,
            "spawn_time_s": self.spawn_time_s,
        }


@dataclass(frozen=True)
class ScenarioConfig:
    seed: int = 0
    kind: ScenarioKind = ScenarioKind.PERPENDICULAR_REVERSE
    map_name: str = "Town04"
    weather: WeatherSpec = WeatherSpec()
    sensors: SensorSuiteSpec = SensorSuiteSpec()
    dynamic_actor_count: int = 0
    occlusion_level: float = 0.0
    calibration_error_m: float = 0.0
    calibration_error_deg: float = 0.0
    slot_width_m: float | None = None
    initial_pose_noise_m: float = 0.0
    initial_yaw_noise_deg: float = 0.0


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    seed: int
    kind: ScenarioKind
    map_name: str
    bounds: tuple[float, float, float, float]
    vehicle: VehicleSpec
    start_pose: Pose
    target_slot: ParkingSlot
    static_obstacles: tuple[RectangleObstacle, ...]
    dynamic_actors: tuple[DynamicActorSpec, ...]
    weather: WeatherSpec
    sensors: SensorSuiteSpec
    occlusion_level: float
    calibration_error_m: float
    calibration_error_deg: float
    expected_gear_sequence: tuple[str, ...]

    def to_world(self) -> ParkingWorld:
        return ParkingWorld(
            bounds=self.bounds,
            obstacles=list(self.static_obstacles),
            slot=self.target_slot,
            start_pose=self.start_pose,
            vehicle=self.vehicle,
        )

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "kind": self.kind.value,
            "map_name": self.map_name,
            "bounds": list(self.bounds),
            "vehicle": asdict(self.vehicle),
            "start_pose": asdict(self.start_pose),
            "target_slot": asdict(self.target_slot),
            "static_obstacles": [asdict(item) for item in self.static_obstacles],
            "dynamic_actors": [item.to_dict() for item in self.dynamic_actors],
            "weather": asdict(self.weather),
            "sensors": asdict(self.sensors),
            "occlusion_level": self.occlusion_level,
            "calibration_error_m": self.calibration_error_m,
            "calibration_error_deg": self.calibration_error_deg,
            "expected_gear_sequence": list(self.expected_gear_sequence),
        }


class ScenarioGenerator:
    """Generate replayable parking scenarios with no global RNG state."""

    def generate(self, config: ScenarioConfig) -> ScenarioSpec:
        rng = random.Random(config.seed)
        vehicle = VehicleSpec()
        bounds = (0.0, 20.0, 0.0, 16.0)
        slot_width = config.slot_width_m or rng.uniform(2.45, 2.85)
        lateral_offset = rng.uniform(-0.45, 0.45)
        lane_y = rng.uniform(3.0, 4.0)
        dynamic_actors = self._dynamic_actors(config, rng)

        if config.kind in (ScenarioKind.PERPENDICULAR_FORWARD, ScenarioKind.PERPENDICULAR_REVERSE):
            slot_center = (10.0 + lateral_offset, 10.0)
            slot_yaw = math.pi / 2.0
            slot = ParkingSlot(slot_center, length=5.0, width=slot_width, yaw=slot_yaw)
            static = (
                RectangleObstacle("parked_left", (5.0 + lateral_offset, 10.0), 4.8, 2.0, slot_yaw),
                RectangleObstacle("parked_right", (15.0 + lateral_offset, 10.0), 4.8, 2.0, slot_yaw),
                RectangleObstacle("back_curb", (10.0 + lateral_offset, 13.8), 16.0, 0.28, 0.0),
            )
            if config.kind is ScenarioKind.PERPENDICULAR_REVERSE:
                start = Pose(16.5 + lateral_offset, lane_y, math.pi)
                gear_sequence = ("drive", "reverse", "drive")
            else:
                start = Pose(3.0 + lateral_offset, lane_y, 0.0)
                gear_sequence = ("drive",)
        elif config.kind is ScenarioKind.PARALLEL_REVERSE:
            slot_center = (10.0 + lateral_offset, 5.5)
            slot_yaw = 0.0
            slot = ParkingSlot(slot_center, length=5.0, width=2.35, yaw=slot_yaw)
            static = (
                RectangleObstacle("parked_front", (14.7 + lateral_offset, 5.5), 4.6, 2.1, 0.0),
                RectangleObstacle("parked_rear", (5.3 + lateral_offset, 5.5), 4.6, 2.1, 0.0),
                RectangleObstacle("curb", (10.0 + lateral_offset, 7.1), 16.0, 0.22, 0.0),
            )
            start = Pose(16.8 + lateral_offset, 3.0 + rng.uniform(-.2, .2), math.pi)
            gear_sequence = ("drive", "reverse", "drive", "reverse", "drive")
        else:
            slot_center = (10.0 + lateral_offset, 10.0)
            slot_yaw = math.radians(rng.choice([45.0, 55.0, 65.0]))
            slot = ParkingSlot(slot_center, length=5.0, width=slot_width, yaw=slot_yaw)
            direction_x, direction_y = math.cos(slot_yaw), math.sin(slot_yaw)
            normal_x, normal_y = -direction_y, direction_x
            static = (
                RectangleObstacle("parked_left", (slot_center[0] - normal_x * 5.0, slot_center[1] - normal_y * 5.0), 4.8, 2.0, slot_yaw),
                RectangleObstacle("parked_right", (slot_center[0] + normal_x * 5.0, slot_center[1] + normal_y * 5.0), 4.8, 2.0, slot_yaw),
                RectangleObstacle("back_curb", (slot_center[0] + direction_x * 4.6, slot_center[1] + direction_y * 4.6), 13.0, .25, slot_yaw + math.pi / 2.0),
            )
            start = Pose(3.0 + lateral_offset, lane_y, 0.0)
            gear_sequence = ("drive", "reverse" if config.kind is ScenarioKind.ANGLED_FORWARD else "drive")

        start = self._perturb_pose(start, config, rng)
        return ScenarioSpec(
            scenario_id=f"{config.kind.value}_{config.seed:06d}",
            seed=config.seed,
            kind=config.kind,
            map_name=config.map_name,
            bounds=bounds,
            vehicle=vehicle,
            start_pose=start,
            target_slot=slot,
            static_obstacles=tuple(static),
            dynamic_actors=dynamic_actors,
            weather=config.weather,
            sensors=config.sensors,
            occlusion_level=config.occlusion_level,
            calibration_error_m=config.calibration_error_m,
            calibration_error_deg=config.calibration_error_deg,
            expected_gear_sequence=gear_sequence,
        )

    def generate_suite(self, count: int, seed: int = 0, kinds: Iterable[ScenarioKind] | None = None) -> list[ScenarioSpec]:
        if count < 1:
            raise ValueError("count must be positive")
        selected = tuple(kinds or tuple(ScenarioKind))
        if not selected:
            raise ValueError("at least one scenario kind is required")
        return [self.generate(ScenarioConfig(seed=seed + index, kind=selected[index % len(selected)])) for index in range(count)]

    def write_suite(self, output: str | Path, count: int, seed: int = 0) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [item.to_dict() for item in self.generate_suite(count, seed)]
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _perturb_pose(pose: Pose, config: ScenarioConfig, rng: random.Random) -> Pose:
        return Pose(
            pose.x + rng.uniform(-config.initial_pose_noise_m, config.initial_pose_noise_m),
            pose.y + rng.uniform(-config.initial_pose_noise_m, config.initial_pose_noise_m),
            pose.yaw + math.radians(rng.uniform(-config.initial_yaw_noise_deg, config.initial_yaw_noise_deg)),
        )

    @staticmethod
    def _dynamic_actors(config: ScenarioConfig, rng: random.Random) -> tuple[DynamicActorSpec, ...]:
        actors = []
        for index in range(config.dynamic_actor_count):
            actors.append(DynamicActorSpec(
                actor_id=f"dynamic_{index:02d}",
                kind=rng.choice(["pedestrian", "shopping_cart", "vehicle"]),
                start_pose=Pose(rng.uniform(4.0, 16.0), rng.uniform(1.0, 12.0), rng.uniform(-math.pi, math.pi)),
                velocity_mps=rng.uniform(.4, 1.8),
                path=rng.choice(["cross_lane", "cross_slot_aisle", "pause_then_cross"]),
                spawn_time_s=rng.uniform(1.0, 5.0),
            ))
        return tuple(actors)
