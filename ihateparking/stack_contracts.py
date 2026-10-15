"""Typed contracts shared by AI perception, planning, control and safety layers.

These dataclasses deliberately avoid ROS2 imports so the core can be replayed in
plain Python. A later ROS2 adapter can map them to messages without changing the
planner or safety supervisor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .geometry import Pose


class Gear(str, Enum):
    PARK = "park"
    DRIVE = "drive"
    REVERSE = "reverse"


class AutonomyMode(str, Enum):
    SEARCH_SLOT = "search_slot"
    APPROACH_SLOT = "approach_slot"
    PLAN_MANOEUVRE = "plan_manoeuvre"
    EXECUTE = "execute"
    VERIFY_PARKED = "verify_parked"
    COMPLETE = "complete"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class SensorFrame:
    timestamp_s: float
    frame_id: str
    rgb_uri: str | None = None
    depth_uri: str | None = None
    lidar_uri: str | None = None
    odometry: dict[str, float] = field(default_factory=dict)
    steering_angle_rad: float | None = None


@dataclass(frozen=True)
class ObstacleEstimate:
    track_id: str
    kind: str
    pose: Pose
    length_m: float
    width_m: float
    confidence: float
    uncertainty_m: float
    dynamic: bool = False
    velocity_mps: float = 0.0


@dataclass(frozen=True)
class ParkingSlotEstimate:
    slot_id: str
    pose: Pose
    length_m: float
    width_m: float
    confidence: float
    uncertainty_m: float
    source: str = "ai_perception"


@dataclass(frozen=True)
class PerceptionFrame:
    timestamp_s: float
    frame_id: str
    slots: tuple[ParkingSlotEstimate, ...] = ()
    obstacles: tuple[ObstacleEstimate, ...] = ()
    drivable_space_confidence: float = 0.0
    out_of_distribution_score: float = 1.0
    model_name: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VehicleState:
    timestamp_s: float
    pose: Pose
    speed_mps: float
    steering_angle_rad: float
    steering_rate_radps: float = 0.0
    gear: Gear = Gear.PARK
    covariance: tuple[float, ...] = ()


@dataclass(frozen=True)
class TrajectoryPoint:
    timestamp_s: float
    pose: Pose
    speed_mps: float
    steering_angle_rad: float
    gear: Gear


@dataclass(frozen=True)
class PlannedTrajectory:
    created_at_s: float
    target_slot_id: str
    points: tuple[TrajectoryPoint, ...]
    minimum_clearance_m: float
    planning_time_ms: float
    planner_name: str = "HybridAStar"


@dataclass(frozen=True)
class SafetyStatus:
    safe: bool
    reason: str
    minimum_clearance_m: float
    stale_input: bool = False
    emergency_stop: bool = False


@dataclass(frozen=True)
class ControlCommand:
    timestamp_s: float
    mode: AutonomyMode
    target_speed_mps: float
    target_steering_angle_rad: float
    gear: Gear
    safety: SafetyStatus
    reason: str
