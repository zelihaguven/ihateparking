"""Dynamic-obstacle prediction, replanning and safety braking.

The module is simulator-independent. It consumes the planner-facing
PerceptionFrame and produces an explicit SafetyDecision plus an optional
replanned Hybrid A* path. The safety monitor is deliberately conservative:
uncertain dynamic tracks can trigger a replan or brake, never an acceleration.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

from .ai_perception import ObstacleEstimate
from .geometry import Pose, distance, oriented_box, polygons_intersect, vehicle_polygon, wrap_angle
from .planner import HybridAStar, PlanResult, PlannerConfig
from .stack_contracts import AutonomyMode, Gear, PerceptionFrame
from .world import ParkingWorld, RectangleObstacle


class SafetyAction(str, Enum):
    PROCEED = "proceed"
    REPLAN = "replan"
    BRAKE = "brake"


@dataclass(frozen=True)
class SafetyConfig:
    reaction_time_s: float = 0.35
    braking_deceleration_mps2: float = 2.8
    safety_margin_m: float = 0.45
    prediction_horizon_s: float = 2.0
    prediction_step_s: float = 0.10
    emergency_ttc_s: float = 0.75
    replan_ttc_s: float = 1.75
    dynamic_confidence_floor: float = 0.35
    max_obstacle_speed_mps: float = 3.5


@dataclass(frozen=True)
class SafetyDecision:
    action: SafetyAction
    safe: bool
    reason: str
    minimum_clearance_m: float
    braking_distance_m: float
    time_to_collision_s: float | None
    predicted_collisions: tuple[str, ...] = ()
    emergency_stop: bool = False

    @property
    def should_replan(self) -> bool:
        return self.action is SafetyAction.REPLAN

    @property
    def should_brake(self) -> bool:
        return self.action is SafetyAction.BRAKE


@dataclass(frozen=True)
class ReplanResult:
    decision: SafetyDecision
    plan: PlanResult | None
    replanned: bool
    plan_message: str
    control_mode: AutonomyMode


def _advance_pose(pose: Pose, speed_mps: float, t_s: float) -> Pose:
    return Pose(
        pose.x + speed_mps * math.cos(pose.yaw) * t_s,
        pose.y + speed_mps * math.sin(pose.yaw) * t_s,
        pose.yaw,
    )


def _signed_speed(speed_mps: float, gear: Gear) -> float:
    speed = abs(speed_mps)
    if gear is Gear.REVERSE:
        return -speed
    if gear is Gear.DRIVE:
        return speed
    return 0.0


class DynamicSafetyMonitor:
    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()

    def braking_distance(self, speed_mps: float) -> float:
        speed = abs(speed_mps)
        return speed * self.config.reaction_time_s + speed * speed / (2.0 * self.config.braking_deceleration_mps2) + self.config.safety_margin_m

    def evaluate(
        self,
        world: ParkingWorld,
        vehicle_pose: Pose,
        speed_mps: float,
        gear: Gear,
        perception: PerceptionFrame,
    ) -> SafetyDecision:
        cfg = self.config
        signed_speed = _signed_speed(speed_mps, gear)
        braking_distance = self.braking_distance(speed_mps)
        minimum_clearance = math.inf
        earliest_collision: float | None = None
        predicted_ids: list[str] = []
        dynamic_tracks = [item for item in perception.obstacles if item.dynamic and item.confidence >= cfg.dynamic_confidence_floor]
        for obstacle in dynamic_tracks:
            obstacle_speed = min(abs(obstacle.velocity_mps), cfg.max_obstacle_speed_mps)
            obstacle_collision_t: float | None = None
            for step in range(int(cfg.prediction_horizon_s / cfg.prediction_step_s) + 1):
                t = step * cfg.prediction_step_s
                ego_prediction = _advance_pose(vehicle_pose, signed_speed, t)
                obstacle_prediction = _advance_pose(obstacle.pose, obstacle_speed, t)
                vehicle_shape = vehicle_polygon(ego_prediction, world.vehicle.length, world.vehicle.width, world.vehicle.rear_overhang)
                obstacle_shape = oriented_box((obstacle_prediction.x, obstacle_prediction.y), obstacle.length_m, obstacle.width_m, obstacle_prediction.yaw)
                if polygons_intersect(vehicle_shape, obstacle_shape, margin=cfg.safety_margin_m):
                    obstacle_collision_t = t
                    break
            center_clearance = distance(vehicle_pose, obstacle.pose) - .5 * math.hypot(world.vehicle.length, world.vehicle.width) - .5 * math.hypot(obstacle.length_m, obstacle.width_m) - cfg.safety_margin_m
            minimum_clearance = min(minimum_clearance, center_clearance)
            if obstacle_collision_t is not None:
                earliest_collision = obstacle_collision_t if earliest_collision is None else min(earliest_collision, obstacle_collision_t)
                predicted_ids.append(obstacle.track_id)
        if math.isinf(minimum_clearance):
            minimum_clearance = 999.0
        if earliest_collision is not None and earliest_collision <= cfg.emergency_ttc_s:
            return SafetyDecision(SafetyAction.BRAKE, False, "dynamic obstacle inside emergency TTC", minimum_clearance, braking_distance, earliest_collision, tuple(predicted_ids), True)
        if earliest_collision is not None and earliest_collision <= cfg.replan_ttc_s:
            return SafetyDecision(SafetyAction.REPLAN, False, "predicted dynamic collision requires replanning", minimum_clearance, braking_distance, earliest_collision, tuple(predicted_ids), False)
        if minimum_clearance <= braking_distance:
            return SafetyDecision(SafetyAction.BRAKE, False, "clearance is shorter than stopping distance", minimum_clearance, braking_distance, earliest_collision, tuple(predicted_ids), True)
        if minimum_clearance <= braking_distance + cfg.safety_margin_m * 2.0:
            return SafetyDecision(SafetyAction.REPLAN, False, "dynamic obstacle is approaching the safety envelope", minimum_clearance, braking_distance, earliest_collision, tuple(predicted_ids), False)
        return SafetyDecision(SafetyAction.PROCEED, True, "predicted dynamic tracks are outside the safety envelope", minimum_clearance, braking_distance, earliest_collision, tuple(predicted_ids), False)


class DynamicReplanner:
    """Safety-aware replanning wrapper around Hybrid A*."""

    def __init__(self, planner_config: PlannerConfig | None = None, safety_config: SafetyConfig | None = None):
        self.planner_config = planner_config or PlannerConfig(max_expansions=20_000)
        self.monitor = DynamicSafetyMonitor(safety_config)
        self.replan_count = 0

    def assess_and_replan(
        self,
        world: ParkingWorld,
        vehicle_pose: Pose,
        speed_mps: float,
        gear: Gear,
        perception: PerceptionFrame,
        current_plan: PlanResult | None = None,
    ) -> ReplanResult:
        decision = self.monitor.evaluate(world, vehicle_pose, speed_mps, gear, perception)
        if decision.should_brake:
            return ReplanResult(decision, current_plan, False, "safety supervisor commands zero speed", AutonomyMode.SAFE_STOP)
        if not decision.should_replan:
            return ReplanResult(decision, current_plan, False, "current plan remains valid", AutonomyMode.EXECUTE)
        replanning_world = self._world_with_dynamic_obstacles(world, perception)
        plan = HybridAStar(replanning_world, self.planner_config).plan(start=vehicle_pose)
        self.replan_count += 1
        if not plan.success:
            emergency = SafetyDecision(
                SafetyAction.BRAKE,
                False,
                f"replan failed: {plan.message}",
                decision.minimum_clearance_m,
                decision.braking_distance_m,
                decision.time_to_collision_s,
                decision.predicted_collisions,
                True,
            )
            return ReplanResult(emergency, current_plan, False, plan.message, AutonomyMode.SAFE_STOP)
        return ReplanResult(decision, plan, True, "dynamic obstacle-aware plan generated", AutonomyMode.PLAN_MANOEUVRE)

    @staticmethod
    def _world_with_dynamic_obstacles(world: ParkingWorld, perception: PerceptionFrame) -> ParkingWorld:
        dynamic_obstacles = [RectangleObstacle(
            name=f"dynamic:{item.track_id}",
            center=(item.pose.x, item.pose.y),
            length=item.length_m,
            width=item.width_m,
            yaw=item.pose.yaw,
        ) for item in perception.obstacles if item.dynamic]
        return ParkingWorld(
            bounds=world.bounds,
            obstacles=list(world.obstacles) + dynamic_obstacles,
            slot=world.slot,
            start_pose=world.start_pose,
            vehicle=world.vehicle,
            collision_margin=world.collision_margin,
        )


def safety_command(decision: SafetyDecision, timestamp_s: float, gear: Gear = Gear.PARK) -> dict[str, object]:
    """Serialize the supervisor output for a dashboard or a vehicle adapter."""
    return {
        "timestamp_s": timestamp_s,
        "mode": AutonomyMode.SAFE_STOP.value if decision.should_brake else AutonomyMode.EXECUTE.value,
        "target_speed_mps": 0.0 if decision.should_brake else None,
        "gear": Gear.PARK.value if decision.should_brake else gear.value,
        "action": decision.action.value,
        "reason": decision.reason,
        "emergency_stop": decision.emergency_stop,
    }
