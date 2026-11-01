"""MetaDrive as an independent simulator for the parking stack (PRD step 3).

MetaDrive (Bullet physics, Panda3D) builds the world: a parking-lot road
block with perpendicular spaces, parked cars in some of them and kerbs
behind each row. This module only translates at the edge:

- ``MetaDriveLot.scan`` turns MetaDrive's LiDAR into the stack's
  ``LidarScan`` (same contract as the built-in simulated LiDAR);
- ``MetaDriveLot.pose`` turns MetaDrive's vehicle state (body centre)
  into the stack's rear-axle ``Pose``;
- ``MetaDriveLot.spaces`` is the ground truth, for scoring only.

MetaDrive 0.4.3 needs Python < 3.12, so this runs in ``.venv-metadrive``
(see docs/metadrive.md); importing the module does not import MetaDrive.

MetaDrive's LiDAR sees vehicles and traffic objects, not sidewalks, so the
kerbs behind the rows are static barriers, as a kerb or wall is in the
built-in simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np

from .geometry import Pose, wrap_angle
from .lidar import LidarConfig, LidarScan
from .world import VehicleSpec

# MetaDrive's DefaultVehicle: 4.515 x 1.852 m, wheels 1.05234 m ahead of and
# 1.4166 m behind the body centre, 40 degrees maximum steering.
METADRIVE_VEHICLE = VehicleSpec(
    length=4.515,
    width=1.852,
    wheelbase=1.05234 + 1.4166,
    rear_overhang=4.515 / 2 - 1.4166,
    max_steering=math.radians(40.0),
)
SPACE_WIDTH_M = 2.7  # MetaDrive derives the space width from the lane width (default 3.5 m)
KERB_PIECE_M = 2.0  # MetaDrive TrafficBarrier length, which lies square to its heading
KERB_OFFSET_M = 0.3  # behind the end of a space
SPEED_KP = 0.6
SPEED_KI = 0.3
SPEED_INTEGRAL_LIMIT = 2.0


@dataclass(frozen=True)
class TruthSpace:
    space_id: str
    center: tuple[float, float]
    yaw: float  # heading into the space from the aisle
    length: float
    width: float
    occupied: bool


class MetaDriveLot:
    """One MetaDrive parking lot with parked cars and kerbs, ego in the entrance road."""

    def __init__(self, seed: int, occupancy: float = 0.5, spaces_per_side: int = 4, jitter: bool = True):
        from metadrive.component.static_object.traffic_object import TrafficBarrier
        from metadrive.component.vehicle.vehicle_type import DefaultVehicle
        from metadrive.envs.marl_envs.marl_parking_lot import MAParkingLotMap
        from metadrive.envs.metadrive_env import MetaDriveEnv
        from metadrive.manager.pg_map_manager import PGMapManager

        map_config = dict(exit_length=20, lane_num=1, parking_space_num=2 * spaces_per_side, lane_width=SPACE_WIDTH_M)

        class _LotMapManager(PGMapManager):
            def reset(self):
                if len(self.spawned_objects) == 0:
                    lot_map = self.spawn_object(MAParkingLotMap, map_config=map_config, random_seed=None)
                else:
                    lot_map = list(self.spawned_objects.values())[0]
                self.load_map(lot_map)

        class _LotEnv(MetaDriveEnv):
            def setup_engine(self):
                super().setup_engine()
                self.engine.update_manager("map_manager", _LotMapManager())

        self.vehicle = METADRIVE_VEHICLE
        self._speed_integral = 0.0
        self.env = _LotEnv(dict(use_render=False, num_scenarios=1, log_level=50, traffic_density=0.0, vehicle_config=dict(enable_reverse=True)))
        self.env.reset()
        rng = random.Random(seed)
        lot_map = self.env.current_map
        lot = lot_map.parking_lot
        spaces = []
        for road in lot_map.parking_space:
            lane = lot_map.road_network.get_lane((road.start_node, road.end_node, 0))
            center = lane.position(lane.length / 2.0, 0.0)
            spaces.append(TruthSpace(road.start_node, (float(center[0]), float(center[1])), float(lane.heading_theta_at(lane.length / 2.0)), float(lane.length), float(lot.parking_space_width), rng.random() < occupancy))
        self.spaces = tuple(spaces)
        engine = self.env.engine
        self.parked = []
        for space in self.spaces:
            if not space.occupied:
                continue
            along = rng.uniform(-0.3, 0.3) if jitter else 0.0
            across = rng.uniform(-0.25, 0.25) if jitter else 0.0
            # Parked nose-in, its front 0.5 m short of the back of the space.
            depth = space.length / 2.0 - self.vehicle.length / 2.0 - 0.5
            c, s = math.cos(space.yaw), math.sin(space.yaw)
            x = space.center[0] + c * (depth + along) - s * across
            y = space.center[1] + s * (depth + along) + c * across
            heading = space.yaw + (math.radians(rng.uniform(-3.0, 3.0)) if jitter else 0.0)
            car = engine.spawn_object(DefaultVehicle, vehicle_config=dict(), position=(x, y), heading=heading)
            car.set_static(True)
            self.parked.append(car)
        # A kerb behind each row: barriers along the far end of the spaces.
        self.kerbs = []
        for row in _rows(self.spaces):
            yaw = row[0].yaw
            c, s = math.cos(yaw), math.sin(yaw)
            backs = [np.array(sp.center) + (sp.length / 2.0 + KERB_OFFSET_M) * np.array([c, s]) for sp in row]
            along_row = np.array([-s, c])
            ts = sorted(float(b @ along_row) for b in backs)
            depth = float(backs[0] @ np.array([c, s]))
            for t in np.arange(ts[0] - row[0].width / 2.0 + KERB_PIECE_M / 2.0, ts[-1] + row[0].width / 2.0, KERB_PIECE_M):
                point = depth * np.array([c, s]) + t * along_row
                barrier = engine.spawn_object(TrafficBarrier, position=(float(point[0]), float(point[1])), heading_theta=yaw, static=True)
                self.kerbs.append(barrier)
        # The world changed after reset: let physics settle one step.
        self.env.step([0.0, 0.0])

    # -- the stack's view ---------------------------------------------------
    def pose(self) -> Pose:
        agent = self.env.agent
        yaw = float(agent.heading_theta)
        back = self.vehicle.length / 2.0 - self.vehicle.rear_overhang
        return Pose(float(agent.position[0]) - back * math.cos(yaw), float(agent.position[1]) - back * math.sin(yaw), wrap_angle(yaw))

    def scan(self, config: LidarConfig = LidarConfig()) -> LidarScan:
        agent = self.env.agent
        lidar = self.env.engine.get_sensor("lidar")
        cloud, _ = lidar.perceive(agent, physics_world=self.env.engine.physics_world.dynamic_world, num_lasers=config.beams, distance=config.max_range)
        cloud = np.asarray(cloud, dtype=float)
        hits = cloud < 0.999
        ranges = np.where(hits, cloud * config.max_range, config.max_range)
        angles = float(agent.heading_theta) + np.arange(config.beams) * 2.0 * math.pi / config.beams
        return LidarScan((float(agent.position[0]), float(agent.position[1])), angles, ranges, hits)

    @property
    def step_s(self) -> float:
        cfg = self.env.engine.global_config
        return float(cfg["physics_world_step_size"]) * int(cfg["decision_repeat"])

    def signed_speed(self) -> float:
        agent = self.env.agent
        return float(np.dot(agent.velocity, (math.cos(agent.heading_theta), math.sin(agent.heading_theta))))

    def drive(self, steering: float, target_speed: float) -> None:
        """One MetaDrive step towards a steering angle (rad) and a signed speed (m/s).

        MetaDrive takes a normalised steering and one throttle/brake value; a
        negative value brakes and, once stopped, reverses. A PI loop on the
        signed speed turns the stack's speed into that value.
        """
        error = target_speed - self.signed_speed()
        if target_speed == 0.0 or (self._speed_integral * error < 0):
            self._speed_integral = 0.0
        self._speed_integral = float(np.clip(self._speed_integral + error * self.step_s, -SPEED_INTEGRAL_LIMIT, SPEED_INTEGRAL_LIMIT))
        throttle = float(np.clip(SPEED_KP * error + SPEED_KI * self._speed_integral, -1.0, 1.0))
        self.env.step([float(np.clip(steering / self.vehicle.max_steering, -1.0, 1.0)), throttle])

    def crashed(self) -> bool:
        agent = self.env.agent
        return bool(agent.crash_vehicle or agent.crash_object or agent.crash_building)

    def teleport(self, pose: Pose) -> None:
        """Place the ego car (rear-axle pose); used for sensing checks, not for driving."""
        agent = self.env.agent
        back = self.vehicle.length / 2.0 - self.vehicle.rear_overhang
        agent.set_position((pose.x + back * math.cos(pose.yaw), pose.y + back * math.sin(pose.yaw)))
        agent.set_heading_theta(pose.yaw)
        agent.set_velocity((math.cos(pose.yaw), math.sin(pose.yaw)), 0.0)

    def close(self) -> None:
        self.env.close()


def _rows(spaces: tuple[TruthSpace, ...]) -> list[list[TruthSpace]]:
    """Spaces grouped by heading (one group per row)."""
    rows: list[list[TruthSpace]] = []
    for space in spaces:
        for row in rows:
            if abs(wrap_angle(space.yaw - row[0].yaw)) < 0.1:
                row.append(space)
                break
        else:
            rows.append([space])
    return rows


def space_overlap(space: TruthSpace, pose: Pose, vehicle: VehicleSpec) -> float:
    """Share of the car footprint (rear-axle ``pose``) inside the space."""
    offset = vehicle.length / 2.0 - vehicle.rear_overhang
    cx, cy = pose.x + offset * math.cos(pose.yaw), pose.y + offset * math.sin(pose.yaw)
    us, vs = np.meshgrid(np.arange(-vehicle.length / 2, vehicle.length / 2 + 1e-9, 0.2), np.arange(-vehicle.width / 2, vehicle.width / 2 + 1e-9, 0.2))
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    xs, ys = cx + c * us - s * vs, cy + s * us + c * vs
    bc, bs = math.cos(space.yaw), math.sin(space.yaw)
    dx, dy = xs - space.center[0], ys - space.center[1]
    inside = (np.abs(bc * dx + bs * dy) <= space.length / 2) & (np.abs(-bs * dx + bc * dy) <= space.width / 2)
    return float(inside.mean())


class MetaDrivePlant:
    """MetaDrive behind the mission's ``Plant`` interface.

    Unlike the built-in simulator's guard, MetaDrive reports contact after
    it happens, so any contact ends the mission as a collision.
    """

    def __init__(self, lot: MetaDriveLot, lidar: LidarConfig = LidarConfig(range_noise_std=0.0, dropout=0.0)):
        self.lot = lot
        self.vehicle = lot.vehicle
        self.dt = lot.step_s
        self.lidar = lidar
        self._last = (0.0, 0.0)

    def true_pose(self) -> Pose:
        return self.lot.pose()

    def scan(self) -> LidarScan:
        return self.lot.scan(self.lidar)

    def apply(self, steering: float, speed: float) -> bool:
        self.lot.drive(steering, speed)
        clipped = float(np.clip(steering, -self.vehicle.max_steering, self.vehicle.max_steering))
        self._last = (self.lot.signed_speed(), clipped)
        return not self.lot.crashed()

    def odometry(self) -> tuple[float, float]:
        return self._last

    def evaluate(self, pose: Pose) -> dict:
        """The space the car overlaps most, scored like the built-in simulator's bays."""
        best = max(self.lot.spaces, key=lambda space: space_overlap(space, pose, self.vehicle))
        overlap = space_overlap(best, pose, self.vehicle)
        if overlap == 0.0:
            return {"in_bay": False, "bay": None, "overlap": 0.0}
        difference = abs(wrap_angle(pose.yaw - best.yaw))
        axis_error = math.degrees(min(difference, math.pi - difference))
        offset = self.vehicle.length / 2.0 - self.vehicle.rear_overhang
        centre = np.array([pose.x + offset * math.cos(pose.yaw), pose.y + offset * math.sin(pose.yaw)]) - np.array(best.center)
        axis = np.array([math.cos(best.yaw), math.sin(best.yaw)])
        return {
            "in_bay": overlap >= 0.8 and axis_error <= 15.0 and not best.occupied,
            "bay": best.space_id,
            "bay_kind": "perpendicular",
            "bay_was_free": not best.occupied,
            "overlap": round(overlap, 3),
            "axis_error_deg": round(axis_error, 1),
            "center_error_m": round(float(np.hypot(*centre)), 3),
            "along_error_m": round(float(abs(centre @ axis)), 3),
            "lateral_error_m": round(float(abs(centre @ np.array([-axis[1], axis[0]]))), 3),
        }
