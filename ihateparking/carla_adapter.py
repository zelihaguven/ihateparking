"""Optional CARLA bridge for world, vehicle and sensor spawning.

The repository remains importable without CARLA. Installing CARLA and running a
server are explicit environment steps; this module keeps CARLA actor handles at
the edge and emits project-level SensorFrame contracts inward.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from queue import Empty, Queue
from typing import Any, Callable

from .scenario_generator import ScenarioSpec
from .stack_contracts import SensorFrame


class CarlaUnavailableError(RuntimeError):
    """Raised when a CARLA operation is requested without the Python API."""


@dataclass(frozen=True)
class CarlaRuntimeConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout_s: float = 5.0
    synchronous_mode: bool = True
    fixed_delta_s: float = 0.05
    record_dir: str | None = None
    vehicle_blueprint: str = "vehicle.tesla.model3"


@dataclass(frozen=True)
class CarlaSensorSpec:
    name: str
    type_id: str
    transform: dict[str, float]
    attributes: dict[str, str]


@dataclass(frozen=True)
class CarlaSpawnPlan:
    scenario_id: str
    map_name: str
    vehicle_blueprint: str
    vehicle_transform: dict[str, float]
    sensors: tuple[CarlaSensorSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "map_name": self.map_name,
            "vehicle_blueprint": self.vehicle_blueprint,
            "vehicle_transform": dict(self.vehicle_transform),
            "sensors": [asdict(sensor) for sensor in self.sensors],
        }


@dataclass(frozen=True)
class SensorPacket:
    sensor_name: str
    timestamp_s: float
    frame_number: int
    payload: Any


class SensorCollector:
    """Thread-safe callback collector for CARLA sensor.listen streams."""

    def __init__(self, sensor_names: tuple[str, ...] = ()):
        self._queues: dict[str, Queue[SensorPacket]] = {name: Queue(maxsize=4) for name in sensor_names}

    def register(self, sensor_name: str) -> None:
        self._queues.setdefault(sensor_name, Queue(maxsize=4))

    def callback(self, sensor_name: str) -> Callable[[Any], None]:
        self.register(sensor_name)

        def receive(data: Any) -> None:
            packet = SensorPacket(
                sensor_name=sensor_name,
                timestamp_s=float(getattr(data, "timestamp", 0.0)),
                frame_number=int(getattr(data, "frame", -1)),
                payload=data,
            )
            queue = self._queues[sensor_name]
            try:
                queue.put_nowait(packet)
            except Exception:
                try:
                    queue.get_nowait()
                except Empty:
                    pass
                queue.put_nowait(packet)

        return receive

    def poll(self, sensor_name: str, timeout_s: float = 0.0) -> SensorPacket | None:
        self.register(sensor_name)
        try:
            return self._queues[sensor_name].get(timeout=timeout_s)
        except Empty:
            return None

    def counts(self) -> dict[str, int]:
        return {name: queue.qsize() for name, queue in self._queues.items()}


class CarlaScenarioAdapter:
    """CARLA runtime boundary for one generated scenario."""

    def __init__(self, runtime: CarlaRuntimeConfig = CarlaRuntimeConfig()):
        self.runtime = runtime
        self._client: Any = None
        self._world: Any = None
        self._vehicle: Any = None
        self._actors: list[Any] = []
        self.collector: SensorCollector | None = None

    @staticmethod
    def available() -> bool:
        try:
            import carla  # type: ignore  # noqa: F401
        except ImportError:
            return False
        return True

    def connect(self, scenario: ScenarioSpec | None = None) -> Any:
        try:
            import carla  # type: ignore
        except ImportError as exc:
            raise CarlaUnavailableError(
                "CARLA Python API is not installed. Install it in the simulator environment "
                "and start a CARLA server before using CarlaScenarioAdapter."
            ) from exc
        self._client = carla.Client(self.runtime.host, self.runtime.port)
        self._client.set_timeout(self.runtime.timeout_s)
        self._world = self._client.get_world()
        if scenario is not None:
            current_map = getattr(getattr(self._world, "get_map", lambda: None)(), "name", "")
            if scenario.map_name not in current_map:
                self._world = self._client.load_world(scenario.map_name)
            self._apply_world_settings(scenario)
        return self._world

    def configure_world(self, scenario: ScenarioSpec) -> dict[str, Any]:
        return {
            "map_name": scenario.map_name,
            "weather": asdict(scenario.weather),
            "synchronous_mode": self.runtime.synchronous_mode,
            "fixed_delta_s": self.runtime.fixed_delta_s,
            "sensor_suite": asdict(scenario.sensors),
            "scenario_id": scenario.scenario_id,
        }

    def build_spawn_plan(self, scenario: ScenarioSpec) -> CarlaSpawnPlan:
        sensors: list[CarlaSensorSpec] = []
        camera_attributes = {
            "image_size_x": str(scenario.sensors.camera_width),
            "image_size_y": str(scenario.sensors.camera_height),
            "fov": str(scenario.sensors.camera_fov_deg),
            "sensor_tick": str(1.0 / scenario.sensors.perception_hz),
        }
        if scenario.sensors.rgb_front:
            sensors.append(CarlaSensorSpec("rgb_front", "sensor.camera.rgb", {"x": 1.45, "y": 0.0, "z": 1.35, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}, camera_attributes))
        if scenario.sensors.rgb_rear:
            sensors.append(CarlaSensorSpec("rgb_rear", "sensor.camera.rgb", {"x": -1.15, "y": 0.0, "z": 1.25, "roll": 0.0, "pitch": 0.0, "yaw": 180.0}, camera_attributes))
        if scenario.sensors.depth_front:
            sensors.append(CarlaSensorSpec("depth_front", "sensor.camera.depth", {"x": 1.45, "y": 0.0, "z": 1.35, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}, camera_attributes))
        if scenario.sensors.lidar:
            sensors.append(CarlaSensorSpec("lidar", "sensor.lidar.ray_cast", {"x": 0.0, "y": 0.0, "z": 1.75, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}, {
                "channels": "64", "range": "50", "points_per_second": "560000", "rotation_frequency": str(1.0 / self.runtime.fixed_delta_s), "upper_fov": "10", "lower_fov": "-30",
            }))
        if scenario.sensors.imu:
            sensors.append(CarlaSensorSpec("imu", "sensor.other.imu", {"x": 0.0, "y": 0.0, "z": 0.8, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}, {"sensor_tick": str(self.runtime.fixed_delta_s)}))
        return CarlaSpawnPlan(
            scenario_id=scenario.scenario_id,
            map_name=scenario.map_name,
            vehicle_blueprint=self.runtime.vehicle_blueprint,
            vehicle_transform={"x": scenario.start_pose.x, "y": scenario.start_pose.y, "z": 0.5, "yaw": math.degrees(scenario.start_pose.yaw), "pitch": 0.0, "roll": 0.0},
            sensors=tuple(sensors),
        )

    def spawn_vehicle_and_sensors(self, scenario: ScenarioSpec, collector: SensorCollector | None = None) -> CarlaSpawnPlan:
        self._ensure_scenario_world(scenario)
        import carla  # type: ignore
        plan = self.build_spawn_plan(scenario)
        blueprint_library = self._world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(plan.vehicle_blueprint)
        vehicle_bp.set_attribute("role_name", "ihateparking_ego")
        vehicle_tf = self._transform_from_dict(plan.vehicle_transform)
        self._vehicle = self._world.try_spawn_actor(vehicle_bp, vehicle_tf)
        if self._vehicle is None:
            raise RuntimeError(f"CARLA could not spawn ego vehicle for {scenario.scenario_id}")
        self._actors.append(self._vehicle)
        self.collector = collector or SensorCollector(tuple(sensor.name for sensor in plan.sensors))
        for spec in plan.sensors:
            sensor_bp = blueprint_library.find(spec.type_id)
            for key, value in spec.attributes.items():
                if sensor_bp.has_attribute(key):
                    sensor_bp.set_attribute(key, value)
            sensor = self._world.spawn_actor(sensor_bp, self._transform_from_dict(spec.transform), attach_to=self._vehicle)
            self._actors.append(sensor)
            sensor.listen(self.collector.callback(spec.name))
        return plan

    @staticmethod
    def _transform_from_dict(values: dict[str, float]) -> Any:
        import carla  # type: ignore
        return carla.Transform(
            carla.Location(x=values.get("x", 0.0), y=values.get("y", 0.0), z=values.get("z", 0.0)),
            carla.Rotation(roll=values.get("roll", 0.0), pitch=values.get("pitch", 0.0), yaw=values.get("yaw", 0.0)),
        )

    def sensor_frame(self, timestamp_s: float, frame_id: str, paths: dict[str, str] | None = None, odometry: dict[str, float] | None = None) -> SensorFrame:
        paths = paths or {}
        return SensorFrame(
            timestamp_s=timestamp_s,
            frame_id=frame_id,
            rgb_uri=paths.get("rgb_front"),
            depth_uri=paths.get("depth_front"),
            lidar_uri=paths.get("lidar"),
            odometry=odometry or {},
        )

    def tick(self) -> int:
        if self._world is None:
            raise RuntimeError("CARLA world is not connected")
        return int(self._world.tick())

    def _ensure_scenario_world(self, scenario: ScenarioSpec) -> None:
        if self._world is None:
            self.connect(scenario)
            return
        current_map = getattr(getattr(self._world, "get_map", lambda: None)(), "name", "")
        if scenario.map_name not in current_map:
            self._world = self._client.load_world(scenario.map_name)
        self._apply_world_settings(scenario)

    def close(self) -> None:
        for actor in reversed(self._actors):
            try:
                actor.destroy()
            except Exception:
                pass
        self._actors.clear()
        self._vehicle = None
        self._world = None
        self._client = None

    def _apply_world_settings(self, scenario: ScenarioSpec) -> None:
        settings = self._world.get_settings()
        settings.synchronous_mode = self.runtime.synchronous_mode
        settings.fixed_delta_seconds = self.runtime.fixed_delta_s if self.runtime.synchronous_mode else None
        self._world.apply_settings(settings)
        try:
            import carla  # type: ignore
            weather_name = scenario.weather.preset
            weather = getattr(carla.WeatherParameters, self._weather_name(weather_name), carla.WeatherParameters.ClearNoon)
            self._world.set_weather(weather)
        except (ImportError, AttributeError):
            pass

    @staticmethod
    def _weather_name(preset: str) -> str:
        return {
            "clear_noon": "ClearNoon",
            "cloudy_noon": "CloudyNoon",
            "wet_noon": "WetNoon",
            "hard_rain_noon": "HardRainNoon",
            "soft_rain_noon": "SoftRainNoon",
            "clear_sunset": "ClearSunset",
        }.get(preset, "ClearNoon")

    def __enter__(self) -> "CarlaScenarioAdapter":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
