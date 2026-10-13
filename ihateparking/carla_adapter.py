"""Optional CARLA bridge for Phase 1.

The repository remains importable without CARLA. Installing CARLA and running a
server are explicit environment steps; this module gives the real simulator a
stable boundary instead of leaking CARLA types into the planner core.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


class CarlaScenarioAdapter:
    """Spawn/configuration boundary for one generated scenario.

    The methods intentionally return plain dictionaries or project contracts;
    CARLA actor handles stay inside this adapter.
    """

    def __init__(self, runtime: CarlaRuntimeConfig = CarlaRuntimeConfig()):
        self.runtime = runtime
        self._client: Any = None
        self._world: Any = None
        self._actors: list[Any] = []

    @staticmethod
    def available() -> bool:
        try:
            import carla  # type: ignore  # noqa: F401
        except ImportError:
            return False
        return True

    def connect(self) -> Any:
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
        return self._world

    def configure_world(self, scenario: ScenarioSpec) -> dict[str, Any]:
        """Return a serializable configuration plan before spawning actors."""
        return {
            "map_name": scenario.map_name,
            "weather": scenario.weather.__dict__,
            "synchronous_mode": self.runtime.synchronous_mode,
            "fixed_delta_s": self.runtime.fixed_delta_s,
            "sensor_suite": scenario.sensors.__dict__,
            "scenario_id": scenario.scenario_id,
        }

    def sensor_frame(self, timestamp_s: float, frame_id: str, paths: dict[str, str] | None = None) -> SensorFrame:
        paths = paths or {}
        return SensorFrame(
            timestamp_s=timestamp_s,
            frame_id=frame_id,
            rgb_uri=paths.get("rgb_front"),
            depth_uri=paths.get("depth_front"),
            lidar_uri=paths.get("lidar"),
        )

    def close(self) -> None:
        for actor in self._actors:
            try:
                actor.destroy()
            except Exception:
                pass
        self._actors.clear()
        self._world = None
        self._client = None

    def __enter__(self) -> "CarlaScenarioAdapter":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
