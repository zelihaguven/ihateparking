"""AI perception interface for Phase 2.

The planner consumes PerceptionFrame, never a framework-specific tensor. A
PyTorch, ONNX or TensorRT implementation can satisfy ModelBackend later without
changing planning/control code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .geometry import Pose
from .stack_contracts import (
    ObstacleEstimate,
    ParkingSlotEstimate,
    PerceptionFrame,
    SensorFrame,
)


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    version: str
    backend: str
    input_modalities: tuple[str, ...]
    classes: tuple[str, ...]


@dataclass(frozen=True)
class RawSlot:
    slot_id: str
    pose: Pose
    length_m: float
    width_m: float
    confidence: float
    uncertainty_m: float


@dataclass(frozen=True)
class RawObstacle:
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
class RawPerceptionOutput:
    slots: tuple[RawSlot, ...] = ()
    obstacles: tuple[RawObstacle, ...] = ()
    drivable_space_confidence: float = 0.0
    out_of_distribution_score: float = 1.0
    extra: Mapping[str, Any] | None = None


class ModelBackend(Protocol):
    metadata: ModelMetadata

    def predict(self, frame: SensorFrame) -> RawPerceptionOutput:
        """Run one inference step and return vehicle-frame geometry."""


class PerceptionModel(Protocol):
    metadata: ModelMetadata

    def infer(self, frame: SensorFrame) -> PerceptionFrame:
        """Return a validated, planner-facing perception frame."""


def _check_probability(value: float, label: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1, got {value}")


def validate_output(output: RawPerceptionOutput) -> None:
    _check_probability(output.drivable_space_confidence, "drivable_space_confidence")
    _check_probability(output.out_of_distribution_score, "out_of_distribution_score")
    for slot in output.slots:
        _check_probability(slot.confidence, f"slot {slot.slot_id} confidence")
        if slot.length_m <= 0 or slot.width_m <= 0 or slot.uncertainty_m < 0:
            raise ValueError(f"invalid geometry for slot {slot.slot_id}")
    for obstacle in output.obstacles:
        _check_probability(obstacle.confidence, f"obstacle {obstacle.track_id} confidence")
        if obstacle.length_m <= 0 or obstacle.width_m <= 0 or obstacle.uncertainty_m < 0:
            raise ValueError(f"invalid geometry for obstacle {obstacle.track_id}")


class NeuralPerceptionAdapter:
    """Adapt any neural backend into the stable planner-facing contract."""

    def __init__(self, backend: ModelBackend):
        self.backend = backend
        self.metadata = backend.metadata

    def infer(self, frame: SensorFrame) -> PerceptionFrame:
        raw = self.backend.predict(frame)
        validate_output(raw)
        return PerceptionFrame(
            timestamp_s=frame.timestamp_s,
            frame_id=frame.frame_id,
            slots=tuple(ParkingSlotEstimate(
                slot_id=item.slot_id,
                pose=item.pose,
                length_m=item.length_m,
                width_m=item.width_m,
                confidence=item.confidence,
                uncertainty_m=item.uncertainty_m,
            ) for item in raw.slots),
            obstacles=tuple(ObstacleEstimate(
                track_id=item.track_id,
                kind=item.kind,
                pose=item.pose,
                length_m=item.length_m,
                width_m=item.width_m,
                confidence=item.confidence,
                uncertainty_m=item.uncertainty_m,
                dynamic=item.dynamic,
                velocity_mps=item.velocity_mps,
            ) for item in raw.obstacles),
            drivable_space_confidence=raw.drivable_space_confidence,
            out_of_distribution_score=raw.out_of_distribution_score,
            model_name=f"{self.metadata.name}:{self.metadata.version}",
            extra=dict(raw.extra or {}),
        )


class GroundTruthBackend:
    """Offline benchmark backend; never use as a real vehicle perception model."""

    def __init__(self, slots: Sequence[RawSlot], obstacles: Sequence[RawObstacle]):
        self.metadata = ModelMetadata(
            name="ground_truth_replay",
            version="0",
            backend="scenario_generator",
            input_modalities=("scenario_record",),
            classes=("slot", "obstacle"),
        )
        self._slots = tuple(slots)
        self._obstacles = tuple(obstacles)

    def predict(self, frame: SensorFrame) -> RawPerceptionOutput:
        return RawPerceptionOutput(
            slots=self._slots,
            obstacles=self._obstacles,
            drivable_space_confidence=1.0,
            out_of_distribution_score=0.0,
            extra={"source": "ground_truth_replay", "not_for_vehicle_control": True},
        )
