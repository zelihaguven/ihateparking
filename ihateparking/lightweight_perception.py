"""Lightweight neural perception over the mock CARLA NumPy artifacts.

This module is intentionally dependency-light: a small NumPy MLP classifies
candidate regions proposed from the synthetic BEV raster. The same backend
contract can later be replaced by an ONNX/TensorRT model without changing the
planner-facing PerceptionFrame schema.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from .ai_perception import ModelMetadata, ModelBackend, RawObstacle, RawPerceptionOutput, RawSlot
from .geometry import Pose
from .stack_contracts import SensorFrame

try:
    import cv2
except ImportError:  # pragma: no cover - the fallback is useful for core installs
    cv2 = None


CLASS_NAMES = ("background", "slot", "static_obstacle", "dynamic_obstacle")
FEATURE_SIZE = 9


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 17
    samples_per_class: int = 160
    epochs: int = 90
    learning_rate: float = 0.08
    hidden_size: int = 24


@dataclass
class NumpyMLP:
    input_size: int = FEATURE_SIZE
    hidden_size: int = 24
    class_count: int = len(CLASS_NAMES)
    seed: int = 17

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / self.input_size), (self.input_size, self.hidden_size)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_size, dtype=np.float32)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden_size), (self.hidden_size, self.class_count)).astype(np.float32)
        self.b2 = np.zeros(self.class_count, dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        stable = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(stable)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0.0, features @ self.w1 + self.b1)
        return self._softmax(hidden @ self.w2 + self.b2)

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.predict_proba(features)
        labels = probabilities.argmax(axis=1)
        return labels, probabilities

    def fit(self, features: np.ndarray, labels: np.ndarray, epochs: int = 90, learning_rate: float = 0.08) -> list[float]:
        labels_one_hot = np.eye(self.class_count, dtype=np.float32)[labels]
        losses: list[float] = []
        for _ in range(epochs):
            hidden_pre = features @ self.w1 + self.b1
            hidden = np.maximum(0.0, hidden_pre)
            probabilities = self._softmax(hidden @ self.w2 + self.b2)
            loss = -np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-8)).mean()
            losses.append(float(loss))
            grad_logits = (probabilities - labels_one_hot) / len(labels)
            grad_w2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ self.w2.T
            grad_hidden[hidden_pre <= 0.0] = 0.0
            grad_w1 = features.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)
            self.w2 -= learning_rate * grad_w2
            self.b2 -= learning_rate * grad_b2
            self.w1 -= learning_rate * grad_w1
            self.b1 -= learning_rate * grad_b1
        return losses

    def save(self, path: str | Path, metadata: dict[str, str] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, metadata=np.array([metadata or {}], dtype=object))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NumpyMLP":
        data = np.load(path, allow_pickle=True)
        model = cls(input_size=data["w1"].shape[0], hidden_size=data["w1"].shape[1], class_count=data["w2"].shape[1])
        model.w1 = data["w1"].astype(np.float32)
        model.b1 = data["b1"].astype(np.float32)
        model.w2 = data["w2"].astype(np.float32)
        model.b2 = data["b2"].astype(np.float32)
        return model


def _prototypes() -> np.ndarray:
    # [mean_b, mean_g, mean_r, area, aspect, cx, cy, depth, lidar_density]
    return np.asarray([
        [0.82, 0.84, 0.83, 0.005, 1.0, 0.50, 0.50, 0.90, 0.00],  # background
        [0.28, 0.63, 0.43, 0.012, 1.0, 0.50, 0.35, 0.55, 0.15],  # slot
        [0.28, 0.34, 0.39, 0.035, 2.3, 0.50, 0.45, 0.40, 0.50],  # static obstacle
        [0.33, 0.41, 0.84, 0.022, 1.7, 0.50, 0.55, 0.65, 0.20],  # dynamic obstacle
    ], dtype=np.float32)


def make_training_data(config: TrainingConfig = TrainingConfig()) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    centers = _prototypes()
    features = []
    labels = []
    scales = np.asarray([.08, .08, .08, .012, .28, .20, .20, .20, .15], dtype=np.float32)
    for label, center in enumerate(centers):
        sample = center + rng.normal(0.0, scales, (config.samples_per_class, FEATURE_SIZE)).astype(np.float32)
        sample[:, 0:3] = np.clip(sample[:, 0:3], 0.0, 1.0)
        sample[:, 3] = np.clip(sample[:, 3], 0.0001, 0.25)
        sample[:, 4] = np.clip(sample[:, 4], 0.2, 5.0)
        sample[:, 5:8] = np.clip(sample[:, 5:8], 0.0, 1.0)
        sample[:, 8] = np.clip(sample[:, 8], 0.0, 2.0)
        features.append(sample)
        labels.extend([label] * len(sample))
    order = rng.permutation(len(labels))
    return np.vstack(features)[order], np.asarray(labels, dtype=np.int64)[order]


def train_model(config: TrainingConfig = TrainingConfig(), checkpoint: str | Path | None = None) -> tuple[NumpyMLP, dict[str, float]]:
    features, labels = make_training_data(config)
    model = NumpyMLP(hidden_size=config.hidden_size, seed=config.seed)
    losses = model.fit(features, labels, epochs=config.epochs, learning_rate=config.learning_rate)
    accuracy = float((model.predict(features)[0] == labels).mean())
    metrics = {"train_samples": float(len(labels)), "train_accuracy": accuracy, "final_loss": losses[-1]}
    if checkpoint is not None:
        model.save(checkpoint, {"backend": "numpy_mlp", "version": "0.1"})
    return model, metrics


def load_or_train(checkpoint: str | Path, config: TrainingConfig = TrainingConfig()) -> NumpyMLP:
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        return NumpyMLP.load(checkpoint)
    return train_model(config, checkpoint=checkpoint)[0]


def _connected_candidates(rgb: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Return x, y, w, h, area candidate boxes from the mock BEV raster."""
    if cv2 is None:
        return []
    b, g, r = [rgb[:, :, i] for i in range(3)]
    masks = [
        (g > 110) & (g > b * 1.22) & (g > r * 1.20),  # slot marker
        (b < 125) & (g < 135) & (r < 150),            # dark static obstacle
        (r > 150) & (r > g * 1.35) & (r > b * 1.35),  # red dynamic actor
    ]
    candidates: list[tuple[int, int, int, int, int]] = []
    for mask in masks:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for index in range(1, count):
            x, y, width, height, area = [int(value) for value in stats[index]]
            if area >= 12 and y + height < rgb.shape[0] - 20:
                candidates.append((x, y, width, height, area))
    return candidates


def _features_for_candidate(rgb: np.ndarray, depth: np.ndarray, lidar: np.ndarray, box: tuple[int, int, int, int, int]) -> np.ndarray:
    x, y, width, height, area = box
    crop = rgb[y:y + height, x:x + width]
    mean_color = crop.reshape(-1, 3).mean(axis=0) / 255.0 if crop.size else np.zeros(3)
    total_area = float(rgb.shape[0] * rgb.shape[1])
    center_x = (x + width / 2) / rgb.shape[1]
    center_y = (y + height / 2) / rgb.shape[0]
    aspect = min(5.0, width / max(height, 1))
    depth_y0 = int(y * depth.shape[0] / rgb.shape[0]); depth_y1 = max(depth_y0 + 1, int((y + height) * depth.shape[0] / rgb.shape[0]))
    depth_x0 = int(x * depth.shape[1] / rgb.shape[1]); depth_x1 = max(depth_x0 + 1, int((x + width) * depth.shape[1] / rgb.shape[1]))
    depth_mean = float(depth[depth_y0:depth_y1, depth_x0:depth_x1].mean()) / 50.0 if depth.size else 1.0
    lidar_density = min(2.0, len(lidar) / max(1.0, area) * 20.0)
    return np.asarray([mean_color[0], mean_color[1], mean_color[2], area / total_area, aspect, center_x, center_y, depth_mean, lidar_density], dtype=np.float32)


class NumpyMLPPerceptionBackend:
    """ModelBackend implementation for synthetic RGB/depth/LiDAR artifacts."""

    metadata = ModelMetadata(
        name="numpy_mlp_perception",
        version="0.1",
        backend="numpy",
        input_modalities=("rgb_front", "depth_front", "lidar"),
        classes=CLASS_NAMES,
    )

    def __init__(self, model: NumpyMLP, pixel_scale_m: float = 14.0, origin_px: tuple[float, float] = (110.0, 230.0)):
        self.model = model
        self.pixel_scale_m = pixel_scale_m
        self.origin_px = origin_px

    def predict(self, frame: SensorFrame) -> RawPerceptionOutput:
        if not frame.rgb_uri or not frame.depth_uri or not frame.lidar_uri:
            return RawPerceptionOutput(extra={"reason": "missing sensor modality"})
        rgb = np.load(frame.rgb_uri)
        depth = np.load(frame.depth_uri)
        lidar = np.load(frame.lidar_uri)
        boxes = _connected_candidates(rgb)
        if not boxes:
            return RawPerceptionOutput(extra={"candidate_count": 0})
        features = np.vstack([_features_for_candidate(rgb, depth, lidar, box) for box in boxes])
        labels, probabilities = self.model.predict(features)
        slots: list[RawSlot] = []
        obstacles: list[RawObstacle] = []
        for box, label, probability in zip(boxes, labels, probabilities):
            x, y, width, height, area = box
            confidence = float(probability[label])
            if confidence < 0.40:
                continue
            center_x_px = x + width / 2.0
            center_y_px = y + height / 2.0
            world_x = (center_x_px - self.origin_px[0]) / self.pixel_scale_m
            world_y = (self.origin_px[1] - center_y_px) / self.pixel_scale_m
            world_length = max(.35, width / self.pixel_scale_m)
            world_width = max(.35, height / self.pixel_scale_m)
            class_name = CLASS_NAMES[int(label)]
            if class_name == "slot":
                slots.append(RawSlot(f"slot_nn_{len(slots):02d}", Pose(world_x, world_y, 0.0), max(2.0, world_length), max(1.8, world_width), confidence, .25))
            elif class_name in ("static_obstacle", "dynamic_obstacle"):
                obstacles.append(RawObstacle(
                    track_id=f"{class_name}_nn_{len(obstacles):02d}",
                    kind=class_name,
                    pose=Pose(world_x, world_y, 0.0),
                    length_m=world_length,
                    width_m=world_width,
                    confidence=confidence,
                    uncertainty_m=.30 if class_name == "dynamic_obstacle" else .20,
                    dynamic=class_name == "dynamic_obstacle",
                ))
        return RawPerceptionOutput(
            slots=tuple(slots),
            obstacles=tuple(obstacles),
            drivable_space_confidence=.72 if slots or obstacles else .35,
            out_of_distribution_score=.18 if boxes else .75,
            extra={"candidate_count": len(boxes), "class_names": CLASS_NAMES},
        )


def evaluate_backend(backend: NumpyMLPPerceptionBackend, frames: Iterable[SensorFrame]) -> dict[str, float]:
    frame_count = 0
    frames_with_slot = 0
    frames_with_obstacle = 0
    for frame in frames:
        output = backend.predict(frame)
        frame_count += 1
        frames_with_slot += int(bool(output.slots))
        frames_with_obstacle += int(bool(output.obstacles))
    return {
        "frames": float(frame_count),
        "frames_with_slot": float(frames_with_slot),
        "frames_with_obstacle": float(frames_with_obstacle),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the lightweight NumPy perception model")
    parser.add_argument("--checkpoint", default="artifacts/lightweight_perception_model.npz")
    args = parser.parse_args()
    model, metrics = train_model(checkpoint=args.checkpoint)
    print({"checkpoint": args.checkpoint, **metrics})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
