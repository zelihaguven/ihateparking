#!/usr/bin/env python3
"""Export reproducible model and planning artifacts for deployment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihateparking.dynamic_safety import SafetyConfig
from ihateparking.lightweight_perception import TrainingConfig, train_model
from ihateparking.planner import PlannerConfig
from ihateparking.scenario_generator import ScenarioGenerator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_pipeline(output_dir: str | Path, scenario_count: int = 24, seed: int = 100) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    training = TrainingConfig()
    checkpoint = root / "perception_model_numpy_mlp_v2.npz"
    _, metrics = train_model(training, checkpoint=checkpoint)
    (root / "perception_metadata.json").write_text(json.dumps({
        "model_name": "numpy_mlp_perception",
        "version": "0.2",
        "backend": "numpy",
        "classes": ["background", "slot", "static_obstacle", "dynamic_obstacle"],
        "feature_size": 9,
        "training_config": asdict(training),
        "metrics": metrics,
        "runtime_note": "Replace this artifact with ONNX/TensorRT weights for production camera inference.",
    }, indent=2), encoding="utf-8")

    planning = {
        "planner": {
            "name": "HybridAStar",
            "algorithmic": True,
            "weights": None,
            "note": "The planner has no learned weights; these are reproducible search parameters.",
            "config": asdict(PlannerConfig()),
        },
        "dashboard_robust_profile": {
            "config": asdict(PlannerConfig(
                xy_resolution=.8, yaw_bins=16, primitive_length=1.2,
                primitive_sample=.4, steering_samples=5,
                goal_position_tolerance=.7, goal_yaw_tolerance=.48869,
                max_expansions=30_000,
            )),
        },
        "dashboard_capture_profile": {
            "config": asdict(PlannerConfig(
                xy_resolution=.65, yaw_bins=16, primitive_length=.75,
                primitive_sample=.25, steering_samples=5,
                goal_position_tolerance=.35, goal_yaw_tolerance=.20944,
                max_expansions=180,
            )),
        },
        "dynamic_safety": asdict(SafetyConfig()),
    }
    (root / "planning_profiles.json").write_text(json.dumps(planning, indent=2), encoding="utf-8")

    suite_path = root / "scenario_suite.json"
    ScenarioGenerator().write_suite(suite_path, count=scenario_count, seed=seed)

    files = [checkpoint, root / "perception_metadata.json", root / "planning_profiles.json", suite_path]
    manifest = {
        "format": "ihateparking-deployment-export-v1",
        "generated_from": "ihateparking",
        "scenario_count": scenario_count,
        "seed": seed,
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output_dir": str(root), "checkpoint": str(checkpoint), "metrics": metrics, "manifest": str(root / "MANIFEST.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export reproducible perception, planning and scenario artifacts")
    parser.add_argument("--output-dir", default="deployment/export")
    parser.add_argument("--scenario-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(export_pipeline(args.output_dir, args.scenario_count, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
