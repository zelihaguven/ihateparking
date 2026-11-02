#!/usr/bin/env python3
"""Export reproducible model and planning artifacts for deployment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihateparking.dynamic_safety import SafetyConfig
from ihateparking.lightweight_perception import TrainingConfig, train_model
from ihateparking.planner import PlannerConfig
from ihateparking.scenario_sweep import DEFAULT_MAX_EXPANSIONS
from ihateparking.scenario_generator import ScenarioGenerator


# Mirrors ihateparking.dashboard without importing the optional FastAPI extra.
DASHBOARD_PLANNER_EXPANSIONS = 12_000


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
        "benchmark_profile": {
            "note": "Used by ihateparking.scenario_sweep for the 100-seed success-rate benchmark.",
            "config": asdict(PlannerConfig(max_expansions=DEFAULT_MAX_EXPANSIONS)),
        },
        "dashboard_profile": {
            "note": "Default search with Reeds-Shepp analytic expansion to the exact goal; the cap bounds reset latency.",
            "config": asdict(PlannerConfig(
                max_expansions=DASHBOARD_PLANNER_EXPANSIONS,
                goal_position_tolerance=.25,
                goal_yaw_tolerance=math.radians(7.0),
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
