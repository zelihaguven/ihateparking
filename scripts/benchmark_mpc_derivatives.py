"""Paired closed-loop benchmark for the finite-difference and CasADi-AD SLSQP modes.

This is a synthetic research benchmark, not a target-hardware real-time test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.demo_constrained_mpc import MPCConfig, _summary, run_comparison


def _aggregate(mode: str, config: MPCConfig, runs: int) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    call_times_ms: list[float] = []
    callback_totals: dict[str, int] = {}
    setup_times_ms: list[float] = []
    for index in range(runs):
        comparison = run_comparison(replace(config, derivative_mode=mode))
        rollout = comparison.constrained_mpc
        summary = _summary(rollout)
        call_times_ms.extend(value * 1000.0 for value in rollout.controller_call_times_s)
        setup_times_ms.append(rollout.solver_setup_time_s * 1000.0)
        callback_summary = summary.get("callback_totals", {})
        if isinstance(callback_summary, dict):
            for key, value in callback_summary.items():
                if isinstance(value, int):
                    callback_totals[key] = callback_totals.get(key, 0) + value
        per_run.append(
            {
                "run": index + 1,
                "success": rollout.success,
                "message": rollout.message,
                "collision_guard_stops": rollout.collisions,
                "minimum_raw_polygon_clearance_m": rollout.min_clearance_m,
                "steps": max(0, len(rollout.poses) - 1),
                "mean_full_controller_call_ms": summary["mean_controller_call_ms"],
                "p95_full_controller_call_ms": summary["p95_controller_call_ms"],
                "p99_full_controller_call_ms": summary["p99_controller_call_ms"],
                "max_full_controller_call_ms": summary["max_controller_call_ms"],
                "p95_optimizer_only_ms": summary["p95_mpc_solve_time_ms"],
                "solver_setup_time_ms": summary["solver_setup_time_ms"],
                "solver_iterations_total": summary["solver_iterations_total"],
                "optimizer_nfev_total": summary["optimizer_nfev_total"],
                "optimizer_njev_total": summary["optimizer_njev_total"],
                "callback_totals": summary["callback_totals"],
            }
        )

    values = np.asarray(call_times_ms, dtype=float)
    return {
        "mode": mode,
        "runs": per_run,
        "all_runs_reached_goal": all(item["success"] for item in per_run),
        "collision_guard_stops_total": sum(
            int(item["collision_guard_stops"]) for item in per_run
        ),
        "sample_count": int(values.size),
        "mean_full_controller_call_ms": float(values.mean()) if values.size else None,
        "p50_full_controller_call_ms": float(np.percentile(values, 50)) if values.size else None,
        "p95_full_controller_call_ms": float(np.percentile(values, 95)) if values.size else None,
        "p99_full_controller_call_ms": float(np.percentile(values, 99)) if values.size else None,
        "max_full_controller_call_ms": float(values.max()) if values.size else None,
        "mean_solver_setup_time_ms": statistics.mean(setup_times_ms) if setup_times_ms else None,
        "calls_over_180ms": int(np.count_nonzero(values > 180.0)),
        "calls_over_50ms": int(np.count_nonzero(values > 50.0)),
        "calls_over_20ms": int(np.count_nonzero(values > 20.0)),
        "callback_totals": callback_totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="paired repetitions per derivative mode (1–20)")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.18)
    parser.add_argument("--extra-clearance", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mpc_demo/derivative_benchmark.json"))
    args = parser.parse_args()
    if not 1 <= args.runs <= 20:
        parser.error("--runs must be between 1 and 20")
    if importlib.util.find_spec("casadi") is None:
        parser.error("CasADi is required for the paired benchmark; install with python -m pip install -e '.[mpc-ad]'")

    config = MPCConfig(
        horizon_steps=args.horizon,
        max_iterations=args.max_iterations,
        dt=args.dt,
        extra_clearance_m=args.extra_clearance,
        warm_start=True,
        derivative_mode="finite_difference",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": "synthetic static-obstacle reverse-in maneuver into a 2.5 m slot",
        "configuration": asdict(config),
        "measurement_scope": "Manus sandbox only; descriptive simulation measurements, not a target-hardware or real-time guarantee",
        "modes": {
            mode: _aggregate(mode, config, args.runs)
            for mode in ("finite_difference", "casadi_ad")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote benchmark: {args.output}")


if __name__ == "__main__":
    main()
