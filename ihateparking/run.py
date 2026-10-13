"""CLI entry point for ihateparking v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller import ControllerConfig, simulate_closed_loop
from .planner import HybridAStar, PlannerConfig
from .render import render_run
from .world import default_world


def main() -> int:
    parser = argparse.ArgumentParser(description="ihateparking: autonomous parking v1")
    parser.add_argument("--output-dir", default="artifacts", help="directory for plots and metrics")
    parser.add_argument("--max-expansions", type=int, default=100_000)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    world = default_world()
    planner = HybridAStar(world, PlannerConfig(max_expansions=args.max_expansions))
    plan = planner.plan()
    print(f"[planner] success={plan.success} expanded={plan.expanded} cost={plan.cost:.2f} path_length={plan.path_length:.2f}m")
    if not plan.success:
        print(f"[planner] {plan.message}")
        return 1

    simulation = simulate_closed_loop(world, plan, ControllerConfig())
    print(f"[controller] success={simulation.success} time={simulation.time:.2f}s distance={simulation.distance_travelled:.2f}m collisions={simulation.collisions}")
    print(f"[controller] {simulation.message}")

    metrics = {
        "project": "ihateparking",
        "version": "v1",
        "planner": {
            "algorithm": "Hybrid A*",
            "success": plan.success,
            "expanded": plan.expanded,
            "cost": plan.cost,
            "path_length_m": plan.path_length,
        },
        "controller": {
            "algorithm": "reverse-aware Pure Pursuit",
            "success": simulation.success,
            "time_s": simulation.time,
            "distance_m": simulation.distance_travelled,
            "collisions": simulation.collisions,
            "message": simulation.message,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not args.no_render:
        image = render_run(world, plan, simulation, output_dir / "parking_run.png")
        print(f"[render] {image}")
    return 0 if simulation.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
