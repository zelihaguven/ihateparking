"""Render a self-contained English instructional parking animation."""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon, Rectangle

from .controller import simulate_closed_loop
from .planner import HybridAStar
from .world import default_world


def _phase(t: float, duration: float, gear: int, steering: float, max_steering: float) -> tuple[str, str, str]:
    progress = t / max(duration, 1e-6)
    if gear < 0:
        return "STEP 03", "REVERSE SLOWLY", "Use the planned curve. Keep the vehicle inside the highlighted corridor."
    if progress < 0.22:
        return "STEP 01", "APPROACH THE SLOT", "Reduce speed and line up with the open parking slot."
    if progress < 0.52:
        return "STEP 02", "TURN RIGHT GENTLY", "Follow the curved path into the slot. Do not rush the steering input."
    if progress < 0.84:
        return "STEP 03", "KEEP THE ARC", "Hold the steering direction while the vehicle clears the parked cars."
    if progress < 0.96:
        return "STEP 04", "STRAIGHTEN THE WHEELS", "Center the vehicle and reduce the steering angle."
    return "STEP 05", "STOP — PARKING COMPLETE", "The vehicle is inside the target slot with zero collision events."


def _draw_box(ax, world, center, length, width, yaw, face, edge, alpha=1.0):
    from .geometry import oriented_box
    ax.add_patch(Polygon(oriented_box(center, length, width, yaw), closed=True, facecolor=face, edgecolor=edge, linewidth=1.5, alpha=alpha))


def _draw_vehicle(ax, world, pose, face, edge="#d9f3e8"):
    poly = world.vehicle_polygon(pose)
    ax.add_patch(Polygon(poly, closed=True, facecolor=face, edgecolor=edge, linewidth=1.7, zorder=8))
    ax.arrow(pose.x, pose.y, 0.85 * math.cos(pose.yaw), 0.85 * math.sin(pose.yaw), head_width=.16, head_length=.24, color=edge, linewidth=1.5, length_includes_head=True, zorder=9)


def _draw_steering_gauge(ax, steering, maximum):
    gauge = ax.inset_axes([0.13, 0.09, 0.74, 0.14])
    gauge.set_xlim(-1, 1); gauge.set_ylim(0, 1); gauge.axis("off")
    gauge.plot([-1, 1], [.32, .32], color="#35536a", linewidth=5, solid_capstyle="round")
    gauge.plot([0, 0], [.19, .47], color="#89a9b9", linewidth=1)
    # In this world the car approaches the slot facing left, so a negative
    # bicycle-model steering input produces a visual right turn. Invert the
    # gauge to match the learner-facing instruction card.
    normalized = max(-1.0, min(1.0, -steering / maximum if maximum else 0.0))
    gauge.scatter([normalized], [.32], s=160, color="#edaa4e", edgecolor="#fff3dc", linewidth=1.5, zorder=3)
    gauge.text(-1, .62, "LEFT", color="#9bc3cd", fontsize=8, ha="left", weight="bold")
    gauge.text(1, .62, "RIGHT", color="#9bc3cd", fontsize=8, ha="right", weight="bold")
    gauge.text(0, -.05, "STEERING INPUT", color="#668492", fontsize=7, ha="center")


def render_tutorial(output: str | Path, fps: int = 20) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    world = default_world()
    plan = HybridAStar(world).plan()
    if not plan.success:
        raise RuntimeError(plan.message)
    simulation = simulate_closed_loop(world, plan)
    if not simulation.success:
        raise RuntimeError(simulation.message)

    total_frames = len(simulation.poses)
    duration = simulation.time
    temp_dir = Path(tempfile.mkdtemp(prefix="ihateparking_tutorial_"))
    try:
        for i, pose in enumerate(simulation.poses):
            t = min(duration, i * 0.05)
            signal = simulation.gears[i - 1] if i > 0 and i - 1 < len(simulation.gears) else 1
            steering = simulation.steerings[i - 1] if i > 0 and i - 1 < len(simulation.steerings) else 0.0
            step, command, detail = _phase(t, duration, signal, steering, world.vehicle.max_steering)
            fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#0e1b27")
            scene = fig.add_axes([0.055, 0.14, 0.60, 0.76], facecolor="#edf4f1")
            info = fig.add_axes([0.70, 0.12, 0.25, 0.78], facecolor="#173347")
            info.set_xlim(0, 1); info.set_ylim(0, 1); info.axis("off")
            scene.set_xlim(*world.bounds[:2]); scene.set_ylim(*world.bounds[2:]); scene.set_aspect("equal"); scene.set_xticks([]); scene.set_yticks([])
            for spine in scene.spines.values(): spine.set_visible(False)
            for obstacle in world.obstacles:
                _draw_box(scene, world, obstacle.center, obstacle.length, obstacle.width, obstacle.yaw, "#455562", "#20313d")
                scene.text(obstacle.center[0], obstacle.center[1], obstacle.name.replace("_", " "), color="white", ha="center", va="center", fontsize=7, weight="bold", rotation=90 if abs(obstacle.yaw) > .5 else 0)
            _draw_box(scene, world, world.slot.center, world.slot.length, world.slot.width, world.slot.yaw, "#cbeedd", "#319266", alpha=.92)
            scene.text(world.slot.center[0], world.slot.center[1], "TARGET\nSLOT", color="#257050", ha="center", va="center", fontsize=9, weight="bold", rotation=90)
            scene.plot([p.x for p in plan.poses], [p.y for p in plan.poses], linestyle=(0, (6, 5)), color="#e5a347", linewidth=1.8, alpha=.95)
            visible = simulation.poses[: i + 1]
            scene.plot([p.x for p in visible], [p.y for p in visible], color="#246b86", linewidth=2.8, alpha=.95, zorder=4)
            _draw_vehicle(scene, world, world.start_pose, "#edb452", edge="#49351c")
            _draw_vehicle(scene, world, pose, "#55bd90", edge="#0f5540")
            scene.text(0.02, 1.02, "VIRTUAL PARKING LESSON", transform=scene.transAxes, fontsize=10, color="#246b86", weight="bold", va="bottom")
            scene.text(.98, 1.02, f"{t:04.1f} / {duration:04.1f} s", transform=scene.transAxes, fontsize=9, color="#6e7f87", weight="bold", ha="right", va="bottom")
            progress = i / max(total_frames - 1, 1)
            scene.plot([world.bounds[0] + .5, world.bounds[0] + .5 + (world.bounds[1] - world.bounds[0] - 1) * progress], [world.bounds[2] + .35] * 2, color="#246b86", linewidth=4, solid_capstyle="round", zorder=10)
            scene.plot([world.bounds[0] + .5 + (world.bounds[1] - world.bounds[0] - 1) * progress] * 2, [world.bounds[2] + .22, world.bounds[2] + .48], color="#edaa4e", linewidth=2, zorder=11)

            info.text(.08, .91, step, color="#edaa4e", fontsize=10, weight="bold", transform=info.transAxes)
            info.text(.08, .82, command, color="white", fontsize=19, weight="bold", transform=info.transAxes, wrap=True)
            info.text(.08, .61, detail, color="#c4d7dc", fontsize=10.5, linespacing=1.45, transform=info.transAxes, wrap=True)
            info.text(.08, .54, "MISSION TELEMETRY", color="#85b5bf", fontsize=8, weight="bold", transform=info.transAxes)
            telemetry = [("Planner", "HYBRID A*"), ("Controller", "PURE PURSUIT"), ("Collisions", "0"), ("Progress", f"{round(progress * 100):02d}%")]
            for row, (label, value) in enumerate(telemetry):
                y = .45 - row * .065
                info.text(.08, y, label, color="white", fontsize=8.5, transform=info.transAxes, family="DejaVu Sans Mono")
                info.text(.52, y, value, color="white", fontsize=8.5, transform=info.transAxes, family="DejaVu Sans Mono")
            _draw_steering_gauge(info, steering, world.vehicle.max_steering)
            info.text(.08, .05, "HUMAN-IN-THE-LOOP DEMO", color="#56bd91", fontsize=8, weight="bold", transform=info.transAxes)
            fig.text(.055, .055, "ihateparking", color="white", fontsize=19, weight="bold")
            fig.text(.255, .059, "explain the next move", color="#9bc3cd", fontsize=10)
            fig.savefig(temp_dir / f"frame_{i:05d}.png", facecolor=fig.get_facecolor())
            plt.close(fig)

        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(fps), "-i", str(temp_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
        subprocess.run(command, check=True)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the ihateparking English virtual parking lesson")
    parser.add_argument("--output", default="assets/ihateparking_virtual_parking_tutorial.mp4")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    path = render_tutorial(args.output, args.fps)
    print(f"[tutorial] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
