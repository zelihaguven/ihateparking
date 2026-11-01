"""Check that MetaDrive runs on this machine: headless env, a few steps, one LiDAR scan.

MetaDrive 0.4.3 needs Python < 3.12, so it lives in its own environment:

    uv venv --python 3.11 .venv-metadrive
    uv pip install --python .venv-metadrive/bin/python metadrive-simulator==0.4.3
    .venv-metadrive/bin/python scripts/metadrive_smoke.py
"""
from __future__ import annotations

import time

import numpy as np


def main() -> int:
    started = time.perf_counter()
    from metadrive.envs.metadrive_env import MetaDriveEnv

    print(f"import {time.perf_counter() - started:.1f} s")
    env = MetaDriveEnv(dict(use_render=False, num_scenarios=1, start_seed=0, log_level=50))
    try:
        started = time.perf_counter()
        env.reset()
        print(f"reset {time.perf_counter() - started:.1f} s")
        for _ in range(50):
            env.step([0.0, 0.5])
        agent = env.agent
        print(f"after 50 steps: position {np.round(agent.position, 2)}, speed {agent.speed:.1f}")
        lidar = env.engine.get_sensor("lidar")
        cloud, objects = lidar.perceive(agent, physics_world=env.engine.physics_world.dynamic_world, num_lasers=240, distance=50)
        print(f"lidar: {len(cloud)} beams, {len(objects)} objects detected")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
