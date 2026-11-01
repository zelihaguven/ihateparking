# MetaDrive on this Mac (PRD step 3)

CARLA is not used (Apple Silicon, no NVIDIA GPU, no dependable free GPU rental; decided 2026-10-26).
MetaDrive is the free simulator candidate.

## Status (2026-11-01)

- MetaDrive 0.4.3 installs and runs headless on an Apple M4 Pro (macOS, arm64).
- It requires Python < 3.12, so it lives in `.venv-metadrive` (Python 3.11 via `uv`), separate from
  the project's `.venv` (Python 3.12). The panda3d 1.10.13 wheel it pins ships as macOS universal2.
- Smoke test (`scripts/metadrive_smoke.py`): import 68 s the first time, environment reset 14 s,
  50 driving steps, a 240-beam LiDAR scan.
- pygame and opencv both bundle SDL2; macOS warns about duplicate SDL classes. Harmless so far.
- MetaDrive ships a `ParkingLot` road block with perpendicular parking spaces (used by its
  multi-agent parking-lot environment), so a parking scenario can be built from it.

## Setup

```bash
uv venv --python 3.11 .venv-metadrive
uv pip install --python .venv-metadrive/bin/python metadrive-simulator==0.4.3
.venv-metadrive/bin/python scripts/metadrive_smoke.py
```

## Next

An adapter that builds a parking lot from the `ParkingLot` block, parks static vehicles in some
spaces, feeds MetaDrive's LiDAR into `OccupancyGrid`, and drives the ego vehicle from
`PlanTracker` commands (steering, target speed -> MetaDrive's normalised steer/throttle).
Ground truth (free spaces) comes from the block's lanes, used for scoring only.
