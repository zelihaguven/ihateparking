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

## Step 3.1: perception bridge (`ihateparking/metadrive_bridge.py`)

- `MetaDriveLot(seed)` builds MetaDrive's parking-lot block (space width set to 2.7 m through the
  lane width; MetaDrive's default 3.5 m is wider than real bays), parks static MetaDrive cars in a
  random half of the spaces (a few decimetres and degrees off centre) and lines the back of each
  row with static barriers as kerbs (MetaDrive's LiDAR does not see sidewalks).
- `scan()` converts MetaDrive's LiDAR into the stack's `LidarScan`; `pose()` converts the body
  centre to the stack's rear-axle pose; `spaces` is the ground truth, for scoring only.
- `scripts/metadrive_perception_check.py` teleports the ego along the aisle, builds the stack's map
  from MetaDrive LiDAR and runs the stack's detector. Seeds 0–8: 5 detections, all 5 in a free
  space (precision 100%). Recall is low because MetaDrive's block has 4 spaces per side: most free
  spaces sit at a row end, where the stack does not park by design.

## Next

An adapter that builds a parking lot from the `ParkingLot` block, parks static vehicles in some
spaces, feeds MetaDrive's LiDAR into `OccupancyGrid`, and drives the ego vehicle from
`PlanTracker` commands (steering, target speed -> MetaDrive's normalised steer/throttle).
Ground truth (free spaces) comes from the block's lanes, used for scoring only.
