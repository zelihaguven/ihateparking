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

## Step 3.2: the full mission in MetaDrive

- `MetaDrivePlant` puts MetaDrive behind the mission's `Plant` interface (`ihateparking/plant.py`);
  the stack's mission loop runs unchanged (`scripts/metadrive_mission.py`). Any contact counts as
  a collision (MetaDrive reports contact after it happens).
- Low-level driving: steering angle -> MetaDrive's normalised steering; a PI loop on the signed
  speed -> MetaDrive's throttle/brake (tracks 1.5 m/s forward and 1 m/s reverse within 0.01 m/s).
- Found by MetaDrive, invisible in the built-in simulator:
  1. Heading from steering drifts on a real chassis (MetaDrive turns up to 7% less at large angles;
     -11 deg in 60 s). Odometry now uses a yaw-rate gyro when the plant has one.
  2. The car follows plans within 5 / 16 / 27 cm (median / p95 / max) in MetaDrive against
     1 / 3 / 4 cm in the built-in simulator. Exploring and approaching legs now keep 0.30 m from
     mapped structure; the park manoeuvre keeps 0.15 m.

12 MetaDrive lots (scan matching, 6 spaces a side):

| Clearance explore / park | Correct park | Contact | Wrong park |
|---|---|---|---|
| 0.15 / 0.15 m, steering odometry | 7 | 2 | 0 |
| 0.15 / 0.15 m, gyro | 8 | 1 | 0 |
| 0.30 / 0.30 m, gyro | 7 | 0 | 0 |
| **0.30 / 0.15 m, gyro** | **9** | **0** | **0** |

The three failures are timeouts or no space found while exploring MetaDrive's open, wall-less
surroundings.

## Next

An adapter that builds a parking lot from the `ParkingLot` block, parks static vehicles in some
spaces, feeds MetaDrive's LiDAR into `OccupancyGrid`, and drives the ego vehicle from
`PlanTracker` commands (steering, target speed -> MetaDrive's normalised steer/throttle).
Ground truth (free spaces) comes from the block's lanes, used for scoring only.
