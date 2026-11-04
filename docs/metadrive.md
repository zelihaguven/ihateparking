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

## Step 3.3: walled lot, 40 lots (e38607e, scan matching, seeds 101–140)

The lot is walled in (perimeter behind the row kerbs, 6 m beyond the outer spaces) and the ego
starts at its entrance. A lot "has a gap" when a free space has parked cars on both sides; without
one, stopping is the designed outcome (no bay lines to place a car in a run).

| Lots | Count | Result |
|---|---|---|
| with a gap between cars | 29 | **27 parked correctly (93%)**, 2 no space found |
| without one | 11 | 8 stopped (correct), 1 parked correctly anyway, 2 timed out |
| all | 40 | **0 contacts, 0 wrong parks** |

Parks: overlap 100% in every one; lateral error median 7 cm, max 18 cm; axis error median 2.1 deg,
max 4.9 deg. Raw outputs: docs/benchmarks/metadrive_e38607e/.

## Step 3.4: the four remaining MetaDrive failures (0c353d2)

- 104: a neighbour parked deep in an 8 m space showed only 1.9 m of its side from the aisle; the
  tentative detector (where to drive up and look) now accepts 1.5 m, parking still needs 3 m.
- 138: gaps refused as "no route" while the map was incomplete were never retried; refusals now
  expire after 20 s.
- 103, 112 (no gap between cars): recovery after recovery until the timeout; three recoveries in a
  row with no frontier or bay now end the mission.

Same 40 lots (seeds 101–140): **29 of 29 lots with a gap between cars parked correctly; the 11
without one stopped (one parked correctly anyway); 0 contacts, 0 wrong parks, 0 timeouts.** Every
park fully inside its space; lateral error median 7 cm, max 33 cm; axis error max 4.9 deg; longest
mission 175 s. Built-in dev 6000–6119 with the same code: 116/120 (was 115), 0 wrong parks,
position error 16 / 35 cm. Raw: docs/benchmarks/metadrive_0c353d2/.

## Next

An adapter that builds a parking lot from the `ParkingLot` block, parks static vehicles in some
spaces, feeds MetaDrive's LiDAR into `OccupancyGrid`, and drives the ego vehicle from
`PlanTracker` commands (steering, target speed -> MetaDrive's normalised steer/throttle).
Ground truth (free spaces) comes from the block's lanes, used for scoring only.

## Watching a mission

- `--video` records MetaDrive's own top-down view: walls, kerbs, parked cars and the ego car, seen by a still camera over the lot. It writes `artifacts/metadrive/mission_<seed>.mp4` at 4× speed, one frame per 0.2 s of driving:

  ```bash
  .venv-metadrive/bin/python scripts/metadrive_mission.py --localization scan_matching --video 101
  ```

- `--window` opens MetaDrive's 3-D window and follows the car live. Run it from your own Terminal, because a window needs a desktop session; from an agent's background shell Panda3D cannot open one. MetaDrive asks for 8x multisampling, and this Mac's OpenGL refuses a window with that. The bridge asks for 4x instead (`WINDOW_MULTISAMPLES`). `scripts/metadrive_window_check.py` shows which settings open a window on a machine.