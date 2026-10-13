# ihateparking

> A closed-loop autonomous parking research prototype.

`ihateparking` is an independent project from RoadLens. RoadLens focuses on perception reliability; this repository is becoming an **AI-assisted, safety-supervised autonomous parking stack** for simulation first and vehicle integration later.

The generated tutorial video is an explanatory artifact only. It is not the autonomy core and must not be treated as a realistic sensor or actuator demo.

## v1 goal

Put a nonholonomic vehicle into a perpendicular parking slot in simulation without collision:

```text
parking world -> collision model -> Hybrid A* -> reverse-aware Pure Pursuit -> closed-loop trace
```

### v1 scope

- 2-D parking world with two parked vehicles, a curb, and a target slot
- Rear-axle bicycle model
- Slot goals converted from footprint centers to rear-axle poses
- Oriented-rectangle collision checking using the separating-axis theorem
- Hybrid A* with forward/reverse motion primitives and steering discretisation
- Reverse-aware Pure Pursuit controller
- Headless matplotlib artifact and JSON metrics
- No dependency on YOLO or RoadLens

## Run

```bash
cd /home/ubuntu/ihateparking
python3 -m ihateparking.run
```

Artifacts are written to `artifacts/`:

- `parking_run.png` — planned path, actual closed-loop trace, start and final poses
- `metrics.json` — planner/controller metrics
- `trace.json` — world, plan and per-frame trace consumed by the browser replay

## Browser demo

After generating the artifacts, serve the project root:

```bash
cd /home/ubuntu/ihateparking
python3 -m http.server 8000 --bind 0.0.0.0
```

Open `/web/` to replay the real v1 trace. The dashboard has Play/Pause, step, reset and replay-speed controls, plus live position, heading, steering, gear and mission metrics.

## English Video Assist demo

The `/web/assist.html` page replays a real in-cabin Pexels clip and overlays the output of `ihateparking.video_assist`. The current v1 perception signal is a deterministic orange-boundary/cone detector implemented with OpenCV. It produces English human-in-the-loop suggestions such as `KEEP CENTERED`, `STEER RIGHT GENTLY`, `STEER LEFT GENTLY`, and `STOP — boundary is too close`.

```bash
python3 -m ihateparking.video_assist assets/pexels_parking_pov.mp4
```

The current clip contains a cone driving course, a turn, and a stop, but not a complete reverse parking manoeuvre. This limitation is recorded in `docs/footage-research.md`; the next dataset should add a fixed rear/360 camera view and a continuous parking-slot sequence.

## Self-generated virtual lesson

The `/web/tutorial.html` page plays a locally rendered English parking lesson. It is not stock footage: every frame is generated from the v1 world, Hybrid A* path and closed-loop trace. The lesson explains approach, turning, arc following, wheel straightening and the final stop while showing the steering gauge, path and zero-collision result.

```bash
python3 -m ihateparking.tutorial_video --output assets/ihateparking_virtual_parking_tutorial.mp4
```

## Architecture

```text
world.py       Scenario, vehicle dimensions, slot and collision queries
scenario_generator.py Seeded multi-scenario contract for CARLA/replay
carla_adapter.py Optional CARLA runtime boundary
ai_perception.py Model-independent AI perception interface and validation
lightweight_perception.py NumPy MLP detector for slots and dynamic obstacles
dynamic_safety.py Time-indexed prediction, replanning and emergency braking
ground_truth_benchmark.py Replayable sensor stream and benchmark runner
pedestrian_stress_test.py Deterministic pedestrian crossing safety benchmark
mock_carla_server.py GPU-free JSON-lines server and pipeline smoke test
dashboard.py FastAPI real-time dynamic safety visualization API
stack_contracts.py Sensor, perception, trajectory, control and safety messages
geometry.py    Pose, oriented rectangles, polygon intersection
planner.py     Hybrid A* search with kinematic motion primitives
controller.py  Reverse-aware Pure Pursuit + simulator
render.py      Static visualization for experiments
run.py         Reproducible CLI entry point + browser trace export
video_assist.py OpenCV-based footage perception and English guidance signals
 tutorial_video.py Self-generated instructional parking animation
web/           Static browser replay dashboard
assets/        Research footage used by the Video Assist demo
```

The target real-system architecture is documented in [`docs/real-system-architecture.md`](docs/real-system-architecture.md). The first transport-independent message contracts live in `stack_contracts.py`; they are designed to map to ROS2 messages later without importing ROS2 into the core planner.

## Phase 1–2: scenarios and AI perception boundary

Generate a deterministic scenario suite without requiring CARLA to be installed:

```bash
python3 - <<'PY'
from ihateparking.scenario_generator import ScenarioGenerator
path = ScenarioGenerator().write_suite("artifacts/scenario_suite.json", count=20, seed=100)
print(path)
PY
```

The suite covers perpendicular forward/reverse, parallel forward/reverse and angled forward/reverse parking, with weather, sensor, calibration, occlusion and dynamic-actor metadata. `carla_adapter.py` is intentionally optional and raises a clear error instead of pretending that the CARLA runtime is available.

Phase 2 exposes `ai_perception.py`: a PyTorch/ONNX/TensorRT backend only needs to implement `predict(SensorFrame) -> RawPerceptionOutput`. `NeuralPerceptionAdapter` validates confidence and geometry, then returns the stable `PerceptionFrame` consumed by planning. `GroundTruthBackend` exists only for offline benchmark/replay and is explicitly marked as not for vehicle control.

### First sensor-flow benchmark

Run the CARLA-independent GroundTruthBackend flow before installing the simulator:

```bash
python3 -m ihateparking.ground_truth_benchmark --count 4 --frames 5
```

This exercises `ScenarioSpec -> SensorFrame stream -> GroundTruthBackend -> NeuralPerceptionAdapter -> PerceptionFrame` and writes `artifacts/ground_truth_benchmark.json`.

### Lightweight neural perception

`lightweight_perception.py` contains a small fully connected neural network implemented with NumPy. Color/geometry candidate proposals are extracted from the synthetic BEV RGB artifact, fused with depth and LiDAR features, and classified as `slot`, `static_obstacle` or `dynamic_obstacle`. The result is converted to the same `RawPerceptionOutput` contract used by future ONNX/TensorRT backends.

Train the model without PyTorch or a GPU:

```bash
python3 -m ihateparking.lightweight_perception \
  --checkpoint artifacts/lightweight_perception_model.npz
```

The mock server trains or loads its own versioned checkpoint automatically. Its `reset` request accepts `dynamic_actor_count`; dynamic vehicles, pedestrians and shopping carts move along deterministic paths, appear in the GroundTruthBackend stream, and are rendered in the synthetic artifacts for neural detection.

### CARLA connection and spawning

When the CARLA Python API and server are available, the adapter can load the requested map, enable synchronous mode, set weather, spawn the ego vehicle and attach front/rear RGB, front depth, LiDAR and IMU sensors:

```python
from ihateparking.carla_adapter import CarlaScenarioAdapter
from ihateparking.scenario_generator import ScenarioConfig, ScenarioGenerator

scenario = ScenarioGenerator().generate(ScenarioConfig(seed=7))
with CarlaScenarioAdapter() as adapter:
    plan = adapter.spawn_vehicle_and_sensors(scenario)
    adapter.tick()
```

The current sandbox does not contain CARLA, so the spawn path is validated through a serializable `CarlaSpawnPlan` and an explicit missing-runtime error rather than a fake live run.

### Local mock CARLA server

For a GPU-free end-to-end contract test, start the local JSON-lines mock server:

```bash
# One-command smoke test: starts an ephemeral TCP server internally, then
# resets, streams synthetic sensors, runs Hybrid A*, and steps controls.
python3 -m ihateparking.mock_carla_server --mode smoke-test --frames 8

# Optional long-running endpoint for a separate client
python3 -m ihateparking.mock_carla_server --mode server --port 2100
```

The smoke test uses the same request boundary as a remote server and writes synthetic `.npy` RGB/depth/LiDAR artifacts. The mock image is explicitly a top-down debug representation, not a photorealistic camera. Supported operations are `health`, `reset`, `sensor_frame`, `plan`, `step`, `spawn_plan` and `shutdown`.

### Dynamic replanning and safety braking

`dynamic_safety.py` predicts each confident dynamic track over a two-second horizon. It compares predicted overlap and time-to-collision against reaction time, braking distance and a safety margin:

```text
clear envelope       → PROCEED
approaching envelope → REPLAN with dynamic obstacles inserted
emergency TTC        → BRAKE / SAFE_STOP with zero speed
```

The safety layer is conservative by design. A failed replan is converted into an emergency stop rather than an unsafe fallback.

### FastAPI real-time dashboard

Install the optional dashboard dependencies and start the local visualization:

```bash
python3 -m pip install -r requirements-dashboard.txt
python3 -m ihateparking.dashboard --host 0.0.0.0 --port 8010
```

Open `http://localhost:8010`. The default dashboard starts with **zero dynamic actors** so the basic parking demo is clean and does not manufacture a collision. Set `Dynamic actors` above zero when you explicitly want the moving-actor stress test. The dashboard shows the mock world, planned path, neural slot/obstacle detections, dynamic actor tracks, clearance, braking distance, time-to-collision, replanning events and safety-stop events. The color code is **cyan = ego vehicle**, **red = moving actor/hazard**, **orange = target slot**, and **gray = parked/static obstacle**. A `BRAKE` state holds the vehicle at zero speed but does not stop the telemetry loop; frames continue advancing so the supervisor can observe the hazard clear. `POST /api/tick` advances one perception/control cycle; `POST /api/reset` changes the seed, scenario kind and dynamic actor count.

## Expanded parking scenario coverage

`ScenarioKind` now exposes six deterministic manoeuvre variants:

- `perpendicular_forward`
- `perpendicular_reverse`
- `parallel_forward`
- `parallel_reverse`
- `angled_forward`
- `angled_reverse`

The parallel baseline uses a 6 m manoeuvring slot rather than a barely
vehicle-length gap. This keeps the scenario physically plausible while still
testing tight clearance around parked vehicles.

## Deployment export and Docker

Export the NumPy perception checkpoint, algorithmic planner profiles, all six
scenario kinds and SHA-256 manifest:

```bash
python3 scripts/export_pipeline.py --output-dir deployment/export --scenario-count 24 --seed 100
```

The Hybrid A* planner has **no learned weights**; `planning_profiles.json`
exports its search/safety parameters. The exported NumPy checkpoint is a
baseline artifact for the mock pipeline, not a production-certified vehicle
perception model.

Build and run the clean deployment image on a host with Docker:

```bash
docker build -t ihateparking:deployment .
docker run --rm -p 8010:8010 ihateparking:deployment
```

The image is a transport-independent runtime skeleton. Real hardware still
needs validated camera/LiDAR/CAN or ROS2 adapters and a production perception
backend.

## Pedestrian safety stress test

Run a deterministic four-pedestrian crossing scenario. The pedestrians are
placed just outside the ego footprint so a safety stop can be evaluated without
manufacturing a collision:

```bash
python3 -m ihateparking.pedestrian_stress_test \
  --output artifacts/pedestrian_stress.json \
  --frames 90 --pedestrians 4
```

The report includes ground-truth pedestrian visibility, neural dynamic-track
visibility, PROCEED/REPLAN/BRAKE counts, first brake frame, minimum predicted
clearance and contact events.

## Initial experiment protocol

The first benchmark will keep the vehicle and slot fixed while varying:

1. start pose error (position and heading),
2. parking-slot width,
3. obstacle clearance,
4. controller choice (Pure Pursuit vs MPC in v2), and
5. planner expansion budget.

Primary metrics:

- parking success rate,
- collision count,
- final position/yaw error,
- minimum obstacle distance,
- manoeuvre time,
- path length,
- number of gear changes, and
- planning expansions/time.

## Roadmap

### v1 — current

- [x] scenario and collision model
- [x] Hybrid A* planner
- [x] closed-loop controller
- [x] reproducible render + metrics
- [x] browser replay dashboard
- [ ] automated scenario sweep

### v2 — research layer

- [ ] compare Pure Pursuit, Stanley and MPC
- [ ] add noisy parking-slot/obstacle estimates
- [ ] model uncertainty-aware safety margins
- [ ] dynamic pedestrian/vehicle obstacle
- [ ] success-rate benchmark across 100 randomized seeds

### v3 — perception and demo layer

- [ ] camera-based parking-slot segmentation
- [ ] top-down local occupancy map
- [ ] browser-based scenario editor
- [ ] optional CARLA bridge

## Portfolio positioning

> **Built an independent closed-loop autonomous parking system using Hybrid A* motion planning, a rear-axle bicycle model, polygon-based collision checking, and reverse-aware Pure Pursuit control. Evaluated manoeuvre success, collision safety, and planning efficiency in constrained parking scenarios.**
