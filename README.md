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

The suite covers perpendicular forward/reverse, parallel reverse and angled parking, with weather, sensor, calibration, occlusion and dynamic-actor metadata. `carla_adapter.py` is intentionally optional and raises a clear error instead of pretending that the CARLA runtime is available.

Phase 2 exposes `ai_perception.py`: a PyTorch/ONNX/TensorRT backend only needs to implement `predict(SensorFrame) -> RawPerceptionOutput`. `NeuralPerceptionAdapter` validates confidence and geometry, then returns the stable `PerceptionFrame` consumed by planning. `GroundTruthBackend` exists only for offline benchmark/replay and is explicitly marked as not for vehicle control.
```

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
