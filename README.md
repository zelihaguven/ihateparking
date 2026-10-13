# ihateparking

> A closed-loop autonomous parking research prototype.

`ihateparking` is an independent project from RoadLens. RoadLens focuses on perception reliability; this repository focuses on **motion planning, vehicle kinematics, collision-aware manoeuvres, and closed-loop control**.

## v1 goal

Put a nonholonomic vehicle into a perpendicular parking slot in simulation without collision:

```text
parking world -> collision model -> Hybrid A* -> reverse-aware Pure Pursuit -> closed-loop trace
```

### v1 scope

- 2-D parking world with two parked vehicles, a curb, and a target slot
- Rear-axle bicycle model
- Slot goals are converted from footprint centers to rear-axle poses
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

## Architecture

```text
world.py       Scenario, vehicle dimensions, slot and collision queries
geometry.py    Pose, oriented rectangles, polygon intersection
planner.py     Hybrid A* search with kinematic motion primitives
controller.py  Reverse-aware Pure Pursuit + simulator
render.py      Static visualization for experiments
run.py         Reproducible CLI entry point
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
- [ ] browser-based replay dashboard
- [ ] optional CARLA bridge

## Portfolio positioning

> **Built an independent closed-loop autonomous parking system using Hybrid A* motion planning, a rear-axle bicycle model, polygon-based collision checking, and reverse-aware Pure Pursuit control. Evaluated manoeuvre success, collision safety, and planning efficiency in constrained parking scenarios.**
