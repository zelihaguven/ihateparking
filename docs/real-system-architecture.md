# ihateparking — Real Autonomous Parking Stack

## Product definition

`ihateparking` is not a tutorial animation and not a direct end-to-end black-box driving policy. The target product is an **AI-assisted, safety-supervised autonomous parking stack** that can consume vehicle sensors, estimate the local scene, plan a kinematically feasible manoeuvre, and produce actuator-ready commands through a vehicle adapter.

The animation remains as an explanatory demo. It is not part of the safety-critical control loop.

## System boundary

```text
RGB / depth / LiDAR / odometry / steering feedback
                    |
                    v
        AI perception and uncertainty
                    |
                    v
       temporal tracking + state estimation
                    |
                    v
        local BEV occupancy representation
                    |
                    v
         parking behavior state machine
                    |
                    v
        Hybrid A* / Reeds-Shepp planner
                    |
                    v
             MPC / geometric control
                    |
                    v
           safety supervisor / watchdog
                    |
                    v
       ROS2 or vehicle-specific actuator adapter
```

The key engineering decision is deliberate: **AI estimates the scene; deterministic geometry and safety logic decide whether a manoeuvre is physically valid.** The AI is not allowed to bypass collision checks or directly command a real vehicle.

## AI contribution

### Perception model

The first research model should be a multi-head perception network trained on CARLA-generated data and later evaluated on real footage:

- parking-slot / curb segmentation,
- drivable-space segmentation,
- obstacle detection and coarse shape estimation,
- optional depth or monocular occupancy prediction,
- confidence and out-of-distribution score.

A lightweight encoder-decoder such as SegFormer-B0, Fast-SCNN or a comparable mobile segmentation backbone is a better first fit than a generic object detector because the planner needs free space, slot boundaries and obstacle geometry. A separate detector can be added for vehicles and pedestrians.

### AI output contract

Every estimate must carry:

- timestamp and sensor frame,
- class/type,
- geometry in a calibrated vehicle frame,
- confidence,
- uncertainty or quality score,
- track ID when temporal association is available.

Low confidence must widen safety margins or trigger a controlled stop. It must never be silently converted into a confident obstacle-free map.

## Deterministic autonomy layers

### 1. Temporal state estimation

Fuse AI geometry with wheel/steering/odometry and visual motion. Use an EKF or factor-graph style estimator to maintain:

- vehicle pose in the local frame,
- longitudinal speed,
- yaw rate,
- steering state,
- covariance / quality.

### 2. Local BEV map

Maintain a rolling local map with:

- static occupied cells,
- dynamic obstacle tracks,
- drivable-space probability,
- parking-slot candidates,
- uncertainty inflation,
- vehicle footprint and configured safety margin.

### 3. Behavior state machine

The behavior layer makes the system inspectable and recoverable:

```text
SEARCH_SLOT
  -> APPROACH_SLOT
  -> ALIGN_FOR_MANOEUVRE
  -> PLAN_MANOEUVRE
  -> EXECUTE_FORWARD
  -> EXECUTE_REVERSE
  -> STRAIGHTEN
  -> VERIFY_PARKED
  -> COMPLETE
```

Any state can transition to `SAFE_STOP` when perception quality drops, a dynamic obstacle enters the corridor, the vehicle deviates beyond tolerance, or a watchdog expires.

### 4. Motion planning

Use Hybrid A* or Reeds-Shepp primitives with the calibrated vehicle footprint and steering limits. Replan at a fixed rate and invalidate the current plan when the occupancy map changes. The planner must report:

- path feasibility,
- minimum predicted clearance,
- gear sequence,
- curvature / steering profile,
- planning time,
- reason for rejection.

### 5. Control

Use MPC as the research controller, with Pure Pursuit or Stanley as a baseline. The controller consumes a short local trajectory, not raw neural-network output. Enforce limits on speed, steering, steering rate, acceleration and jerk.

### 6. Safety supervisor

The safety supervisor is independent from the AI model and planner. It checks the next command against:

- current occupancy and uncertainty inflation,
- predicted footprint sweep,
- minimum clearance,
- stale sensor timestamps,
- state-estimator covariance,
- actuator command limits,
- emergency-stop conditions.

The only safe fallback is a controlled stop; it must not invent a recovery manoeuvre when the scene is unknown.

## Scenario suite

The benchmark must grow beyond one hand-authored slot:

1. perpendicular forward-in parking,
2. perpendicular reverse-in parking,
3. reverse parallel parking,
4. angled parking,
5. narrow slots and asymmetric neighboring vehicles,
6. partial occlusion and missing slot markings,
7. pedestrians or carts crossing the corridor,
8. night, rain, glare and low contrast,
9. camera noise, delayed frames and dropped frames,
10. start-pose and calibration errors.

Each scenario is seeded and replayable. The same scenario must be available in closed-loop simulation and, where possible, as a recorded sensor bag.

## Interfaces for real integration

The software should expose ROS2-compatible concepts even when the first local runner is a Python adapter:

```text
/perception/parking_slots
/perception/obstacles
/perception/occupancy_grid
/state/vehicle
/planning/trajectory
/control/command
/safety/status
```

The real vehicle adapter must be isolated from the planner. A CARLA adapter, rosbag replay adapter and real camera/CAN adapter should all implement the same input/output contracts.

## Runtime budgets

Initial target budgets are explicit and measurable, not implied by a demo:

- AI perception: 10–15 Hz,
- state estimation: 30–50 Hz,
- replanning: 5–10 Hz,
- control: 20–50 Hz,
- safety supervisor: every control cycle,
- stale-input watchdog: less than one control period plus a configured grace window.

The benchmark records end-to-end latency, not only model FPS.

## Implementation phases

### Phase 0 — contracts and replayable core

- typed sensor/perception/state/command contracts,
- deterministic safety checks,
- scenario seed and recording format,
- current Hybrid A* and controller behind the new interfaces.

### Phase 1 — realistic simulator adapter

- CARLA parking maps and sensor streams,
- camera calibration and vehicle dimensions,
- scenario randomization,
- RGB/depth/LiDAR/odometry recording,
- closed-loop collision and latency metrics.

### Phase 2 — AI perception

- generate labelled CARLA data,
- train slot/free-space/obstacle model,
- export inference model,
- calibrate confidence and test domain shift,
- compare AI estimates with ground truth before planner integration.

### Phase 3 — integrated autonomy

- connect AI output to BEV map,
- add EKF/state estimation,
- add behavior state machine,
- add MPC and safety supervisor,
- evaluate randomized scenario suite.

### Phase 4 — real-vehicle adapter

- camera/IMU/odometry calibration,
- rosbag replay before live operation,
- actuator command sandbox,
- low-speed closed-course validation only,
- hardware emergency stop and operator override.

## Current honest status

The current repository has a working 2-D planner/controller and a generated instructional video. It does **not** yet have CARLA, ROS2, calibrated perception, a learned model, or real actuator integration. The next code checkpoint must add the contracts and simulator adapter rather than make the current animation look more realistic.
