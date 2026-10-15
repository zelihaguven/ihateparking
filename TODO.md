# Website Outcomes

- [ ] **Permanent public website**
  - Host the actual interactive autonomous-parking application as a permanent public website with a stable HTTPS URL.
  - Preserve the real-time FastAPI-backed safety dashboard and live API behavior.

- [x] **Synthetic parking simulation and planner**
  - Provide a synthetic/mock CARLA simulation experience with no implication of a live CARLA or vehicle-hardware connection.
  - Offer all six parking maneuvers: perpendicular, parallel, and angled parking, each in forward and reverse.
  - Visualize and run Hybrid A* path planning for the selected parking maneuver.
  - Apply reverse-aware Pure Pursuit control during simulated trajectory execution.

- [x] **Perception and dynamic safety supervision**
  - Retain lightweight NumPy-based perception within the simulation pipeline.
  - Preserve dynamic-safety supervision with real-time status and intervention visibility.

- [x] **English lab documentation**
  - Present clear English documentation covering architecture, controls, API usage, limitations, and how to use the lab.

- [x] **Research-only boundary**
  - Display prominent research-prototype disclaimers stating that the system is not production-certified or intended for real-vehicle operation.

- [x] **Interactive constrained-MPC web demo**
  - “Turn the simulation results and script into an interactive web demo or dashboard.”
  - Provide browser controls for bounded MPC configuration, run the existing Python simulation through a same-origin API, and visualize both Pure Pursuit and constrained-MPC results in the same scenario.
  - Include an on/off shifted warm-start option that uses the prior feasible control sequence only as an initial guess; retain the hard MPC constraints and independent executed-footprint check.
  - Show completion/collision-guard outcome, clearance, solver timing, and whether the measured solve fits the selected simulation-step threshold; label the 2.5 m reverse-in world as synthetic and static-obstacle only.
  - Keep the feature in English, link it from the existing lab console, and synchronize its page path in `manus-routes.json`.

- [x] **Solver performance and real-vehicle integration guidance**
  - “How can we optimize the SLSQP solver performance or use alternative optimization libraries to bring solve time below the 180ms threshold?”
  - Measure the current and selectable SLSQP configurations, report actual latency and feasibility, and do not claim sub-180 ms unless the measured result supports it.
  - “This algorithm will be used in the real autonomous car”: explain candidate optimization paths and the validation boundary; distinguish the 180 ms simulation comparison from the project’s 20–50 Hz control target and do not present the current Python simulation as ready to control a real vehicle.


- [x] **SLSQP callback and derivative optimization**
  - “Profile callbacks and reduce callback work, then benchmark analytic/automatic-differentiation Jacobians.”
  - Compare the current finite-difference SLSQP path against an optional CasADi automatic-differentiation Jacobian path on the same synthetic scenario and configuration.
  - Report setup, objective/constraint/Jacobian evaluation counts, optimizer-only time, full controller-call p50/p95/p99/max, feasibility, clearance, safety-stop outcome and simulation-step misses.
  - Preserve hard polygon constraints, the independent executed-footprint collision check and safe-stop behavior; make no real-time claim without target-hardware evidence.

- [ ] **Push recent project code to GitHub**
  - “Do not forget for pushing the recent codes to GitHub.”
  - Synchronize the completed project source to the existing private `zelihaguven/ihateparking` repository’s default branch without rewriting or force-pushing its history, and keep the separate Manus project repository intact.
