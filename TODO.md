# Website Outcomes

- [ ] **Permanent public website**
  - Host the actual interactive autonomous-parking application as a permanent public website with a stable HTTPS URL.
  - Preserve the real-time FastAPI-backed safety dashboard and live API behavior.

- [ ] **Synthetic parking simulation and planner**
  - Provide a synthetic/mock CARLA simulation experience with no implication of a live CARLA or vehicle-hardware connection.
  - Offer all six parking maneuvers: perpendicular, parallel, and angled parking, each in forward and reverse.
  - Visualize and run Hybrid A* path planning for the selected parking maneuver.
  - Apply reverse-aware Pure Pursuit control during simulated trajectory execution.

- [ ] **Perception and dynamic safety supervision**
  - Retain lightweight NumPy-based perception within the simulation pipeline.
  - Preserve dynamic-safety supervision with real-time status and intervention visibility.

- [ ] **English lab documentation**
  - Present clear English documentation covering architecture, controls, API usage, limitations, and how to use the lab.

- [ ] **Research-only boundary**
  - Display prominent research-prototype disclaimers stating that the system is not production-certified or intended for real-vehicle operation.
