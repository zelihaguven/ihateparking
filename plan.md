# ihateparking Permanent Website — Plan

## Product and implementation

Publish the existing `ihateparking` autonomous-parking research lab as a stable HTTPS website on Manus-managed hosting. Reuse its working FastAPI dashboard and simulation stack rather than replacing it with a static mockup. The public site must make clear that this is a synthetic research prototype; it is not a live CARLA connection, production-certified autonomy software, or a real-vehicle control interface.

The browser dashboard remains the main experience. The existing FastAPI process serves the HTML, CSS and JavaScript assets and JSON API from one origin. The container runs one application process, listens on the platform-provided `PORT` (default 3000), and exposes an unauthenticated health endpoint. It uses the frozen NumPy perception checkpoint already exported in `deployment/export`. Generated sensor artifacts go to a temporary runtime directory; no database or cross-session run history is included because the persistent-lab-data option was not selected. The live demo is a shared simulation session, and in-memory telemetry can reset on process replacement or idle wake.

Keep same-origin, relative API paths so Preview and production work behind the platform proxy. Publish as a server container using the project Dockerfile and `/api/health`; do not depend on the sandbox URL or a user-owned machine for hosting. Use the platform's ordinary hosting default unless the owner separately requests an always-on option; the stable website URL may have a cold start after inactivity.

## Project structure

- `ihateparking/`: simulator-agnostic scenario generator, geometry, Hybrid A*, controller, perception contracts and NumPy model, mock-CARLA sensor loop, dynamic safety supervisor, and FastAPI dashboard.
- `web/`: English dashboard and its responsive styles, browser-side rendering, planner replay, human-in-the-loop video assist, and virtual lesson.
- `assets/`: source clips required by the video-assist and generated virtual-lesson pages.
- `artifacts/`: small, checked-in replay and video-assist signal JSON inputs served by the website; transient simulation outputs are written to `/tmp`.
- `deployment/export/`: frozen NumPy perception checkpoint, planner profiles, reproducible scenario suite and manifest.
- `docs/`: CARLA boundary, real-system architecture, footage and research limitations.
- `tests/`: regression tests for geometry, planner, scenarios, perception, API session and safety behavior.
- `Dockerfile`: self-contained Python runtime image, model and web assets, health check, unprivileged service user, and platform `PORT` binding.
- `manus-routes.json`: public page-route manifest, excluding APIs and assets.
- `README.md`: English project, API, limitations, and deployment documentation.

## Design

### Design movement

An instrumented autonomy-lab console: an automotive human-machine interface adapted for engineering review, not a consumer driving product.

### Core principles

1. Make safety state and supervisor decisions more visually prominent than motion or performance metrics.
2. Put the simulation evidence first: a large top-down world view, an explicit planned trajectory, and inspectable event telemetry.
3. Prefer restrained, high-contrast colors with consistent meanings for ego, path, slot, perception and hazard.
4. State simulation and research boundaries beside the controls, not only in supporting documentation.

### Color philosophy

A near-black blue-green foundation keeps the scene and telemetry calm and legible. Cool cyan identifies the ego vehicle and live connection; pale mint marks a valid plan or safe progression; amber marks the parking target and attention states; coral-red is reserved for moving hazards and braking; violet outlines learned detections. The signature brand color is **signal cyan `#72D8DC`**—an ownable sensor/status accent rather than a generic corporate blue.

### Layout paradigm

Use a telemetry-first instrument layout rather than a conventional marketing landing page: a compact status header and navigation, a full-width research warning, then a wide simulation canvas beside a narrower supervisor/perception/vehicle rail. Place a concise operator guide, API reference and limitations beneath the live instrument. At narrow widths, stack the canvas and status rail in their diagnostic order, and let navigation wrap instead of shrinking the canvas.

### Signature elements

- A bright live-connection indicator paired with the autonomy-mode badge.
- A mint dashed Hybrid A* trajectory ending at an amber outlined slot.
- Coral safety-event rows that make `BRAKE` and `REPLAN` impossible to mistake for ordinary telemetry.

### Interaction philosophy and animation

Controls remain explicit and reversible: reset chooses a deterministic scenario, single-step advances one supervised cycle, and run/pause toggles closed-loop execution. Never bypass an emergency brake to make the animation look successful. Motion only advances when the visitor requests it, and the event log explains interventions. Keep decorative transitions short and subtle; the simulation itself is the only continuous movement. Honor reduced-motion preferences and avoid pulsing beyond the existing low-amplitude connection/status indicator.

### Typography system

Use the existing system sans-serif stack for explanatory copy and compact IBM Plex Mono/system-monospace fallbacks for frame IDs, numerical telemetry, API paths and event timestamps. Headings use tight tracking and clear scale contrast; small all-caps mono labels are reserved for instrumentation categories, never paragraphs.

### Brand essence and voice

**Positioning:** An independent autonomous-parking lab for students and autonomy engineers, distinguished by exposing the full plan–control–safety loop in a runnable simulation. Personality: **curious, exacting, candid**.

Headlines and CTAs are short, technical and transparent. Example lines: “See the safety envelope move.” and “Reset the scene. Inspect the next decision.” Microcopy calls out uncertainty and simulator boundaries instead of implying real-world competence.

### Wordmark and logo

Keep the distinctive lowercase `ihateparking` wordmark beside a compact rounded parking-bay/P mark, with a cyan outline, mint glyph and amber proximity marker matching the dashboard legend. Do not use a stock vehicle silhouette.

## Runtime and deployment constraints

- Bind to `0.0.0.0:$PORT`, defaulting to port 3000; keep the configured WebDev Preview port unchanged.
- Keep `/api/health` available without authentication for readiness.
- Serve the public `manus-routes.json` manifest at the website origin.
- Keep browser API URLs relative and allow HTTPS Preview framing; do not add a restrictive `X-Frame-Options` or `frame-ancestors` header.
- The public container filesystem and in-memory simulation state are ephemeral. Do not claim persistent saved runs or isolated per-visitor sessions.
- Preserve English planner replay, footage-assist and self-generated tutorial pages alongside the real-time console.
- Validate the repository regression suite, application syntax, health endpoint, route manifest, and actual maneuver reset/tick flow before saving the canonical website checkpoint and publishing it.

## Interactive constrained-MPC lab extension

Add a dedicated English page at `/web/mpc-lab.html`, linked from the existing console, and keep its API same-origin. The page reuses the reproducible constrained-MPC simulation in `scripts/demo_constrained_mpc.py`; it is not a separate browser-side approximation. Visitors can adjust the prediction horizon, simulation time step, SLSQP iteration limit, extra clearance, and shifted warm-start toggle, then request a paired Pure Pursuit/MPC run. The page renders both trajectories over the same static 2.5 m reverse-in scene and reports success, collisions, minimum clearance, optimizer-only and full controller-call latency, and the 180 ms demonstration threshold.

Expose a bounded `POST /api/mpc/run` request through FastAPI. Keep expensive optimizer work serialized and off the event loop, reject concurrent requests rather than queueing unbounded CPU work, and return transient simulation results only; do not introduce a database or imply saved run history. Include scene geometry and trajectory points needed by the browser view. Keep the Python script independently runnable and use the same simulation/controller implementation in the API path.

The initial selectable solver remains SciPy SLSQP. Show measured timing and feasibility for the exact selected configuration, and distinguish an observed pass on this synthetic scenario from a real-time guarantee. The accompanying solver note and dashboard should identify gradient/Jacobian cost, warm-starting and problem-size reductions as SLSQP investigation directions. Treat CasADi with acados SQP-RTI as the primary nonlinear real-time candidate and OSQP as a candidate only after a safe local convexification/QP formulation; neither alternative is claimed implemented or deadline-compliant until benchmarked on the intended hardware. The existing architecture target of 20–50 Hz means 20–50 ms control periods; 180 ms is only the current simulation-step comparison, not an acceptable real-vehicle deadline.

Reuse the existing dark autonomy-console design, brand colors, responsive layout, typography and visible simulation-only notice. Add the MPC page link to `manus-routes.json`. Keep the output ephemeral and the research/safety boundary prominent: no real CARLA, ROS 2, vehicle hardware or actuator connection is added.

The implementation touches `ihateparking/dashboard.py` for the bounded API route, `scripts/demo_constrained_mpc.py` for reusable comparison output, `web/mpc-lab.html`, `web/mpc-lab.js` and `web/mpc-lab.css` for the interactive page, `Dockerfile` and `pyproject.toml` for the MPC runtime dependency, `manus-routes.json` for the page declaration, and `docs/mpc-solver-performance.md` for measured results and sourced solver guidance.
