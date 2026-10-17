# Scenario sweep: 100-seed success-rate benchmark

`ihateparking.scenario_sweep` plans and executes every manoeuvre kind over a
seed range and records the README experiment-protocol metrics. It is the
regression gate for planner and controller changes.

```bash
python3 -m ihateparking.scenario_sweep --seeds 100 --workers 8
```

Results go to `artifacts/scenario_sweep.json` (runtime output, ignored by Git)
with one record per case and a per-kind summary. The command exits non-zero if
any case ends with a collision-guard stop.

## What one case does

1. `ScenarioGenerator` builds the seeded scene (slot width, lateral offset,
   lane position and, for angled bays, the bay angle all vary with the seed).
2. `HybridAStar` plans from the start pose with the benchmark profile
   (`PlannerConfig(max_expansions=30_000)`).
3. `simulate_closed_loop` executes the plan with the default tracker. Every
   proposed pose is checked against the world before it is applied, so a
   collision is recorded as a rejected command (`guard_stop`), never as contact.
4. The case is `parked` only if the final pose is within 0.35 m and 12° of the
   rear-axle goal; otherwise it is `missed_goal` or `no_plan`.

Minimum clearance is the exact footprint-to-obstacle polygon distance along the
executed trace (every second 5 cm control step plus the final pose).

## Results (2026-10-15)

600 cases (6 kinds × seeds 0–99), Apple M4 Pro, 8 worker processes,
151 s wall time. Plan times are wall-clock and machine dependent; everything
else is deterministic for the seed range.

| Manoeuvre | Parked | Planned | Collisions | Final pos. err. (median / p95) | Final yaw err. (median / p95) | Min clearance | Median time | Plan time (median / max) |
|---|---|---|---|---|---|---|---|---|
| perpendicular_forward | 100% | 100% | 0 | 0.023 / 0.030 m | 0.4 / 0.7° | 1.136 m | 14.1 s | 0.00 / 0.0 s |
| perpendicular_reverse | 100% | 100% | 0 | 0.051 / 0.076 m | 0.7 / 1.5° | 0.234 m | 24.5 s | 0.01 / 0.0 s |
| parallel_forward | 99% | 99% | 0 | 0.027 / 0.048 m | 2.0 / 3.2° | 0.171 m | 26.9 s | 5.11 / 19.8 s |
| parallel_reverse | 99% | 100% | 0 | 0.027 / 0.042 m | 2.1 / 3.6° | 0.172 m | 26.5 s | 4.93 / 14.4 s |
| angled_forward | 100% | 100% | 0 | 0.038 / 0.046 m | 1.1 / 2.5° | 1.766 m | 12.5 s | 0.00 / 0.0 s |
| angled_reverse | 100% | 100% | 0 | 0.026 / 0.032 m | 0.4 / 0.8° | 0.163 m | 26.0 s | 0.02 / 0.1 s |
| **all** | **99.7%** | **99.8%** | **0** | 0.028 / 0.069 m | 0.9 / 2.7° | 0.163 m | 24.8 s | 0.01 / 19.8 s |

The two unsuccessful cases both ended safely:

- `parallel_forward` seed 91: no path within 30 000 expansions (`no_plan`; the
  vehicle never moves).
- `parallel_reverse` seed 84: the plan was executed without contact, but the
  final pose ended just outside the 0.35 m / 12° tolerance (`missed_goal`).

## Before and after

The same protocol on the code before this change (seeds 0–19, 30 000
expansions, original planner and controller):

| Manoeuvre | Parked before | Guard stops before | Parked after (seeds 0–99) |
|---|---|---|---|
| perpendicular_forward | 13/20 | 7 | 100/100 |
| perpendicular_reverse | 7/20 | 10 | 100/100 |
| parallel_forward | 0/20 | 6 | 99/100 |
| parallel_reverse | 0/20 | 3 | 99/100 |
| angled_forward | 14/20 | 6 | 100/100 |
| angled_reverse | 1/20 | 7 | 100/100 |
| **all** | **35/120 (29%)** | **39** | **598/600 (99.7%)** |

Before the change, none of the `perpendicular_reverse` plans used reverse gear,
because the goal was defined nose-in, and parallel planning took up to 23 s per
attempt.

## What changed

| Problem | Fix |
|---|---|
| Reverse kinds parked nose-in; `parallel_reverse` even had to turn 180° in the lane | Reverse kinds now flip the slot heading so the goal faces out of the bay (tail-in) |
| Some `angled_reverse` seeds started inside a parked car | The generator slides a blocked start pose along the lane until it has 0.30 m clearance |
| Plans ended anywhere within 0.35 m / 12° of the goal | Reeds-Shepp analytic expansion connects search nodes to the **exact** goal; a tolerance hit is kept only as a fallback after a bounded extra search |
| Reverse Pure Pursuit negated the arc curvature, which diverges | Curvature is gear-independent for a rear-axle reference; only the heading term flips |
| Lookahead crossed gear changes; the final approach oscillated between gears while rotating in place | `PlanTracker` drives one gear segment at a time, ramps speed down and stops at each cusp, and judges only the final pose |
| Lookahead-only steering rounded off short, sharp parallel shuffles | The default law is planned-curvature feedforward plus lateral/heading feedback (`tracking_law="feedforward"`); `"pure_pursuit"` remains available |
| Plans grazed obstacles at exactly the world margin | The search keeps an extra 0.05 m `clearance_margin` |
| Collision checks cost about 140 µs each (NumPy on four-point arrays) | An arithmetic oriented-box SAT costs about 2.8 µs and matched the polygon test on 40 000 random poses; exactly axis-aligned poses touching a boundary can differ at float rounding (~1e-16 m) |

## Limitations

- These are synthetic, static scenes with an ideal kinematic bicycle model: no
  localization error, actuator lag, tyre slip or moving actors.
- Parallel planning is the expensive case (median about 5 s, worst case about
  20 s in Python). The live dashboard caps the search at 12 000 expansions to
  bound reset latency, so it can fail on parallel seeds that the benchmark
  profile solves (for example `parallel_reverse` seed 84 ends in a safe stop
  there). The search runs outside the session lock, so `/api/state` keeps
  answering while a reset plans; ticks hold the vehicle still until the plan
  is ready. The dashboard plans to its own 0.25 m / 7° capture tolerance.
- On the live dashboard, parallel scenes still end in a safe stop: the NumPy
  perception checkpoint labels the forward parked car as a dynamic obstacle
  (confidence 0.50), and the dynamic-safety supervisor brakes. This is a
  perception false positive; the frozen checkpoint was not retrained here.
- If the exact goal is blocked (for example by a perceived actor that the
  dynamic replanner inserts), the planner skips the Reeds-Shepp shot and falls
  back to the tolerance-only search.
- A missing plan means "no route found within the budget", not proof that no
  route exists.
