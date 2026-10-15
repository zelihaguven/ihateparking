# Constrained-MPC Solver Performance

## Implemented optimization

The dashboard and CLI compare two derivative paths for the **same SciPy SLSQP solver**:

- **Finite-difference baseline:** cached/vectorized numerical evaluator; objective and inequality callbacks share the same kinematic rollout for an identical decision vector.
- **CasADi AD:** the same bicycle model, objective and hard inequality constraints are expressed as a parameterized CasADi graph. CasADi supplies the objective gradient and full constraint Jacobian to SciPy SLSQP.

Both paths retain the same acceleration, steering, speed, steering-rate, world-boundary and sampled full-vehicle-polygon clearance constraints. Static obstacle orientation axes and horizon reference arrays are prepared once per controller solve. The shifted feasible-sequence warm start is only an initial guess; the code skips rebuilding the Pure Pursuit seed while that shifted seed is usable. An independent next-footprint collision check still rejects an unsafe command before the pose is applied.

Run `python3 scripts/benchmark_mpc_derivatives.py --runs 5` to compare both modes on identical closed-loop runs and write `artifacts/mpc_demo/derivative_benchmark.json` (runtime output, ignored by Git). The report records per-call setup, function/Jacobian counts, feasibility, collision-guard stops and latency percentiles.

## Paired sandbox benchmark

Measured in the Manus sandbox on **2026-10-15**. Both modes used a 4-step horizon, 12 SLSQP iterations, `dt=0.18 s`, 0.05 m extra clearance, shifted warm start enabled, and the same 2.5 m reverse-in static-obstacle scene. Each mode ran five closed-loop repetitions, yielding **235 command-cycle samples per mode**. Mean and percentiles in the table are computed over all 235 per-command samples pooled across the five runs, not by averaging five run-level percentiles. These results describe this one synthetic scene and execution environment, not a target-hardware real-time guarantee.

| Mode | Goal runs / guard stops | Mean | p50 | p95 | p99 | Max | Calls >20 ms | Calls >50 ms | Calls >180 ms | Mean model setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cached NumPy + finite differences | 5/5; 0 | 24.84 ms | 14.79 ms | 76.00 ms | 84.30 ms | 117.13 ms | 65/235 | 27/235 | 0/235 | 0.10 ms |
| CasADi Jacobians + SciPy SLSQP | 5/5; 0 | 6.01 ms | 5.43 ms | 12.23 ms | 13.51 ms | 23.21 ms | 1/235 | 0/235 | 0/235 | 114.92 ms |

Under this configuration, the CasADi-AD mode reduced pooled **p95 command-cycle latency by 83.9% (6.2×)** and mean latency by 75.8% versus the cached finite-difference path. SLSQP objective evaluations fell from 827 to 167 per maneuver (79.8% fewer); constraint callbacks fell from 957 to 214 per maneuver (77.6% fewer), and cached evaluator misses fell from 1,625 to 167 per maneuver. AD supplied 83 objective gradients and 83 constraint Jacobians per maneuver. Both modes completed all five runs with no MPC guard stops; minimum raw polygon clearances were 0.2586 m and 0.2589 m respectively. Both had zero calls over the selected 180 ms simulation-step budget.

CasADi graph construction/import is a separate **controller setup cost**, not a per-command operation. The five-run mean was 114.92 ms per controller instance. Build the graph before starting the control loop and do not reconstruct it on every step. Setup time is reported separately from the command-cycle measurements.

## What the timing includes—and excludes

Each full command-cycle sample starts before `controller.solve` and ends after the proposed next pose has been checked by the independent footprint collision guard. It includes Python reference/initial-guess preparation, SLSQP, post-solve feasibility evaluation and the synthetic next-pose collision check. It excludes perception, localization, global planning, transport/I/O, operating-system scheduling, actuator communication and physical vehicle response. It also excludes the separate controller/model setup time shown in the table.

The 180 ms value is only the demo’s selected `dt`; it is **not** a suitable real-car deadline. The project architecture’s 20–50 Hz control target corresponds to 50–20 ms periods. Although 234/235 CasADi sandbox samples were below 20 ms, that cannot establish a 50 Hz deadline on a target ECU or for a complete vehicle control cycle; one command cycle reached 23.21 ms. The finite-difference p95 remained above both period references.

The Pure Pursuit comparison can encounter a predicted unsafe next pose; the guard rejects it before it is appended or applied. That is a rejected-command stop, not a physical contact. The MPC result and controller choice do not prove general safety or superiority: this fixed scene has no moving actors, sensing/localization error, delayed data, tire dynamics or actuator lag.

## Callback profiling and derivative checks

The previous path repeatedly rebuilt intermediate rollout and static obstacle data through SLSQP callbacks. The current NumPy baseline now:

1. caches static obstacle orientation axes and horizon reference arrays per solve;
2. vectorizes the midpoint kinematic rollout and tracking-cost terms;
3. evaluates objective and constraints together and memoizes the last decision-vector result so separate SLSQP callbacks reuse work; and
4. records objective/constraint calls, explicit Jacobian calls, cache hits/misses, `nfev`/`njev`, solver setup, optimizer-only time and command-cycle percentiles.

The CasADi form uses a piecewise-smooth maximum over separating-axis gaps. Its objective and constraint values match the NumPy evaluator in regression tests; automatic gradients/Jacobians are checked against central differences at a smooth tested point. The numerical rectangle-support SAT gap matches the project polygon SAT over randomized poses. These checks do not establish differentiability at active-axis switches or prove collision safety between discrete prediction samples. Preserve margins, hard feasibility checks and the independent exact-footprint rejection path.

## Solver path beyond the prototype

- **Near term:** retain the cached finite-difference path as a dependency-light reference and use CasADi AD only as a measured research mode. Validate gradient behavior around separating-axis changes; scale state, control and constraint units; then repeat tests over all six maneuvers, dynamic actors, disturbances, model mismatch and failed/infeasible solves.
- **Primary real-time candidate:** reformulate the bicycle model and constraints as a structured optimal-control problem for **CasADi + acados SQP-RTI**. Its preparation/feedback phases are intended for low-latency NMPC. That is a separate backend port—not a drop-in flag—and must be benchmarked, compiled/configured and validated on the intended target CPU.
- **Alternative:** investigate **OSQP** only inside sequential QPs or another carefully validated local convexification. The current nonlinear bicycle dynamics and polygon-separation constraints are not a direct convex QP. Retain an independent exact-footprint check and safe-stop on infeasible or missed-deadline results.

A missed or infeasible solve must trigger a separately validated safe stop; do not silently switch to an unverified controller. Before vehicle integration, require target-processor worst-case measurements, input freshness checks, vehicle-specific identification, uncertainty-aware margins, swept-volume/continuous-time review, hardware-in-the-loop testing and controlled closed-course validation. This Python simulation remains research/demo software, not vehicle-ready autonomy.

## References

[1] SciPy, `minimize(method='SLSQP')`: <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html>

[2] CasADi documentation: <https://web.casadi.org/docs/>

[3] acados features, including real-time iterations: <https://docs.acados.org/features/index.html>

[4] OSQP constrained MPC example: <https://osqp.org/docs/examples/mpc.html>

[5] Project control-period context: [`real-system-architecture.md`](real-system-architecture.md#runtime-budgets).
