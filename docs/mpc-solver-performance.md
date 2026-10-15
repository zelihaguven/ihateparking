# Constrained-MPC Solver Performance

## Measured result

The interactive demo evaluates a synthetic, static-obstacle reverse-in maneuver into a 2.5 m slot. These measurements were made in the Manus sandbox on 2026-10-15 using `time.perf_counter`; they are descriptive results from one scenario, not a target-hardware benchmark. Each closed-loop sample has only 47–48 calls, so percentile estimates are sensitive to run-to-run noise.

| Profile | Outcome | Mean full controller call | p95 | p99 | Calls over 180 ms |
|---|---|---:|---:|---:|---:|
| Horizon 4, 12 iterations, Pure Pursuit seed (`warm_start=false`), direct run | Goal reached; 0 collision-guard stops; 0.2595 m minimum raw clearance | 93.63 ms | 132.97 ms | 155.86 ms | 0 / 47 |
| Horizon 4, 12 iterations, shifted feasible-sequence seed (`warm_start=true`), direct run | Goal reached; 0 collision-guard stops; 0.2587 m minimum raw clearance | 42.06 ms | 121.47 ms | 130.98 ms | 0 / 47 |
| Horizon 4, 12 iterations, shifted seed, second API run | Goal reached; 0 collision-guard stops | 43.64 ms | 125.69 ms | 135.57 ms | 0 / 47 |
| Horizon 4, 12 iterations, Pure Pursuit seed, second API run | Goal reached; one pre-execution collision-guard stop | 107.46 ms | 155.92 ms | 179.53 ms | 1 / 47 |
| Horizon 8, 40 iterations, Pure Pursuit seed, direct run | Goal reached; 0 collision-guard stops; 0.2595 m minimum raw clearance | 353.52 ms | 562.66 ms | 594.24 ms | 48 / 48 |

The shifted initial-guess warm start reduced the mean call time in both observed paired runs; p95 ranged from 121.47 to 125.69 ms. The two cold-seed runs varied more: p95 ranged from 132.97 to 155.92 ms, and one run had a single call exceed the 180 ms budget. The long-horizon reference missed every 180 ms deadline. **These data show that warm starting can help this one case, not that the controller is reliably real-time.**

Even the warm profile’s p95 is above both project control-period references: 50 ms at 20 Hz and 20 ms at 50 Hz. The 180 ms line is only the chosen simulation-step budget; it is not an acceptable real-vehicle target. The project architecture specifies 20–50 Hz control, and the actual deadline must come from the target vehicle’s system design.

The API distinguishes **optimizer-only time** (the `scipy.optimize.minimize` call) from **full controller-call time** (Python setup, optimization and post-solve feasibility checks). The table uses full controller-call time, which is closer to the computation required before a command can be accepted. Re-run the page to measure the selected profile on the current worker.

The paired Pure Pursuit baseline triggered one collision guard in one run: its proposed next pose was rejected before it was appended to the trace. No colliding pose was applied. The constrained MPC reached the goal without a rejected pose in the reported samples. This is evidence about one synthetic path, not proof that MPC is generally safer. The scene has no moving actors, localization error, sensor delay, tire dynamics, actuator lag or real vehicle.

## What the warm-start toggle actually does

When enabled, the controller shifts its previous **feasible control sequence** forward by one step and uses it as the next SLSQP initial guess. The first solve still uses the geometric Pure Pursuit seed. If a solve is infeasible, the stored warm start is cleared. Every solve still evaluates the hard polygon-clearance constraints, and the executed command still passes the independent footprint collision check.

This is an initial-guess warm start only: SciPy SLSQP is still invoked anew each time; the demo does not preserve a solver workspace, Hessian approximation or factorization. It does not guarantee feasibility, convergence or a deadline. The checkbox lets visitors compare it against the fresh Pure Pursuit seed under the same horizon and scenario.

## Improving the current SLSQP baseline

1. **Profile callbacks and evaluations.** Separate objective, constraint, rollout and safety-check costs. Precompute static obstacle polygons/axes and path-reference samples once rather than rebuilding them in every callback.
2. **Use analytic or automatic-differentiation Jacobians.** SLSQP currently relies on numerical differences. Exact/sparse derivatives can reduce repeated Python callbacks, but the active separating axis can switch at polygon features; validate gradients around those transitions and preserve conservative collision margins.
3. **Scale variables and constraints consistently.** Position, yaw, speed, steering and clearance have different units and ranges. Record convergence status and constraint residuals as well as elapsed time.
4. **Keep the shifted warm start, but test it broadly.** Include disturbances, replanning, mode/gear transitions and infeasible solves. Warm start is only a seed; retain the safe-stop path and independent collision checker.
5. **Change problem size cautiously.** Shortening horizon or limiting iterations can reduce compute, but may remove look-ahead or robustness. Test success, swept-volume clearance and deadline misses across all maneuver types, dynamic actors and model-error cases before accepting a trade-off.
6. **Measure p95/p99 and maximum on the target processor.** Include state input, constraint construction, solver, feasibility validation and command output. A low average is not a real-time guarantee.

SciPy documents finite-difference settings and a `workers` map for parallel numerical differentiation in recent releases (added in SciPy 1.16). It may or may not help this small, Python-heavy problem: dispatch overhead and callback/GIL costs need a local benchmark. The demo does not enable `workers` or provide analytic Jacobians. [1]

## Candidate solver paths

**Primary nonlinear candidate: CasADi + acados SQP-RTI.** CasADi provides algorithmic differentiation, sparse derivative information, and building blocks for optimal-control formulations. It is not a drop-in automatic-differentiation wrapper for the current Python functions; the vehicle model/objective/constraints must be expressed in CasADi’s symbolic form. acados supports real-time iteration with preparation and feedback phases intended to reduce control delay. Port the bicycle model, constraints and costs to a structured OCP, then build and benchmark for the target CPU. Neither tool is wired into this repository’s dashboard or shown to meet a deadline. [2] [3]

**Convexified alternative: OSQP inside sequential QPs/SQP.** OSQP’s official MPC example formulates a linear time-invariant model as a constrained QP, then warm-starts and updates the initial-state bounds each receding-horizon step. The current nonlinear bicycle model and polygon-separation constraints are not a direct OSQP problem. A candidate implementation would need safe local linearization/convexification and an independent exact-footprint check; OSQP alone is not a drop-in nonlinear MPC solver. [4]

## Real-vehicle acceptance boundary

The 180 ms comparison is just the current simulation step, **not a suitable real-vehicle target**. The project architecture calls for 20–50 Hz control, corresponding to 50–20 ms periods; the actual deadline must come from the target vehicle’s system design.

Before any vehicle integration, measure the full call path on the exact target hardware under worst-case scene complexity and repeated maneuvers. Record p50, p95, p99, maximum, solver status and deadline-miss rate. A missed/infeasible solve must trigger a separately validated safe-stop, not an unverified fallback controller. Discrete-time polygon checks do not prove continuous-time or swept-volume safety. Also require vehicle-specific system identification, uncertainty-aware margins, stale-input monitoring, actuator limits, hardware-in-the-loop tests and controlled closed-course validation. This Python code remains research/demo software, not vehicle-ready autonomy.

## References

[1] SciPy, `minimize(method='SLSQP')`: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html

[2] CasADi documentation: https://web.casadi.org/docs/

[3] acados features, including real-time iterations: https://docs.acados.org/features/index.html

[4] OSQP constrained MPC example: https://osqp.org/docs/examples/mpc.html

[5] Project control-period context: [`real-system-architecture.md`](real-system-architecture.md#runtime-budgets).
