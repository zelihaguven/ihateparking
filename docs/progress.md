# Progress log

Where the work stands against [prd.md](prd.md), so a new session can resume without redoing anything.

## Seed sets (do not mix them up)

| Set | Seeds | Use |
|-----|-------|-----|
| Baseline | 5000–5099 | Measured once before PRD step 1. Do not tune on it. |
| Development | 6000–6059 | Failure analysis and comparing fixes. |
| Acceptance | 7000–7099 | Untouched. Run only for the final PRD check. |

Run a set: `python -m ihateparking.mission --benchmark FIRST:LAST --workers 12 --json <path>`
(about 17 min for 60 lots on a 14-core Mac).

## Measurements (commit 5e82601 code + centre-error metric)

| Metric | PRD target | Baseline 5000–5099 | Dev 6000–6059 |
|--------|-----------|--------------------|---------------|
| Correct park | ≥ 95% | 74% (74/100) | 75% (45/60) |
| Wrong place | 0 | 1 (5026, parallel) | 1 (6055, parallel) |
| Collisions | 0 | 0 | 0 |
| NO_SPACE_FOUND / TIMEOUT | — | 24 / 1 | 13 / 1 |
| Median / max centre error | ≤ 20 cm | 21 / 129 cm | 26 / 84 cm |
| Median / max axis error | ≤ 5° | 1.9 / 11.2° | 1.7 / 9.0° |
| Median / max time when parked | ≤ 90 / 240 s | 41 / 255 s | 48 / 264 s |

Raw results: [benchmarks/](benchmarks/).

## Findings

1. **Centre error comes from target choice, not tracking.** The final pose lands on the
   detector's target; the target sits off the true bay centre (e.g. seed 5047: target 54 cm
   off, final 54 cm off). The detector keeps the best-scoring lattice anchor instead of the
   middle of the gap between neighbours.
2. **Both wrong parks are parallel bays** in long gaps, where "snug behind the end car"
   straddles two bays.
3. **NO_SPACE_FOUND is the biggest loss** (~24%). Not analysed per lot yet.

## Done (dev set 6000–6059, full missions)

| Commit | Change | Correct park | Wrong | Median / max centre error | Median / max time |
|--------|--------|--------------|-------|---------------------------|-------------------|
| 6193952 | baseline | 45/60 (75%) | 1 | 26 / 84 cm | 48 / 264 s |
| aa85b60 | centre bays between neighbours | 45/60 | 1 | 22 / 84 cm | 47 / 264 s |
| de3afc3 | detect bays in angled rows | 53/60 (88%) | 1 | — | — |
| 0f81706 | more row directions; parallel bays pulled to the kerb | 54/60 (90%) | 0 | 16 / 51 cm | 39 / 187 s |
| e4b2013 | bays in runs of free bays (row lattice) | 55/60 (92%) | 0 | 15 / 47 cm | 35 / 187 s |
| 042dc32 | recovery to seen-free space; waypoints to far frontiers | **58/60 (97%)** | 0 | 18 / 47 cm | 42 / 181 s |

Detector quality is checked in seconds on aisle-sweep maps (dev 6000–6119, held-out 8000–8199):
e4b2013 is 208/208 and 327/327 correct (precision 100%).

## Open failures (dev)

- 6010: no two-flanked free bay; runs of free bays in **angled** rows are not yet placed
  (row lattice works for perpendicular rows only so far).
- 6035: angled run of two free bays; a parallel bay closed by a car on one end and a wall on
  the other is rejected; recovery repeats to the same open spot.

## Acceptance (042dc32, unseen lots 7000–7099, run once)

| PRD metric | Target | Result | |
|---|---|---|---|
| Correct park | ≥ 95% | **99/100** (1 NO_SPACE_FOUND, stopped safely) | met |
| Wrong place | 0 | 0 | met |
| Collisions | 0 | 0 | met |
| Centre error | ≤ 20 cm | median 19.3 cm, max 55 cm; 48 of 99 parks above 20 cm | met as a median only |
| Axis error | ≤ 5° | median 2.2°, max 14.1°; 16 of 99 parks above 5° | met as a median only |
| Time when parked | ≤ 90 / 240 s | median 37 s, max 224 s | met |

Bays parked in: 63 perpendicular, 36 angled, 0 parallel (parallel parking is barely exercised).

## PRD step 2: pose estimation (branch feat/pose-estimation)

`--localization truth | odometry | scan_matching`. The true pose drives the simulation, sensor,
guard and scoring; the stack sees only the estimate.

| Step | Commit | What |
|------|--------|------|
| 2.1 | 153cfce | noisy wheel/steering odometry (scale error, steering bias, noise) |
| 2.2 | 3e5eaaf | mission runs on the estimate |
| 2.3 | 0cf2693 | scan-to-map matching corrects the estimate every scan |

Dev 6000–6059, odometry alone: 58/60, 0 wrong, 0 collisions, but centre error median 24 cm /
p95 46 cm, longest mission 351 s, estimate off by up to 10 m — fails the PRD alignment and time
targets. Scan matching on 6 drifting lots: estimate error 0.18–1.65 m -> 0.03–0.21 m.

Dev 6000–6059 with scan matching (0cf2693): 57/60, 0 wrong, 0 collisions, centre error median
19 cm / p95 38 cm, axis 2.2° / 7.2°, longest 214 s, estimate error median 6 cm / max 59 cm. All
PRD targets met on dev, success right at 95%.

## Parallel parking (scenario `--scenario parallel`, seeds 9000–9059, scan matching)

| Commit | Correct park | Wrong |
|--------|--------------|-------|
| 0cf2693 | 6/59 | 1 (9021: parked past the end of a row) |
| 7428a63 | 23/59 | 0 |
| a9d9a69 | 32/59 | 1 (9016: kerb was the side of a perpendicular car) |
| ffc2211 | **34/59** | **0** |

ffc2211 (parallel scenario): 0 collisions; axis error median 0.8° / p95 5.0°; centre error
median 42 cm / p95 109 cm, almost all along the bay (a 4.4 m car in a 6.2–7.0 m bay; bay ends
known only from parked cars that sit up to 1 m off centre; across the bay p95 is 25 cm); longest
mission 289 s. Remaining failures: 25 NO_SPACE_FOUND (mostly free bays only at the wall end of a
row), 1 TIMEOUT.

Mixed dev 6000–6059 at a9d9a69 (scan matching): 58/60, 0 wrong, 0 collisions; centre error p95
63 cm because the car now often picks the nearer parallel bay (missions are faster: parallel
parks in 25–37 s instead of 80–150 s).

PRD questions raised by this: centre error along a parallel bay, and the 240 s limit when the car
must explore a whole lot to find its only kind of free bay.

Failures at 7428a63: 22 lots whose free bays are only at the wall end of a row or in runs with
parked cars on one side, 13 lots with a free bay between two cars (being diagnosed: mostly
detected but rejected as "no route", i.e. the parallel manoeuvre is not planned), 1 lot with
no free bay.

### Wall-end bays (user decision b: park like a driver) — tried, not adopted

Detector precision must stay 100% (one false detection is a wrong park). Measured with
`scripts/eval_detector.py` on dev 6000–6119, held-out 8000–8199 and parallel 9000–9059:

1. Extend the kerb lattice towards a wall across the kerb: no gain (rows at the wall end show
   only 1–2 whole cars; the lattice needs 3), and it mislabelled some car ends as walls.
2. Accept any one-car gap closed at both ends: 26 + 54 false detections in mixed lots (a car
   placed sideways across perpendicular bays).
3. + require an unbroken kerb: 17 + 43 false.
4. + require each end to be a car parked along the kerb or a wall across it: 3 + 5 false, for
   +1 correct parallel detection.

None is safe enough. Next idea if revisited: decide the bay kind of a whole row first
(parallel vs perpendicular, from the parked cars' orientation), then apply gap rules only in
rows known to be parallel.

## Against the PRD rules of 2026-10-29 (57ad0bd, scan matching)

Position error = centre error for perpendicular/angled bays, lateral error for parallel bays.
Park time = from committing to the bay to standing in it.

| Metric | Target | Mixed dev 6000–6059 | Parallel 9000–9059 |
|---|---|---|---|
| Correct park | ≥ 95% | 58/60 (97%) ✅ | 34/59 (58%) ❌ |
| Wrong park / collisions | 0 / 0 | 0 / 0 ✅ | 0 / 0 ✅ |
| Overlap, every park | ≥ 80% | min 89.6% ✅ | min 81.7% ✅ |
| Position error median / p95 | ≤ 20 / 40 cm | 17 / 39 cm ✅ | 20.4 / 26 cm (median 0.4 cm over) |
| Axis error median / p95 | ≤ 5° / 10° | 1.8° / 7.2° ✅ | 0.8° / 5.0° ✅ |
| Total time median | ≤ 90 s | 40 s ✅ | 72 s ✅ |
| Park time worst | ≤ 240 s | 50 s ✅ | 91 s ✅ |

Both are development sets (tuned on); the step 2 sign-off needs a fresh acceptance run.

## Step 2 acceptance (d4cbb6f, scan matching, unseen mixed lots 7100–7199, run once)

| PRD metric | Target | Result | |
|---|---|---|---|
| Correct park | ≥ 95% | **98/100** (2 NO_SPACE_FOUND, stopped safely) | met |
| Wrong park / collisions | 0 / 0 | 0 / 0 | met |
| Overlap, every park | ≥ 80% | min 90% | met |
| Position error median / p95 | ≤ 20 / 40 cm | **22.7 / 40.2 cm** | **missed** (by 2.7 cm / 2 mm) |
| Axis error median / p95 | ≤ 5° / 10° | 2.0° / 7.3° | met |
| Total time median | ≤ 90 s | 35 s | met |
| Park time worst | ≤ 240 s | 44 s | met |
| Pose estimate error at the end | — | median 5 cm, max 21 cm | |

Bays parked in: 60 perpendicular, 27 angled, 11 parallel. Position error by kind: perpendicular
25 / 38 cm, angled 26 / 45 cm, parallel (lateral) 16 / 26 cm. Step 1 acceptance (truth pose, lots
7000–7099) had perpendicular 19 / 40 cm and angled 21 / 38 cm; different lots, so how much of the
rise comes from pose estimation is not separated yet.

### Where the position error comes from

- Same lots 7100–7199 with the true pose: 96/100 correct, **1 wrong park (7154)**, position error
  median 20.6 / p95 39.1 cm. On the 65 parks into the same bay, pose estimation adds 0.7 cm on
  average: localization is not the cause.
- 7154 was a run bay placed from a lattice aliased onto the edge of the width range (2.45 m for a
  2.83 m row); fixed in 98c0062 (lattice on a range edge is refused). Lots 7100–7199 were opened
  for this bug, so the next acceptance run uses fresh lots (7200–7299).
- Tracking is not the cause either: on 34 dev missions the car stops 3.6 cm (median, p95 8.4 cm)
  from its target; the target itself is 20 cm (median, p95 37 cm) from the true bay centre.
- Detector target error on dev sweeps, perpendicular / angled: along 11 / 10 cm, lateral 10 / 12 cm
  (medians). With bay lines invisible and parked cars up to ~0.45 m off their bay centre, a ~20 cm
  median is close to the floor of what neighbours can tell; seeing the lines (camera) would lift it.

## Step 2 acceptance, second run (98c0062, scan matching, fresh lots 7200–7299, run once)

| PRD metric | Target | Result | |
|---|---|---|---|
| Correct park | ≥ 95% | **94/100** (5 NO_SPACE_FOUND, 1 TIMEOUT; all stopped safely) | **missed by 1** |
| Wrong park / collisions | 0 / 0 | 0 / 0 | met |
| Overlap, every park | ≥ 80% | min 95.2% | met |
| Position error median / p95 | ≤ 25 / 45 cm | 21.3 / 40.9 cm | met |
| Axis error median / p95 | ≤ 5° / 10° | 2.1° / 7.0° | met |
| Total time median | ≤ 90 s | 39 s | met |
| Park time worst | ≤ 240 s | 61 s | met |
| Pose estimate error at the end | — | median 5 cm, max 74 cm | |

Bays parked in: 58 perpendicular, 34 angled, 2 parallel. The two acceptance runs (98/100 on
7100–7199, 94/100 here) bracket the 95% target; step 2 is not signed off. These lots are not
to be analysed (no tuning on acceptance data).

## Next steps

1. Raise success on the dev set: classify the NO_SPACE_FOUND failures again at the current code
   (dev 6000–6059 and a second dev set 6060–6119), fix the largest class, then a fresh
   acceptance run (7300–7399).
2. Wall-end parallel bays (branch wip/wall-end-parallel): needs row-level evidence before merging.
3. Free Mac simulator (MetaDrive) feasibility check; ROS2 layer.
