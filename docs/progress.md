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

## Next steps

1. Parallel scenario with scan matching (9000–9059, running).
2. Parallel-parking test set.
3. Fresh acceptance set (7100–7199) with scan matching.
