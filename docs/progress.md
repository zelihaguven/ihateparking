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

## Done

- Bay centring between neighbours (aa85b60): dev 45/60 unchanged, median centre error 26 -> 22 cm.

## Failure classification (dev set, 15 failed lots)

In almost every NO_SPACE_FOUND lot the car had fully **seen** free bays; the detector rejected them.

1. **Angled bays are never detected**, even fully seen and flanked on both sides (6003, 6016,
   6018, 6037, 6046). Cause (hypothesis, being verified): in angled rows neighbours are staggered
   along the bay axis by width/tan(angle) (1-2.7 m), so one neighbour pokes past the bay mouth
   (read as a wall continuing) and the other sits back (fails the per-half flank coverage); the
   kerb meets the bay obliquely, failing the 0.6 m near-back rule.
2. **Lots with no two-flanked free bay** (6010, 6048, 6059, partly 6001/6006): free bays come in
   runs, which the two-flank rule rejects by design. Needs row/lattice evidence.
3. Wrong park 6055: parallel bay in a long gap.

## Next steps (PRD step 1)

1. Make side-by-side rules tolerate angled-row stagger and oblique kerbs; check detector precision stays 100%.
2. Long parallel gaps: infer bay pitch/phase from parked cars in the row instead of snugging.
3. Classify every NO_SPACE_FOUND on the dev set (free bay never seen / seen but rejected /
   unreachable) with figures, then fix the largest class.
4. Repeat until dev ≥ 95% with 0 wrong parks, then run acceptance set 7000–7099 once.
