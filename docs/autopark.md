# Autopark: "park in that bay" in a parking lot

The single-slot scenarios tell the planner which manoeuvre to run. `autopark`
does not: the caller names only the target bay, or none at all, and the
vehicle decides how to get there.

```bash
python3 -m ihateparking.autopark --seed 3 --list      # bays and occupancy
python3 -m ihateparking.autopark --seed 3 --bay C4    # park in C4
python3 -m ihateparking.autopark --seed 3             # nearest free bay
```

Each run prints a JSON summary and writes a figure to
`artifacts/autopark/autopark.png` (`--output` changes the path).

The same logic runs in the web app at `/web/autopark.html`: click a bay (or
use the bay list) and the page shows the options the car weighed, the chosen
manoeuvre and an animated replay. `POST /api/autopark/run` shares the
single-run lock with the other compute-heavy endpoints and answers 429 while
another run is in progress.

## The lot

`ihateparking.parking_lot.build_lot(seed, occupancy=0.5)` builds a 40 × 28 m
lot with 21 bays: 5 parallel bays along the south wall (row A), 9
perpendicular bays (row B) and 7 angled 60° bays behind a separator kerb (row
C). Each bay is occupied by a parked car with probability `occupancy`; the
seed fixes which. The ego starts at the west end of the lower aisle. Row C
can be reached only by driving round the end of the rows, through the west
gap or round the east end.

## Decision procedure

1. **Refuse** an occupied bay or an invalid start, without moving.
2. **Enumerate final headings.** Perpendicular and angled bays allow nose-in
   and tail-in; parallel bays allow facing either way along the kerb. An
   option whose footprint does not fit with the safety margin is dropped.
3. **Plan every option** with Hybrid A* (Reeds-Shepp analytic expansion plus
   an obstacle-aware grid-distance heuristic). Rank plans that end exactly
   on the goal first, then by cost: path length, a reverse-driving penalty
   and a gear-change penalty.
4. **Execute the best plan** in closed loop with the segment tracker and
   the independent collision guard. If execution fails, try the next option.
   The controller's time budget scales with plan length.
5. With no bay given, **rank free bays** by obstacle-aware driving distance
   from the start and try them nearest-first (up to six).

The result reports the chosen manoeuvre, why it beat the alternative, every
option's cost and planning effort, and the final error.

## Results (2026-10-17)

Every free bay of lots with seeds 0–4 (54 bays, 50% occupancy):

| Outcome | Count |
|---|---|
| Parked | 54 / 54 |
| Collisions | 0 |

Final position error had a median of 6.4 cm and a maximum of 11.2 cm. Heading
error had a median of 0.2° and a maximum of 2.1°. Wall-clock time
per bay, covering planning of both options and simulation, had a median of
4.6 s and a maximum of 31 s on an Apple M4 Pro.

The first run had four row-C failures. In each one the controller hit its
fixed 55 s simulated-time limit on the long route round the rows. After the
time budget was scaled with plan length, all four parked. One bay (seed 3,
B8) had also parked 30 cm off: its cheapest plan reached the goal only within
tolerance, not exactly. Options whose plans end exactly on the goal now rank
first, and that bay parks within 9 cm by reversing in.

Chosen manoeuvres across the 54 bays (the planner decided; nothing was
forced):

| Row | Chosen |
|---|---|
| A (parallel) | facing east 10, facing west 1 |
| B (perpendicular) | nose-in 16, tail-in 5 |
| C (angled) | nose-in 19, tail-in 3 |

## Limitations

- Empty bays are treated as drivable space, so routes sometimes cut across
  them. A real deployment would add lane rules or driving-direction costs.
- The lot is static: no moving actors and no perception noise. The parked
  cars are ground truth, not detections.
- Planning is single-threaded Python. Row-C targets take tens of seconds
  when both options are planned. The cost of each option is reported, so a
  deployment could plan options in parallel or stop after the first good one.
