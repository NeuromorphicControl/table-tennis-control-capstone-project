# Benchmark script

This folder holds the benchmark harness used to evaluate this project's
behaviour end-to-end — the same measurement every change to the control
pipeline is judged against before being kept.

## `benchmark.py`

Runs a fixed set of seeds, each a full headless rally session (no viewer, no
overlay) for a fixed simulated duration, and aggregates the results across
all of them into one report:

* **strike rate** — fraction of serves the robot actually returned
* **on-target-half rate** — fraction of returned balls that landed on the
  opponent's half
* **net-clip rate** — fraction of struck balls that never registered a
  landing at all (typically a net or table clip on the way)
* **mean / median landing error** — distance between where the ball landed
  and the target, over every successful return

```bash
python scripts/benchmark.py --seeds 15 --duration 20
python scripts/benchmark.py --seeds 6 --duration 30 --start-seed 1
```

`--seeds` controls how many independent sessions to run (default 15),
`--duration` how many simulated seconds each one lasts (default 20), and
`--start-seed` where the seed range begins (default 1). The same
`(start-seed, seeds, duration)` combination always reproduces the same set
of serves, so two runs of the benchmark are a fair before/after comparison.

Assumes the package is installed (`pip install -e .` from the project root)
and is run from the project root.
