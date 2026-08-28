# Scripts

This folder holds the benchmark harness used to evaluate this project's
behaviour end-to-end, plus a `report_figures/` subfolder of scripts that
generate report-ready figures for the write-up.

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

## `report_figures/`

Scripts that turn a headless run into a report-ready figure instead of a
number: `serve_error_figure.py` (paddle position/orientation error over one
exemplary serve), `render_phase_gallery.py` (five snapshot frames of that
same serve's stroke) and `sensor_delay_sweep.py` (target error vs. sensor
delay). See [`report_figures/README.md`](report_figures/README.md) for
details.
