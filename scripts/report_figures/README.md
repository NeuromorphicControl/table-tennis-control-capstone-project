# Report figures

Scripts that run a headless session (or several) and render the result
straight to a report-ready figure, rather than printing a number like
`../benchmark.py` does.

## `serve_error_figure.py`

Runs one headless session, captures the same per-tick tracking diagnostics as
`--debug-plots` (see `table_tennis_control.visualization.debug_plots`), and
renders just the two error curves — paddle position error and orientation
error over time — for one exemplary serve, as a clean two-panel figure sized
for a report rather than the full six-panel debug grid. The background is
shaded by stroke phase (idle/track/swing/recover) and the moment of
ball-paddle impact is marked on both panels, so the figure reads consistently
with the debug plots.

```bash
python scripts/report_figures/serve_error_figure.py
python scripts/report_figures/serve_error_figure.py --seed 3 --serve-index 2
python scripts/report_figures/serve_error_figure.py --output figures/serve_error --formats pdf png
```

By default it picks the earliest genuinely struck serve (skipping the first
rally, and requiring a real TRACK -> SWING commit rather than a
last-instant correction — see `_has_genuine_commit`) out of whatever it
finds within `--max-duration` simulated seconds (default 15) for the given
`--seed` (default 37), falling back to the struck serve with the smallest
landing error if none qualifies. That selection rule mirrors
`render_phase_gallery.py`'s, so with the same `--seed` the two scripts
reference the same serve — the error curves here and the approach/impact/
follow-through frames there are one consistent example rather than two
different rallies. Pass `--serve-index` to plot a specific serve instead.
Output is written next to `--output` (default
`output/report_figures/serve_error`) once per entry in `--formats` (default
`pdf png`) — `pdf` for including in a LaTeX report, `png` for a quick look.

## `render_phase_gallery.py`

Renders five PNG frames from a single genuine rally, for a side-by-side
figure of the stroke: `ready_pose`, `early_approach` and `late_approach`
(both TRACK), `impact` and `follow_through` (both SWING). Runs the session
twice with the same seed (the sim is fully deterministic given one): a
headless pass to find which physics tick each snapshot falls on, and a
second, identical pass with a renderer attached that saves a frame whenever
the tick counter matches one of the five found in the first pass.

```bash
python scripts/report_figures/render_phase_gallery.py
python scripts/report_figures/render_phase_gallery.py --seed 3 --duration 30
```

It scans up to `--duration` simulated seconds (default 20) for the given
`--seed` (default 37) and picks the earliest rally, from the second serve
onward, with a genuine TRACK -> SWING commit and an actual strike — the
same rule `serve_error_figure.py` uses by default, so the two scripts
reference the same serve when given the same `--seed`. Frames are written
to `--output-dir` (default `output/report_figures/phase_gallery`).

## `sensor_delay_sweep.py`

Sweeps `SensorConfig.delay_steps` (see `--sensor-delay` in `ttc-sim` /
`ttc-render`) from zero upward in fixed-size steps. Each delay value is
measured the same way `benchmark.py` measures its metrics — several headless
seeds pooled together — and plotted as mean target (landing) error vs. delay,
with a shaded ±1 std band and a secondary top axis in milliseconds. The sweep
stops and marks the cutoff once the strike rate collapses below
`--strike-rate-floor`, since target error stops being a meaningful number once
the robot mostly isn't hitting the ball any more; if that never happens it
just runs up to `--max-delay` and says so.

```bash
python scripts/report_figures/sensor_delay_sweep.py
python scripts/report_figures/sensor_delay_sweep.py --seeds 8 --duration 20 --delay-step 5
python scripts/report_figures/sensor_delay_sweep.py --max-delay 200 --output figures/sensor_delay
```

This is the slowest script here: it runs `--seeds` × one session per delay
value tried, so a full sweep is many times the cost of one `benchmark.py`
run. The default `--max-delay` (100 ticks = 100 ms) is a guess — the control
loop tolerated every delay tried in testing up to 60 ticks without its strike
rate dropping at all, so finding an actual breakdown may need a larger
`--max-delay` (and patience). Start with a short, coarse run (few seeds,
large `--delay-step`) to see roughly where things start to degrade before
committing to a dense final sweep.

---

All three scripts assume the package is installed (`pip install -e .` from
the project root) and are run from the project root, not from inside this
folder.
