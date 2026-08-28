from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from table_tennis_control.agent import Phase, RallyAgent
from table_tennis_control.config import SimulationConfig
from table_tennis_control.control import RobotArm
from table_tennis_control.visualization.colors import Color, PHASE_COLOR, phase_runs
from table_tennis_control.visualization.debug_plots import ServeDebugPlotter, _ServeRecord
from table_tennis_control.world import load_scene

MIN_TRACK_TICKS = 5


class _ServeCapture(ServeDebugPlotter):

    def __init__(self, config: SimulationConfig, scratch_dir: Path, min_strikes: int):
        super().__init__(scratch_dir, config=config)
        self.min_strikes = min_strikes
        self.records: list[_ServeRecord] = []

    def _render(self, record: _ServeRecord) -> None:
        if record.strikes_this_serve >= self.min_strikes:
            self.records.append(record)


def _capture_serves(seed: int, max_duration: float, min_strikes: int) -> list[_ServeRecord]:
    config = SimulationConfig(seed=seed)
    config.visualisation.enabled = False

    scene = load_scene(config)
    arm = RobotArm(scene.model, scene.data, config.arm, config.control, config.collision)
    agent = RallyAgent(scene, arm, config)

    scratch_dir = Path(tempfile.mkdtemp(prefix="ttc_serve_capture_"))
    capture = _ServeCapture(config, scratch_dir, min_strikes=min_strikes)
    try:
        while scene.time < max_duration:
            served = agent.maybe_serve()
            serve_time = scene.time
            diagnostics = agent.step()
            if served:
                capture.begin_serve(agent, serve_time)
            capture.record(agent, diagnostics)
        capture.close()
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return capture.records


def _has_genuine_commit(phases: list[Phase], min_track_ticks: int) -> bool:
    runs = list(phase_runs(phases))
    for index, (phase, start, end) in enumerate(runs):
        if phase is Phase.TRACK:
            long_enough = (end - start) >= min_track_ticks
            followed_by_swing = index + 1 < len(runs) and runs[index + 1][0] is Phase.SWING
            return long_enough and followed_by_swing
    return False


def _select_serve(records: list[_ServeRecord], serve_index: int | None, min_track_ticks: int = MIN_TRACK_TICKS) -> _ServeRecord:
    if not records:
        raise SystemExit("no serve with a strike was captured -- try a different --seed or a longer --max-duration")

    if serve_index is not None:
        for record in records:
            if record.index == serve_index:
                return record
        found = [record.index for record in records]
        raise SystemExit(f"serve {serve_index} was not captured with a strike; captured serves were {found}")

    for record in records:
        if record.index >= 2 and _has_genuine_commit(record.phase, min_track_ticks):
            return record

    with_landing = [record for record in records if record.landing_errors]
    if with_landing:
        return min(with_landing, key=lambda record: record.landing_errors[-1])
    return records[0]


def _mark_impact(ax, impact_time: float | None) -> None:
    if impact_time is None:
        return
    ax.axvline(impact_time, color=Color.STRIKE[:3], linestyle="--", linewidth=1.3, zorder=4)
    ax.text(impact_time, 1.02, "impact", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=Color.STRIKE[:3])


def _plot_error_panel(ax, time: np.ndarray, values: list[float], phases, impact_time: float | None, ylabel: str) -> None:
    values = np.asarray(values)

    for phase, start, end in phase_runs(phases):
        ax.axvspan(time[start], time[end], color=PHASE_COLOR[phase][:3], alpha=0.20, lw=0)

    ax.plot(time, values, color="0.15", linewidth=1.6, zorder=3)

    if impact_time is not None:
        index = int(np.argmin(np.abs(time - impact_time)))
        ax.scatter([time[index]], [values[index]], marker="o", s=28, facecolor=Color.STRIKE[:3], edgecolor="white", linewidths=0.8, zorder=5)

    _mark_impact(ax, impact_time)

    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _trim_trailing_idle(time: np.ndarray, phases: list, position: np.ndarray, orientation: np.ndarray, trailing_margin: float):
    active = [index for index, phase in enumerate(phases) if phase != Phase.IDLE]
    last_active = active[-1] if active else len(phases) - 1
    cutoff = int(np.searchsorted(time, time[last_active] + trailing_margin, side="right"))
    return time[:cutoff], phases[:cutoff], position[:cutoff], orientation[:cutoff]


def render_report_figure(record: _ServeRecord, output_base: Path, formats: list[str], dpi: int, title: bool, trailing_margin: float = 0.3) -> list[Path]:
    time = np.asarray(record.time) - record.start_time
    position = np.asarray(record.position_error)
    orientation = np.asarray(record.orientation_error)
    time, phases, position, orientation = _trim_trailing_idle(time, record.phase, position, orientation, trailing_margin)

    strike_times = [t - record.start_time for t in record.strike_times if t - record.start_time <= time[-1]]
    impact_time = strike_times[0] if strike_times else None

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
        }
    ):
        fig, (ax_position, ax_orientation) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True, layout="constrained")

        _plot_error_panel(ax_position, time, position, phases, impact_time, "Position error [m]")
        _plot_error_panel(ax_orientation, time, orientation, phases, impact_time, "Orientation error [rad]")
        ax_orientation.set_xlabel("Time since serve start [s]")
        ax_position.set_xlim(time[0], time[-1])

        if title:
            fig.suptitle("Paddle tracking error during an exemplary serve", fontsize=11)

        phase_handles = ServeDebugPlotter._phase_legend_handles(2.0)
        impact_handle = None
        if impact_time is not None:
            from matplotlib.lines import Line2D

            impact_handle = Line2D([0], [0], color=Color.STRIKE[:3], linestyle="--", linewidth=1.3, label="Impact")
        handles = phase_handles + ([impact_handle] if impact_handle is not None else [])
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)

        output_base.parent.mkdir(parents=True, exist_ok=True)
        saved = []
        for fmt in formats:
            path = output_base.with_suffix(f".{fmt}")
            fig.savefig(path, dpi=dpi)
            saved.append(path)
        plt.close(fig)

    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=37, help="random seed for serves and targets (default: 37)")
    parser.add_argument("--max-duration", type=float, default=15.0, help="simulated seconds to search for a struck serve (default: 15)")
    parser.add_argument("--min-strikes", type=int, default=1, help="only consider serves with at least this many strikes (default: 1)")
    parser.add_argument("--serve-index", type=int, default=None, help="plot this specific serve number instead of auto-selecting the best return")
    parser.add_argument("--output", type=Path, default=Path("output/report_figures/serve_error"), help="output path without extension (default: output/report_figures/serve_error)")
    parser.add_argument("--formats", nargs="+", default=["png"], help="file formats to save, e.g. pdf png svg (default: png)")
    parser.add_argument("--dpi", type=int, default=300, help="raster resolution for non-vector formats (default: 300)")
    parser.add_argument("--no-title", action="store_true", help="omit the figure title (useful when the caption will carry it in the report)")
    parser.add_argument("--trailing-margin", type=float, default=0.3, help="seconds of idle context to keep after recovery finishes, before trimming the flat tail (default: 0.3)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    records = _capture_serves(args.seed, args.max_duration, args.min_strikes)
    record = _select_serve(records, args.serve_index)

    outcome = f"strikes={record.strikes_this_serve}"
    if record.landing_errors:
        outcome += f" landing_error={record.landing_errors[-1]:.3f} m"
    print(f"plotting serve {record.index} (launch t={record.start_time:.2f} s, {outcome})")

    saved = render_report_figure(record, args.output, args.formats, args.dpi, title=not args.no_title, trailing_margin=args.trailing_margin)
    for path in saved:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
