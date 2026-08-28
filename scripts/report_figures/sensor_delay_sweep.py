from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from table_tennis_control.agent import RallyAgent, RallyStatistics
from table_tennis_control.config import SimulationConfig
from table_tennis_control.control import RobotArm
from table_tennis_control.visualization.colors import Color
from table_tennis_control.world import load_scene


@dataclass
class SweepPoint:
    delay_steps: int
    serves: int = 0
    strikes: int = 0
    landings: int = 0
    on_half: int = 0
    errors: list[float] = field(default_factory=list)

    @property
    def strike_rate(self) -> float:
        return self.strikes / self.serves if self.serves else float("nan")

    @property
    def mean_error(self) -> float:
        return float(np.mean(self.errors)) if self.errors else float("nan")

    @property
    def std_error(self) -> float:
        return float(np.std(self.errors)) if self.errors else float("nan")


def _run_one(seed: int, duration: float, delay_steps: int) -> RallyStatistics:
    config = SimulationConfig(seed=seed)
    config.visualisation.enabled = False
    config.sensor.delay_steps = delay_steps

    scene = load_scene(config)
    arm = RobotArm(scene.model, scene.data, config.arm, config.control, config.collision)
    agent = RallyAgent(scene, arm, config)

    while scene.time < duration:
        agent.maybe_serve()
        agent.step()
    return agent.statistics


def _measure_delay(delay_steps: int, seeds: list[int], duration: float) -> SweepPoint:
    point = SweepPoint(delay_steps=delay_steps)
    for seed in tqdm(seeds, desc=f"delay {delay_steps:>4d} ticks", unit="seed", leave=False):
        stats = _run_one(seed, duration, delay_steps)
        point.serves += stats.serves
        point.strikes += stats.strikes
        point.landings += stats.landings
        point.on_half += stats.on_target_half
        point.errors.extend(stats.landing_errors)
    return point


def run_sweep(seeds: list[int], duration: float, delay_start: int, delay_step: int, max_delay: int, strike_rate_floor: float) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    delay = delay_start
    while delay <= max_delay:
        point = _measure_delay(delay, seeds, duration)
        points.append(point)

        err_text = f"{point.mean_error:.3f} m" if point.errors else "n/a"
        print(f"  delay={delay:4d} ticks   strike_rate={point.strike_rate:.2f}   mean_error={err_text}   landings={point.landings}/{point.strikes} strikes")

        if point.strike_rate < strike_rate_floor:
            print(f"strike rate fell below {strike_rate_floor:.0%} at delay={delay} ticks -- stopping the sweep here")
            break
        delay += delay_step
    else:
        print(f"reached --max-delay ({max_delay} ticks) without the strike rate collapsing -- stopping there")

    return points


def render_report_figure(points: list[SweepPoint], timestep: float, output_base: Path, formats: list[str], dpi: int, title: bool, strike_rate_floor: float, benchmark_error: float | None = None) -> list[Path]:
    delays = np.array([point.delay_steps for point in points], dtype=float)
    mean_error = np.array([point.mean_error for point in points])
    std_error = np.array([point.std_error for point in points])
    valid = ~np.isnan(mean_error)

    breakdown = next((point for point in points if point.strike_rate < strike_rate_floor), None)

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
        fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")

        lower = np.clip((mean_error - std_error)[valid], 0.0, None)
        upper = (mean_error + std_error)[valid]
        ax.fill_between(delays[valid], lower, upper, color="0.15", alpha=0.15, linewidth=0, zorder=2, label="±1 std (across seeds)")
        ax.plot(delays[valid], mean_error[valid], color="0.15", linewidth=1.8, marker="o", markersize=4, zorder=3, label="mean target error")

        if breakdown is not None:
            ax.axvline(breakdown.delay_steps, color=Color.STRIKE[:3], linestyle="--", linewidth=1.3, zorder=4)
            ax.text(breakdown.delay_steps, 1.02, f"strike rate < {strike_rate_floor:.0%}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=Color.STRIKE[:3])

        if benchmark_error is not None:
            ax.axhline(benchmark_error, color="0.6", linestyle=":", linewidth=1.3, zorder=1, label="standard benchmark mean error")

        ax.set_xlabel("Sensor delay [ticks]")
        ax.set_ylabel("Mean target error [m]")
        ax.set_xlim(delays[0], delays[-1])
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", fontsize="small", frameon=False)

        secondary = ax.secondary_xaxis("top", functions=(lambda ticks: ticks * timestep * 1000.0, lambda ms: ms / (timestep * 1000.0)))
        secondary.set_xlabel("Sensor delay [ms]")

        if title:
            fig.suptitle("Target error vs. sensor delay", fontsize=11)

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
    parser.add_argument("--seeds", type=int, default=5, help="seeds pooled per delay value (default: 5)")
    parser.add_argument("--start-seed", type=int, default=1, help="first seed value (default: 1)")
    parser.add_argument("--duration", type=float, default=15.0, help="simulated seconds per seed (default: 15)")
    parser.add_argument("--delay-start", type=int, default=0, help="first sensor delay to test, in ticks (default: 0)")
    parser.add_argument("--delay-step", type=int, default=50, help="ticks between consecutive delay values (default: 50)")
    parser.add_argument("--max-delay", type=int, default=1000, help="largest delay to test if the strike rate never collapses, in ticks (default: 1000)")
    parser.add_argument("--strike-rate-floor", type=float, default=0.2, help="stop the sweep once the strike rate drops below this fraction (default: 0.2)")
    parser.add_argument("--output", type=Path, default=Path("output/report_figures/sensor_delay_sweep"), help="output path without extension (default: output/report_figures/sensor_delay_sweep)")
    parser.add_argument("--formats", nargs="+", default=["png"], help="file formats to save, e.g. pdf png svg (default: png)")
    parser.add_argument("--dpi", type=int, default=300, help="raster resolution for non-vector formats (default: 300)")
    parser.add_argument("--no-title", action="store_true", help="omit the figure title (useful when the caption will carry it in the report)")
    parser.add_argument("--benchmark-seeds", type=int, default=6, help="seeds for the standard-benchmark reference line, matching scripts/benchmark.py (default: 15)")
    parser.add_argument("--benchmark-duration", type=float, default=30.0, help="simulated seconds per seed for the standard-benchmark reference line (default: 20)")
    parser.add_argument("--no-benchmark-line", action="store_true", help="omit the standard-benchmark mean error reference line")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))

    print(f"sweeping sensor delay: {args.delay_start}..{args.max_delay} ticks (step {args.delay_step}), {len(seeds)} seeds x {args.duration:g} s each")
    points = run_sweep(seeds, args.duration, args.delay_start, args.delay_step, args.max_delay, args.strike_rate_floor)

    benchmark_error = None
    if not args.no_benchmark_line:
        benchmark_seeds = list(range(args.start_seed, args.start_seed + args.benchmark_seeds))
        print(f"running standard benchmark for the reference line: {len(benchmark_seeds)} seeds x {args.benchmark_duration:g} s each")
        benchmark_error = _measure_delay(0, benchmark_seeds, args.benchmark_duration).mean_error
        print(f"  standard benchmark mean_error={benchmark_error:.3f} m")

    config = SimulationConfig(seed=seeds[0])
    config.visualisation.enabled = False
    timestep = load_scene(config).timestep

    saved = render_report_figure(
        points,
        timestep,
        args.output,
        args.formats,
        args.dpi,
        title=not args.no_title,
        strike_rate_floor=args.strike_rate_floor,
        benchmark_error=benchmark_error,
    )
    for path in saved:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
