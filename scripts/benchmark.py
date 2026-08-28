"""Headless N-seed x T-second benchmark."""

from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from table_tennis_control.agent import RallyAgent, RallyStatistics
from table_tennis_control.config import SimulationConfig
from table_tennis_control.control import RobotArm
from table_tennis_control.world import load_scene


BAR_FORMAT = "{desc}{percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f} {unit} [{elapsed}<{remaining}]"


def run_one(seed: int, duration: float, desc: str | None = None) -> RallyStatistics:
    """Run one full headless session and return its accumulated statistics.

    Args:
        seed: Random seed to use for this session.
        duration: Simulated seconds to run this session.
        desc: Progress bar label; defaults to a plain seed label if omitted.

    Returns:
        RallyStatistics object containing the accumulated statistics for this session.
    """
    config = SimulationConfig(seed=seed)
    config.visualisation.enabled = False

    # Load the scene, create the robot arm and agent
    scene = load_scene(config)
    arm = RobotArm(scene.model, scene.data, config.arm, config.control, config.collision)
    agent = RallyAgent(scene, arm, config)

    # Run the simulation for the specified duration, stepping the agent and serving as needed
    with tqdm(total=duration, desc=desc or f"seed {seed}", unit="s", bar_format=BAR_FORMAT, leave=False) as run_bar:
        while scene.time < duration:
            agent.maybe_serve()
            agent.step()
            run_bar.update(min(scene.time, duration) - run_bar.n)
    return agent.statistics


def run_benchmark(seeds: list[int], duration: float) -> dict[str, float]:
    """Run every seed and aggregate into the standard metric set."""
    total_serves = total_strikes = total_landings = total_on_half = 0
    all_errors: list[float] = []

    # Pad descriptions to a common width so the outer and inner bars line up
    seed_width = max(len(str(seed)) for seed in seeds)
    label_width = max(len("benchmark"), len("seed ") + seed_width) + 1

    # Run each seed and accumulate statistics
    outer_desc = "benchmark".ljust(label_width)
    for seed in tqdm(seeds, desc=outer_desc, unit="seed", bar_format=BAR_FORMAT, leave=False):
        inner_desc = f"seed {seed:>{seed_width}}".ljust(label_width)
        stats = run_one(seed, duration, desc=inner_desc)
        total_serves += stats.serves
        total_strikes += stats.strikes
        total_landings += stats.landings
        total_on_half += stats.on_target_half
        all_errors.extend(stats.landing_errors)

    errors = np.asarray(all_errors) if all_errors else np.array([0.0])
    net_clips = total_strikes - total_landings

    return {
        "serves": total_serves,
        "strikes": total_strikes,
        "landings": total_landings,
        "on_half": total_on_half,
        "strike_rate": total_strikes / max(total_serves, 1),
        "on_half_rate": total_on_half / max(total_strikes, 1),
        "net_clip_rate": net_clips / max(total_strikes, 1),
        "mean_err": float(errors.mean()),
        "median_err": float(np.median(errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=15, help="number of seeds to run")
    parser.add_argument("--start-seed", type=int, default=1, help="first seed value")
    parser.add_argument("--duration", type=float, default=20.0, help="simulated seconds per seed")
    args = parser.parse_args()

    # Run the benchmark and print the results
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    result = run_benchmark(seeds, args.duration)

    print(f"{len(seeds)} seeds x {args.duration:g} s ({seeds[0]}-{seeds[-1]}):")
    print(f"  serves={result['serves']} strikes={result['strikes']} landings={result['landings']} on_half={result['on_half']}")
    print(f"  strike_rate       = {result['strike_rate']:.3f}")
    print(f"  on_half_rate      = {result['on_half_rate']:.3f}")
    print(f"  net_clip_rate     = {result['net_clip_rate']:.3f}")
    print(f"  mean_landing_err  = {result['mean_err']:.3f} m")
    print(f"  median_landing_err= {result['median_err']:.3f} m")


if __name__ == "__main__":
    main()
