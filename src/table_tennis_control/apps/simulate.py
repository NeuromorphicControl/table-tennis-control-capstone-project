"""Command-line app (``ttc-sim`` / ``python -m table_tennis_control``) that runs the interactive MuJoCo simulation with an in-window diagnostic overlay."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from ..agent import RallyAgent
from ..config import SimulationConfig
from ..control import RobotArm
from ..visualization import PlayOverlay
from ..visualization.resolution import line_width_scale
from ..world import load_scene

_KEY_SPACE = 32
_KEY_PAUSE = ord("6")
_KEY_BALL_PATH = ord("7")
_KEY_PADDLE_PATH = ord("8")
_KEY_AUTO_SERVE = ord("9")

_VIEWER_SYNC_RATE = 60.0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by ``ttc-sim`` and ``ttc-render`` (see ``apps/render.py``)."""
    parser.add_argument("--seed", type=int, default=37, help="random seed for serves and targets (default: 37)")
    parser.add_argument("--serve-interval", type=float, default=None, help="seconds between serves")
    parser.add_argument("--no-overlay", action="store_true", help="disable the in-window overlay")
    parser.add_argument("--no-ball-path", action="store_true", help="hide the ball's real flight trace, independently of the predicted/planned trajectories")
    parser.add_argument("--no-paddle-path", action="store_true", help="hide the paddle's real position trace, colored by the robot's current stroke phase")
    parser.add_argument("--no-collision-avoidance", action="store_true", help="switch off the repulsive potential field (the arm may then hit the table)")
    parser.add_argument("--sensor-delay", type=int, default=None, help="measurement delay in control steps")
    parser.add_argument("--debug-plots", action="store_true", help="export one PNG per serve with tracking/planning diagnostics (see --debug-plots-dir)")
    parser.add_argument("--debug-plots-dir", type=Path,default=Path("output/serve_plots"), help="output directory for --debug-plots")


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser for ``ttc-sim``."""
    parser = argparse.ArgumentParser(prog="ttc-sim", description="Interactive ping-pong robot simulation with an in-window overlay.")

    # Add flags shared with ``ttc-render`` (see ``apps/render.py``).
    _add_common_arguments(parser)

    # Add flags specific to ``ttc-sim``.
    parser.add_argument("--duration", type=float, default=None, help="stop after this many simulated seconds")
    parser.add_argument("--speed", type=float, default=1.0, help="real-time factor (0 = as fast as possible)")
    parser.add_argument("--no-auto-serve", action="store_true", help="disable automatic serving entirely; serve manually with [space] instead (can also be toggled at runtime with [9])")
    return parser


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    """Create a SimulationConfig from the command line arguments."""
    config = SimulationConfig(seed=args.seed)

    # Set the visualization and collision flags based on the command line arguments
    config.visualisation.enabled = not args.no_overlay
    config.visualisation.show_ball_path = not args.no_ball_path
    config.visualisation.show_paddle_path = not args.no_paddle_path
    config.collision.enabled = not args.no_collision_avoidance

    # Set the serve interval if specified
    if args.serve_interval is not None:
        config.launcher.serve_interval = args.serve_interval

    # Set the sensor delay if specified
    if args.sensor_delay is not None:
        config.sensor.delay_steps = args.sensor_delay

    # Set the real-time flag based on the speed argument
    config.real_time = args.speed > 0.0
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    # Load the scene, create the robot arm and rally agent
    scene = load_scene(config)
    arm = RobotArm(scene.model, scene.data, config.arm, config.control, config.collision)
    agent = RallyAgent(scene, arm, config)

    # Initialize the debug plotter if requested
    recorder = None
    if args.debug_plots:
        from ..visualization.debug_plots import ServeDebugPlotter

        recorder = ServeDebugPlotter(args.debug_plots_dir, config=config)

    # Initialize the state dictionary to keep track of pause and serve status
    state = {"paused": False, "serve": False, "auto_serve": not args.no_auto_serve}

    def key_callback(keycode: int) -> None:
        if keycode == _KEY_SPACE:
            if not state["auto_serve"]:
                state["serve"] = True
        elif keycode == _KEY_PAUSE:
            state["paused"] = not state["paused"]
        elif keycode == _KEY_BALL_PATH:
            config.visualisation.show_ball_path = not config.visualisation.show_ball_path
        elif keycode == _KEY_PADDLE_PATH:
            config.visualisation.show_paddle_path = not config.visualisation.show_paddle_path
        elif keycode == _KEY_AUTO_SERVE:
            state["auto_serve"] = not state["auto_serve"]

    # Print the program description and key bindings for user interaction
    print(__doc__.strip().splitlines()[0]) # type: ignore
    print(
        "keys: [space] serve now (manual mode only)   [6] pause   "
        "[7] toggle ball's actual-path trace   [8] toggle paddle's actual-position trace   "
        "[9] toggle auto-serve (on: serves by itself, off: [space] serves)"
    )

    timestep = scene.timestep
    with mujoco.viewer.launch_passive(scene.model, scene.data, key_callback=key_callback) as viewer:
        overlay = PlayOverlay(scene.model, config, viewer.user_scn, data=scene.data)

        wall_start = time.perf_counter()
        next_sync = -np.inf
        sync_period = 1.0 / _VIEWER_SYNC_RATE

        # Run the simulation loop until the viewer is closed or the specified duration is reached
        while viewer.is_running():
            if args.duration is not None and scene.time >= args.duration:
                break

            # Handle pausing
            loop_start = time.perf_counter()
            if state["paused"]:
                viewer.sync()
                time.sleep(0.01)
                continue

            # Handle serving based on the current state (auto-serve or manual serve)
            if state["auto_serve"]:
                served = agent.maybe_serve()
            elif state["serve"]:
                agent.serve()
                state["serve"] = False
                served = True
            else:
                served = False

            # Update the simulation state
            serve_time = scene.time
            diagnostics = agent.step()

            # Record diagnostics if the debug plotter is enabled
            if recorder is not None:
                if served:
                    recorder.begin_serve(agent, serve_time)
                recorder.record(agent, diagnostics)

            # Update the overlay if it's due for a redraw based on the diagnostics time
            if overlay.due(diagnostics.time):
                viewport = viewer.viewport
                if viewport is not None and viewport.height > 0:
                    overlay.resolution_scale = line_width_scale(viewport.height)
                overlay.draw(agent, diagnostics)

            # Sync the viewer at a fixed rate to avoid excessive updates
            if diagnostics.time >= next_sync:
                next_sync = diagnostics.time + sync_period
                viewer.sync()

            # If real-time mode is enabled, sleep for the remaining time to maintain the desired speed
            if config.real_time:
                remaining = timestep / max(args.speed, 1e-6) - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

        # Calculate the total wall clock time taken for the simulation
        wall = time.perf_counter() - wall_start

    if recorder is not None:
        recorder.close()

    # Print the simulation statistics after the loop ends
    statistics = agent.statistics
    print()
    print(f"simulated {scene.time:.1f} s in {wall:.1f} s wall clock")
    print(f"serves            {statistics.serves}")
    print(f"strokes           {statistics.strikes}")
    print(f"landings recorded {statistics.landings}")
    print(f"on opponent half  {statistics.on_target_half}")
    if statistics.landing_errors:
        errors = np.asarray(statistics.landing_errors)
        print(f"landing error     mean {errors.mean():.3f} m, median {np.median(errors):.3f} m")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
