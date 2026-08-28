"""Command-line app (``ttc-render``) that runs the rally agent off-screen and writes the result to an MP4 video with the overlay baked in."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm

from ..agent import RallyAgent
from ..config import SimulationConfig
from ..control import RobotArm
from ..visualization import PlayOverlay
from ..visualization.resolution import line_width_scale, nearest_font_scale
from ..world import load_scene
from .simulate import _add_common_arguments, config_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttc-render", description="Render the ping-pong robot to a video file.")

    # Add flags shared with ``ttc-sim`` (see ``apps/simulate.py``).
    _add_common_arguments(parser)

    # Add flags specific to ``ttc-render``.
    parser.add_argument("--output", type=Path, default=Path("output/match.mp4"), help="output video file")
    parser.add_argument("--duration", type=float, default=30.0, help="simulated seconds to render")
    parser.add_argument("--fps", type=int, default=30, help="frames per second")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera", default="render_cam", help="name of the camera to render from")
    return parser


def _open_writer(path: Path, fps: int, width: int, height: int):
    """Video writer, preferring OpenCV (which ships its own encoder)."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - opencv is a hard dependency
        raise SystemExit("opencv-python is required to write videos") from error

    # Create the output directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Open the video writer with the specified parameters
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)) # type: ignore
    if not writer.isOpened():
        raise SystemExit(f"could not open {path} for writing")

    # Define a function to write frames to the video file, converting from RGB to BGR format
    def write(frame: np.ndarray) -> None:
        writer.write(frame[:, :, ::-1])  # RGB -> BGR

    # Return the write function and a function to release the writer when done
    return write, writer.release


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.speed = 0.0  # rendering never waits for the wall clock
    config: SimulationConfig = config_from_args(args)

    # Load the scene, create the robot arm and rally agent
    scene = load_scene(config, viewport=(args.width, args.height))
    arm = RobotArm(scene.model, scene.data, config.arm, config.control, config.collision)
    agent = RallyAgent(scene, arm, config)

    # Initialize the debug plotter if requested
    recorder = None
    if args.debug_plots:
        from ..visualization.debug_plots import ServeDebugPlotter

        recorder = ServeDebugPlotter(args.debug_plots_dir, config=config)

    # Initialize the video writer and calculate the frame period based on the specified FPS
    write, close = _open_writer(args.output, args.fps, args.width, args.height)
    frame_period = 1.0 / args.fps
    next_frame = 0.0

    # Initialize the renderer and overlay for off-screen rendering, and set up a progress bar for the rendering loop
    with mujoco.Renderer(scene.model, height=args.height, width=args.width, font_scale=nearest_font_scale(args.height)) as renderer:
        overlay = PlayOverlay(scene.model, config, renderer.scene, reset_scene=False, data=scene.data, resolution_scale=line_width_scale(args.height))

        # Initialize a progress bar for the rendering loop
        pbar = tqdm(total=int(args.duration * args.fps), desc="Rendering", unit="frames")

        # Run the simulation loop until the specified duration is reached
        while scene.time < args.duration:
            # Try serving the ball
            served = agent.maybe_serve()

            # Update the simulation state
            serve_time = scene.time
            diagnostics = agent.step()

            # Record diagnostics if the debug plotter is enabled
            if recorder is not None:
                if served:
                    recorder.begin_serve(agent, serve_time)
                recorder.record(agent, diagnostics)

            # Render a frame if it's time for the next frame based on the specified FPS
            if scene.time >= next_frame:
                # Update the scene and overlay
                next_frame += frame_period
                renderer.update_scene(scene.data, camera=args.camera)
                overlay.draw(agent, diagnostics)

                # Write the rendered frame to the video file and update the progress bar
                write(renderer.render())
                pbar.update(1)

        pbar.close()

    close()
    if recorder is not None:
        recorder.close()

    # Print the simulation statistics after the loop ends
    statistics = agent.statistics
    print()
    print(f"wrote {args.output} ({scene.time:.1f} s, {args.fps} fps)")
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
