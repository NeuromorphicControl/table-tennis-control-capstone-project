from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco

from table_tennis_control.agent import Phase, RallyAgent
from table_tennis_control.config import SimulationConfig
from table_tennis_control.control import RobotArm
from table_tennis_control.visualization import PlayOverlay
from table_tennis_control.visualization.resolution import line_width_scale
from table_tennis_control.world import Scene, load_scene

OUTPUT_DIR = Path("output/report_figures/phase_gallery")

EARLY_APPROACH_FRACTION = 0.55
LATE_APPROACH_FRACTION = 1.10

MIN_TRACK_TICKS = 5

_CAMERA_LOOKAT = (0.0, -1.0, 0.9)
_CAMERA_DISTANCE = 4.0
_CAMERA_AZIMUTH = 205.0
_CAMERA_ELEVATION = -10.0


class _UnlabelledOverlay(PlayOverlay):

    def _draw_state_indicator(self, phase) -> None:
        pass


def _make_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = _CAMERA_LOOKAT
    camera.distance = _CAMERA_DISTANCE
    camera.azimuth = _CAMERA_AZIMUTH
    camera.elevation = _CAMERA_ELEVATION
    return camera


def _build_session(seed: int, width: int, height: int, visual: bool = False) -> tuple[Scene, RallyAgent, SimulationConfig]:
    config = SimulationConfig(seed=seed)
    config.visualisation.enabled = visual
    config.visualisation.show_predicted_trajectory = False
    config.visualisation.show_plan = False
    config.visualisation.show_ball_path = True
    config.visualisation.show_paddle_path = True
    scene = load_scene(config, viewport=(width, height))
    arm = RobotArm(scene.model, scene.data, config.arm, config.control, config.collision)
    agent = RallyAgent(scene, arm, config)
    return scene, agent, config


def _ball_paddle_contact(scene: Scene, paddle_geom_id: int) -> bool:
    data = scene.data
    ball_geom = scene.ball.geom_id
    for index in range(data.ncon):
        contact = data.contact[index]
        geoms = (int(contact.geom1), int(contact.geom2))
        if ball_geom not in geoms:
            continue
        other = geoms[0] if geoms[1] == ball_geom else geoms[1]
        if other == paddle_geom_id:
            return True
    return False


@dataclass
class _Tick:
    index: int
    phase: Phase
    in_contact: bool
    serve_index: int


def _record_timeline(seed: int, duration: float, width: int, height: int) -> list[_Tick]:
    scene, agent, _ = _build_session(seed, width, height)
    paddle_geom_id = int(scene.model.geom("paddle").id)

    ticks: list[_Tick] = []
    serve_index = 0
    index = 0
    while scene.time < duration:
        if agent.maybe_serve():
            serve_index += 1
        agent.step()
        in_contact = _ball_paddle_contact(scene, paddle_geom_id)
        ticks.append(_Tick(index, agent.phase, in_contact, serve_index))
        index += 1
    return ticks


def _phase_run(rally: list[_Tick], phase: Phase, after: int) -> tuple[int, int] | None:
    start = None
    for i in range(after, len(rally)):
        if rally[i].phase is phase:
            if start is None:
                start = i
        elif start is not None:
            return start, i - 1
    return (start, len(rally) - 1) if start is not None else None


def _find_targets(ticks: list[_Tick]) -> dict[str, int] | None:
    max_serve = ticks[-1].serve_index
    for serve in range(2, max_serve + 1):
        rally = [t for t in ticks if t.serve_index == serve]
        if not rally:
            continue

        track_run = _phase_run(rally, Phase.TRACK, after=0)
        if track_run is None:
            continue
        track_start, track_end = track_run
        track_len = track_end - track_start
        if track_len < MIN_TRACK_TICKS:
            continue

        if track_end + 1 >= len(rally) or rally[track_end + 1].phase is not Phase.SWING:
            continue
        swing_run = _phase_run(rally, Phase.SWING, after=track_end + 1)
        assert swing_run is not None
        swing_start, swing_end = swing_run

        contact_offsets = [i for i in range(swing_start, swing_end + 1) if rally[i].in_contact]
        if not contact_offsets:
            continue

        previous_rally = [t for t in ticks if t.serve_index == serve - 1]
        idle_indices = [i for i, t in enumerate(previous_rally) if t.phase is Phase.IDLE]
        if not idle_indices:
            continue

        early = rally[track_start + round(EARLY_APPROACH_FRACTION * track_len)]
        late = rally[track_start + round(LATE_APPROACH_FRACTION * track_len)]
        impact = rally[contact_offsets[0]]
        follow_through = rally[swing_end]
        ready = previous_rally[idle_indices[-1]]

        return {
            "1_ready_pose": ready.index,
            "2_early_approach": early.index,
            "3_late_approach": late.index,
            "4_impact": impact.index,
            "5_follow_through": follow_through.index,
        }
    return None


def _render_targets(targets: dict[str, int], seed: int, width: int, height: int, output_dir: Path) -> None:
    scene, agent, config = _build_session(seed, width, height, visual=True)
    camera = _make_camera()
    output_dir.mkdir(parents=True, exist_ok=True)

    name_by_index = {tick_index: name for name, tick_index in targets.items()}
    last_index = max(targets.values())

    with mujoco.Renderer(scene.model, height=height, width=width) as renderer:
        overlay = _UnlabelledOverlay(scene.model, config, renderer.scene, reset_scene=False, data=scene.data, resolution_scale=line_width_scale(height))

        index = 0
        while index <= last_index:
            agent.maybe_serve()
            diagnostics = agent.step()
            name = name_by_index.get(index)
            if name is not None:
                renderer.update_scene(scene.data, camera=camera)

                if name == "1_ready_pose":
                    config.visualisation.show_ball_path = False
                    config.visualisation.show_paddle_path = False
                overlay.draw(agent, diagnostics)

                if name == "1_ready_pose":
                    config.visualisation.show_ball_path = True
                    config.visualisation.show_paddle_path = True

                frame = renderer.render()
                path = output_dir / f"{name}.png"
                cv2.imwrite(str(path), frame[:, :, ::-1])  # RGB -> BGR
                print(f"  wrote {path}")
            index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=37, help="random seed (default: 37)")
    parser.add_argument("--duration", type=float, default=20.0, help="how many simulated seconds to scan for a clean rally (default: 20)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    print(f"pass 1/2: scanning a {args.duration:g} s session (seed {args.seed}) for a clean rally...")
    ticks = _record_timeline(args.seed, args.duration, args.width, args.height)
    targets = _find_targets(ticks)
    if targets is None:
        raise SystemExit(
            "no rally with a distinct approach and a genuine strike was found; "
            "try a longer --duration or a different --seed"
        )

    print(f"pass 2/2: re-rendering the same session, saving 5 frames to {args.output_dir}...")
    _render_targets(targets, args.seed, args.width, args.height, args.output_dir)


if __name__ == "__main__":
    main()
