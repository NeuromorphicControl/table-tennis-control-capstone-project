"""Renders what the agent believes and intends, straight into the 3-D scene."""

from __future__ import annotations

import numpy as np

from ..agent import AgentDiagnostics, Phase, RallyAgent
from ..config import SimulationConfig
from ..estimation.predictor import Trajectory
from ..physics import flight_position
from .colors import PHASE_COLOR, Color, phase_runs
from .scene_overlay import SceneOverlay

__all__ = ["PlayOverlay"]


_STATE_LABEL_HEIGHT = 0.75


class PlayOverlay:
    """Draws the prediction, the plan and the robot's state indicator.

    The overlay is throttled to its own refresh rate and never touches the
    physics, so switching it off changes nothing but the picture.
    """

    def __init__(self, model, config: SimulationConfig, scene, reset_scene: bool = True, data=None, resolution_scale: float = 1.0):
        """Create a new overlay for the given model and scene."""
        self.model = model
        self.data = data
        self.config = config
        self.visual = config.visualisation
        self.overlay = SceneOverlay(scene, reset_on_begin=reset_scene)
        self.resolution_scale = float(resolution_scale)
        self._robot_body_id = int(model.body("base").id)

        self._next_update = -np.inf

    def due(self, time: float) -> bool:
        """Return whether the overlay is due for a redraw at the given time.
        
        Args:
            time: The current simulation time.
            
        Returns:
            True if the overlay should be redrawn, False otherwise.
        """
        return time >= self._next_update

    # ----------------------------------------------------------------- drawing
    def draw(self, agent: RallyAgent, diagnostics: AgentDiagnostics) -> None:
        """Redraw the whole overlay.
        
        Args:
            agent: The agent whose state is being visualised.
            diagnostics: The agent's diagnostics, including its prediction and plan.
        """
        if not self.visual.enabled:
            return
        self._next_update = diagnostics.time + 1.0 / max(self.visual.update_rate, 1e-3)

        self.overlay.begin()

        if self.visual.show_predicted_trajectory and diagnostics.prediction is not None:
            self._draw_prediction(diagnostics.prediction)
        if self.visual.show_plan:
            self._draw_plan(diagnostics)
        if self.visual.show_actual_path:
            self._draw_actual_path(agent.actual_path, agent.actual_path_phases)
        self._draw_target(agent)
        self._draw_state_indicator(diagnostics.phase)

    def _line_width(self, width: float) -> float:
        return width * self.resolution_scale

    def _draw_prediction(self, prediction: Trajectory) -> None:
        self.overlay.polyline(prediction.positions, Color.PREDICTION, width=self._line_width(2.5), stride=2)
        for event in prediction.events:
            color = Color.BOUNCE if event.surface == "table" else Color.GRID
            self.overlay.sphere(event.position, 0.025, color)

    def _draw_plan(self, diagnostics: AgentDiagnostics) -> None:
        plan = diagnostics.plan
        if plan is None:
            return

        # Where the paddle will meet the ball, and how it will be held.
        self.overlay.sphere(plan.position, 0.03, Color.STRIKE)
        self.overlay.arrow(plan.position, plan.normal * 0.16, Color.STRIKE, width=0.007)
        if plan.paddle_speed > 1e-3:
            self.overlay.arrow(plan.position, plan.paddle_velocity * 0.12, Color.PLANNED, width=0.006)

        # The return the plan is aiming for, drawn through its bounce.
        acceleration = plan.acceleration
        if plan.bounce_point is not None:
            # The ball's planned flight before the bounce
            first = np.arange(0.0, max(plan.time_to_bounce, 1e-3), 0.01)
            self.overlay.polyline(
                flight_position(first, plan.position, plan.outgoing_velocity, acceleration),
                Color.PLANNED,
                width=self._line_width(2.5),
            )

            # The ball's predicted bounce point
            self.overlay.sphere(plan.bounce_point, 0.028, Color.BOUNCE)
            rebound = plan.rebound_velocity(self.config.ball, acceleration)

            # The ball's planned flight after the bounce
            second = np.arange(0.0, max(plan.time_after_bounce, 1e-3), 0.01)
            self.overlay.polyline(
                flight_position(second, plan.bounce_point, rebound, acceleration),
                Color.PLANNED,
                width=self._line_width(2.5),
            )

        else:
            # The ball's planned flight, if it won't bounce
            times = np.arange(0.0, max(plan.time_after_bounce, 1e-3), 0.01)
            self.overlay.polyline(
                flight_position(times, plan.position, plan.outgoing_velocity, acceleration),
                Color.PLANNED,
                width=self._line_width(2.5),
            )

    def _draw_actual_path(self, path: np.ndarray, phases: list[Phase]) -> None:
        """The ball's real (simulated) flight, colored by whichever phase the
        robot was in while each segment of it was recorded -- an old segment
        keeps the color of the phase active when it was flown through, only
        the newest segment picks up the current one.

        Args:
            path: The ball's actual path, as a sequence of 3-D positions.
            phases: The robot's stroke phase at each point in the path.
        """
        if path.shape[0] < 2 or len(phases) != path.shape[0]:
            return

        for phase, start, end in phase_runs(phases):
            self._draw_path_segment(path[start : end + 1], phase)

    def _draw_path_segment(self, segment: np.ndarray, phase: Phase) -> None:
        if segment.shape[0] < 2:
            return
        
        color = PHASE_COLOR[phase]
        faded = (color[0], color[1], color[2], Color.ACTUAL[3])
        self.overlay.polyline(segment, faded, width=self._line_width(2.5))

    def _draw_target(self, agent: RallyAgent) -> None:
        target = agent.scene.target.position
        self.overlay.sphere(target, 0.045, Color.TARGET, label="target")
        self.overlay.line(target, target + np.array([0.0, 0.0, 0.25]), Color.TARGET, self._line_width(2.0))

    def _draw_state_indicator(self, phase: Phase) -> None:
        """A text label hovering above the robot, naming its current stroke phase."""
        if self.data is None:
            return
        
        position = self.data.xpos[self._robot_body_id] + np.array([0.0, 0.0, _STATE_LABEL_HEIGHT])
        self.overlay.label(position, phase.value.upper(), color=Color.TEXT)
