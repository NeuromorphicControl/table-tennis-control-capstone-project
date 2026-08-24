"""Serves balls from the opponent's side of the table.

The launcher does not pick a velocity at random -- it picks *where the serve
should bounce* on the robot's half and inverts the ballistic flight for it.
That way every serve is a legal, playable ball instead of an arbitrary throw
from the middle of the world.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import GRAVITY, ArmSpec, BallSpec, LauncherConfig, TableSpec
from ..kinematics import point_in_box
from ..physics import bounce_velocity, flight_position, flight_velocity, velocity_for_flight_time


@dataclass
class Serve:
    """A single planned serve."""

    position: np.ndarray
    velocity: np.ndarray
    bounce_point: np.ndarray
    time_to_bounce: float

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))


class BallLauncher:
    """Generates incoming balls from the opponent's half."""

    def __init__(self, ball, table: TableSpec, ball_spec: BallSpec, config: LauncherConfig | None = None, generator: np.random.Generator | None = None, strike_zone: tuple[tuple[float, float], ...] | None = None):
        """Initialize the ball launcher.
        
        Args:
            ball: The :class:`Ball` to be launched.
            table: The :class:`TableSpec` describing the table.
            ball_spec: The :class:`BallSpec` describing the ball.
            config: The :class:`LauncherConfig` describing the launcher's behavior.
            generator: The random number generator to use for sampling.
            strike_zone: The 3D box in which the ball must pass after bouncing, in world coordinates. If None, the default workspace of the robot arm is used.
        """
        self.ball = ball
        self.table = table
        self.ball_spec = ball_spec
        self.config = config or LauncherConfig()
        self.generator = generator or np.random.default_rng()
        self.strike_zone = strike_zone

    # ------------------------------------------------------------------ plan
    def plan_serve(self, attempts: int = 40) -> Serve:
        """Sample a serve that clears the net and bounces on the robot's half."""
        opponent = self.table.opponent_side
        robot = self.table.robot_side
        bounce_z = self.ball_spec.bounce_plane(self.table)

        fallback: Serve | None = None
        for _ in range(attempts):
            # Sample a random start point in the opponent's half and a random bounce point in the robot's half
            start = np.array([
                self.generator.uniform(*self.config.x_range),
                opponent * self.generator.uniform(*self.config.y_range),
                self.generator.uniform(*self.config.z_range),
            ])
            bounce = np.array([
                self.generator.uniform(*self.config.bounce_x_range),
                robot * abs(self.generator.uniform(*self.config.bounce_y_range)),
                bounce_z,
            ])

            # Sample a flight time and invert the ballistic flight to get the required launch velocity
            flight_range = self.config.shallow_flight_time_range if self.generator.random() < self.config.shallow_serve_probability else self.config.flight_time_range

            flight_time = float(self.generator.uniform(*flight_range))
            velocity = velocity_for_flight_time(start, bounce, flight_time, GRAVITY) # type: ignore

            # If the serve is playable, return it. Otherwise, keep the first one as a fallback.
            serve = Serve(start, velocity, bounce, flight_time)
            fallback = fallback or serve
            if self._is_playable(serve):
                return serve

        assert fallback is not None
        return fallback

    def _is_playable(self, serve: Serve) -> bool:
        """Check net clearance and that the ball arrives in a hittable region."""
        # 1. clear the net on the way in
        if abs(serve.velocity[1]) < 1e-6:
            return False

        time_at_net = -serve.position[1] / serve.velocity[1]
        if not 0.0 < time_at_net < serve.time_to_bounce:
            return False

        height_at_net = flight_position(np.array([time_at_net]), serve.position, serve.velocity)[0, 2]
        if height_at_net < self.table.net_height + self.config.net_clearance:
            return False

        # 2. after the bounce, the ball must be moving toward the robot's half and pass through the strike zone
        impact_velocity = flight_velocity(np.array([serve.time_to_bounce]), serve.velocity)[0]
        rebound = bounce_velocity(impact_velocity, self.ball_spec.table_restitution, self.ball_spec.table_tangential)

        # After the bounce, the ball must be moving toward the robot's half of the table.
        rebound = rebound + GRAVITY * self.ball_spec.table_impact_duration
        if rebound[1] * self.table.robot_side <= 0.0:
            return False

        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.strike_zone or ArmSpec().workspace

        # Check that the ball passes through the strike zone after the bounce
        # The ball is considered playable if it passes through the strike zone at least 8 times in a 1.2 second window after the bounce
        times = np.arange(0.0, 1.2, 0.01)
        path = flight_position(times, serve.bounce_point, rebound)
        inside = point_in_box(path, ((x_min, x_max), (min(y_min, y_max), max(y_min, y_max)), (z_min, z_max)))

        return bool(np.count_nonzero(inside) >= 8)

    # ------------------------------------------------------------------ act
    def serve(self) -> Serve:
        """Plan a serve and apply it to the simulated ball."""
        serve = self.plan_serve()
        self.ball.set_state(serve.position, serve.velocity)
        return serve
