"""Ballistics, impact model and the return solver."""

from __future__ import annotations

import numpy as np
import pytest

from table_tennis_control.config import GRAVITY, BallSpec, TableSpec
from table_tennis_control.estimation import BallPredictor
from table_tennis_control.physics import (
    flight_position,
    flight_velocity,
    paddle_impact_inverse,
    reflect_off_paddle,
    time_to_plane,
    velocity_for_flight_time,
)
from table_tennis_control.planning import solve_bounce_return


def test_flight_matches_closed_form():
    times = np.linspace(0.0, 1.0, 11)
    position = np.array([0.0, 1.0, 2.0])
    velocity = np.array([1.0, -2.0, 3.0])

    positions = flight_position(times, position, velocity)
    expected = position + np.outer(times, velocity) + 0.5 * np.outer(times**2, GRAVITY)
    assert np.allclose(positions, expected)
    assert np.allclose(flight_velocity(times, velocity), velocity + np.outer(times, GRAVITY))


def test_time_to_plane_finds_the_next_crossing():
    position = np.array([0.0, 0.0, 1.0])
    velocity = np.array([0.0, 0.0, 0.0])
    # free fall from 1 m to 0.5 m: t = sqrt(2 * 0.5 / g)
    assert time_to_plane(position, velocity, 0.5) == pytest.approx(np.sqrt(1.0 / 9.81), rel=1e-6)


def test_time_to_plane_is_infinite_when_unreachable():
    position = np.array([0.0, 0.0, 1.0])
    velocity = np.array([0.0, 0.0, 0.0])
    assert np.isinf(time_to_plane(position, velocity, 2.0))


def test_velocity_for_flight_time_hits_the_goal():
    start = np.array([0.0, -1.8, 1.0])
    goal = np.array([0.3, 1.0, 0.78])
    for flight_time in (0.3, 0.6, 0.9):
        velocity = velocity_for_flight_time(start, goal, flight_time)
        arrival = flight_position(np.array([flight_time]), start, velocity)[0]
        assert np.allclose(arrival, goal, atol=1e-9)


@pytest.mark.parametrize("tangential", [1.0, 0.89, 0.6])
def test_paddle_impact_inverse_round_trips(tangential):
    generator = np.random.default_rng(0)
    restitution = 0.87
    for _ in range(200):
        incoming = generator.normal(scale=3.0, size=3)
        outgoing = generator.normal(scale=3.0, size=3)
        normal, speed = paddle_impact_inverse(incoming, outgoing, restitution, tangential)
        achieved = reflect_off_paddle(incoming, normal, speed * normal, restitution, tangential)
        assert np.allclose(achieved, outgoing, atol=1e-9)
        assert np.linalg.norm(normal) == pytest.approx(1.0)


def test_paddle_impact_inverse_broadcasts():
    generator = np.random.default_rng(1)
    incoming = generator.normal(size=(4, 3, 3))
    outgoing = generator.normal(size=(4, 3, 3))
    normal, speed = paddle_impact_inverse(incoming, outgoing, 0.87, 0.89)
    assert normal.shape == (4, 3, 3)
    assert speed.shape == (4, 3)


class TestReturnSolver:
    table = TableSpec()
    ball = BallSpec()

    def _predictor(self):
        return BallPredictor(self.table, self.ball, sample_time=0.004, horizon=4.0)

    def test_bounce_return_lands_on_the_target(self):
        """A serve-style return must touch the own half and then the target."""
        strikes = np.array([[0.1, -1.7, 1.0], [-0.3, -2.0, 1.25], [0.5, -1.6, 0.9]])
        target = np.array([0.2, 0.9, self.ball.bounce_plane(self.table)])
        solutions = solve_bounce_return(
            strikes,
            target,
            self.table,
            self.ball,
            np.array([0.3, 0.45, 0.6]),
            bounce_side=self.table.robot_side,
        )
        assert solutions.feasible.any()

        predictor = self._predictor()
        checked = 0
        for i, k in np.argwhere(solutions.feasible):
            trajectory = predictor.predict(strikes[i], solutions.outgoing_velocity[i, k])
            table_bounces = [e for e in trajectory.events if e.surface == "table"]
            assert len(table_bounces) >= 2, "the return has to bounce twice"
            # first on our own half ...
            assert table_bounces[0].position[1] * self.table.robot_side > 0
            # ... then exactly on the target
            assert np.allclose(table_bounces[1].position[:2], target[:2], atol=0.02)
            checked += 1
        assert checked > 0

    def test_bounce_points_stay_on_the_table(self):
        strikes = np.array([[0.0, -1.8, 1.05]])
        target = np.array([-0.3, 1.1, self.ball.bounce_plane(self.table)])
        solutions = solve_bounce_return(
            strikes,
            target,
            self.table,
            self.ball,
            np.array([0.28, 0.4, 0.55, 0.7]),
            bounce_side=self.table.robot_side,
            bounce_margin=0.1,
        )
        bounces = solutions.bounce_point[solutions.feasible]
        assert bounces.size > 0
        assert np.all(np.abs(bounces[:, 0]) <= self.table.half_width)
        assert np.all(bounces[:, 1] <= 0.0)
        assert np.all(bounces[:, 1] >= -self.table.half_length)

    def test_returns_into_the_net_are_rejected(self):
        """Aiming just over the far edge from very low must not pass the net."""
        strikes = np.array([[0.0, -1.5, 0.62]])
        target = np.array([0.0, 0.25, self.ball.bounce_plane(self.table)])
        solutions = solve_bounce_return(
            strikes,
            target,
            self.table,
            self.ball,
            np.array([0.06]),
            bounce_side=self.table.robot_side,
            net_clearance=0.1,
        )
        assert not solutions.feasible.any()
