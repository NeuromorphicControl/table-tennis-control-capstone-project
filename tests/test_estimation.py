"""Observer and forward model."""

from __future__ import annotations

import numpy as np
import pytest

from table_tennis_control.config import GRAVITY, BallSpec, ObserverConfig, TableSpec
from table_tennis_control.estimation import BallObserver, BallPredictor
from table_tennis_control.physics import flight_position, flight_velocity


def test_observer_recovers_velocity_from_positions_only():
    """The observer only ever sees positions and has to infer the velocity."""
    dt = 1e-3
    observer = BallObserver(dt, ObserverConfig(bandwidth=90.0))

    position = np.array([0.0, 2.0, 1.2])
    velocity = np.array([0.2, -5.0, 1.5])
    observer.reset(position, np.zeros(3))  # deliberately wrong initial guess

    for step in range(400):
        time = step * dt
        truth = flight_position(np.array([time]), position, velocity)[0]
        observer.update(truth)

    expected = flight_velocity(np.array([0.4]), velocity)[0]
    assert np.allclose(observer.velocity, expected, atol=0.02)


def test_observer_estimates_an_unmodelled_acceleration():
    """A constant extra acceleration shows up in the disturbance state."""
    dt = 1e-3
    observer = BallObserver(dt, ObserverConfig(bandwidth=80.0))
    drag = np.array([0.0, 1.5, -0.4])

    position = np.array([0.0, 1.0, 1.0])
    velocity = np.array([0.0, -4.0, 1.0])
    observer.reset(position, velocity)

    for step in range(800):
        time = step * dt
        truth = flight_position(np.array([time]), position, velocity, GRAVITY + drag)[0]
        observer.update(truth)

    assert np.allclose(observer.disturbance, drag, atol=0.15)


def test_observer_reinitialises_after_a_jump():
    observer = BallObserver(1e-3, ObserverConfig(reset_innovation=0.05))
    observer.reset(np.zeros(3), np.zeros(3))
    observer.update(np.array([0.0, 0.0, 0.0]))
    observer.update(np.array([5.0, 0.0, 0.0]))  # teleport
    assert np.allclose(observer.position, [5.0, 0.0, 0.0])


class TestPredictor:
    table = TableSpec()
    ball = BallSpec()

    def predictor(self):
        return BallPredictor(self.table, self.ball, sample_time=0.005, horizon=3.0)

    def test_ball_bounces_off_the_table(self):
        trajectory = self.predictor().predict(np.array([0.0, 0.5, 1.5]), np.zeros(3))
        assert trajectory.events
        first = trajectory.events[0]
        assert first.surface == "table"
        assert first.position[2] == pytest.approx(self.ball.bounce_plane(self.table), abs=1e-9)
        assert first.outgoing_velocity[2] > 0.0

    def test_ball_beside_the_table_falls_to_the_floor(self):
        trajectory = self.predictor().predict(np.array([1.6, 0.5, 1.5]), np.zeros(3))
        assert trajectory.events[0].surface == "floor"
        assert trajectory.terminal == "floor"

    def test_low_shot_hits_the_net(self):
        """Above the table but below the top of the net: the net comes first."""
        trajectory = self.predictor().predict(np.array([0.0, -0.3, 0.85]), np.array([0.0, 5.0, 0.2]))
        assert trajectory.events[0].surface == "net"
        assert trajectory.terminal == "net"

    def test_bounce_is_detected_when_already_touching(self):
        """A ball resting exactly on the plane must not fall through it."""
        start = np.array([0.0, -0.5, self.ball.bounce_plane(self.table)])
        trajectory = self.predictor().predict(start, np.array([0.0, -1.0, -2.0]))
        assert trajectory.events[0].surface == "table"
        assert trajectory.events[0].time == pytest.approx(0.0)

    def test_prediction_is_sampled_over_the_whole_horizon(self):
        """A serve that clears the net is predicted across all of its bounces."""
        trajectory = self.predictor().predict(np.array([0.0, 2.0, 1.2]), np.array([0.0, -6.0, 1.2]))
        assert trajectory.events[0].surface == "table"
        assert len(trajectory) > 100
        assert np.all(np.diff(trajectory.times) >= 0.0)
