"""Implements an extended-state Luenberger observer that reconstructs the ball's velocity and residual acceleration from delayed position measurements alone."""

from __future__ import annotations

import numpy as np

from ..config import GRAVITY, ObserverConfig


class BallObserver:
    """Estimates position, velocity and residual acceleration of the ball."""

    def __init__(self, dt: float, config: ObserverConfig | None = None, gravity: np.ndarray | None = None):
        """Create a new observer.

        Args:
            dt: Time step between measurements.
            config: Observer configuration.
            gravity: Gravity vector to use in the forward model. Defaults to the global ``GRAVITY`` constant.
        """
        self.dt = float(dt)
        self.config = config or ObserverConfig()
        self.gravity = np.asarray(GRAVITY if gravity is None else gravity, dtype=float)

        omega = float(self.config.bandwidth)
        self.gains = np.array([3.0 * omega, 3.0 * omega**2, omega**3])

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.disturbance = np.zeros(3)
        self.innovation = np.zeros(3)
        self.initialised = False
        self.in_contact = False

        self._previous_measurement: np.ndarray | None = None
        self._settle_velocity: np.ndarray | None = None

    # ------------------------------------------------------------------ state
    @property
    def acceleration(self) -> np.ndarray:
        """Total acceleration the forward model should use."""
        return self.gravity + self.disturbance

    def reset(self, position, velocity=None, measurement=None) -> None:
        """Re-initialise the estimate, e.g. after a serve or a paddle impact.

        Args:
            position: Filtered position estimate to seed with
            velocity: Seed velocity
            measurement: Raw measurement to seed the raw-velocity finite
                difference with; defaults to ``position`` if None.

        Note:
            To avoid a one-tick gap in the estimate, ``position`` is
            sometimes seeded with a *predicted* position one tick ahead of
            ``measurement`` (see the contact-settle branch of
            :meth:`update`), so that when the next ``update()`` runs a full
            tick later than this ``reset()`` call, the estimate lines back
            up with the real world exactly one tick later.
        """
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float).copy()
        self.disturbance = np.zeros(3)
        self.innovation = np.zeros(3)
        self.in_contact = False
        self._settle_velocity = None
        self._previous_measurement = self.position.copy() if measurement is None else np.asarray(measurement, dtype=float).copy()
        self.initialised = True

    # ----------------------------------------------------------------- update
    def update(self, measurement, settled: bool = True) -> np.ndarray:
        """Run one observer step and return the estimated velocity.

        Detects contacts (bounces, paddle hits) from the position stream.
        The observer already tracks a *smoothed* velocity estimate
        (:attr:`velocity`); comparing it each tick against a *raw*
        two-sample finite-difference velocity computed from consecutive
        measurements exposes a genuine impact almost immediately, because
        a real contact changes the ball's velocity by metres per second
        within one or two milliseconds (far more than gravity, drag or
        measurement noise ever could between two ticks).

        Once a deviation is flagged the estimate is frozen exactly as
        before, and every subsequent raw two-sample velocity is compared
        against the *previous* one instead of against :attr:`velocity`:
        once two consecutive raw velocities agree with each other, the
        motion has become smooth again and the contact is over, regardless
        of what caused it.

        Args:
            measurement: Measured ball position for this step (typically
                :meth:`~table_tennis_control.world.ball_sensor.BallSensor.measure`,
                i.e. a *delayed* reading -- the observer doesn't care either way).
            settled: Whether ``measurement`` is a genuinely new sample rather
                than a repeat of the previous call's, e.g.
                :attr:`~table_tennis_control.world.ball_sensor.BallSensor.settled`.
                While False the estimate is held at its current value, since
                the two-sample finite difference below would otherwise see
                zero motion between two identical measurements and mistake it
                for the ball having stopped.

        Returns:
            Estimated ball velocity for this step.
        """
        measurement = np.asarray(measurement, dtype=float)

        # Initialise the observer on the first measurement, or if it was reset
        if not self.initialised:
            self.reset(measurement)
            return self.velocity

        # If the caller's sample isn't a fresh one yet, just return the current velocity estimate unchanged.
        if not settled:
            return self.velocity

        # Compute the raw two-sample velocity from this measurement and the previous one, then update the previous sample for the next tick.
        raw_velocity = (measurement - self._previous_measurement) / self.dt
        self._previous_measurement = measurement.copy()

        # If is frozen due to contact, check if velocity has settled again
        if self.in_contact:
            if self._settle_velocity is not None and float(np.linalg.norm(raw_velocity - self._settle_velocity)) <= self.config.contact_clear_threshold:
                position = measurement + raw_velocity * self.dt
                self.reset(position, raw_velocity, measurement=measurement)
            else:
                self._settle_velocity = raw_velocity
            return self.velocity

        # Compute the innovation (measurement error) for this measurement
        innovation = measurement - self.position

        # If the innovation is too large, reset the observer to this measurement and the raw velocity
        if float(np.linalg.norm(innovation)) > self.config.reset_innovation:
            self.reset(measurement, raw_velocity)
            return self.velocity

        # If the raw velocity deviates too much from the estimated velocity, flag a contact and freeze the estimate until the raw velocity settles again
        if float(np.linalg.norm(raw_velocity - self.velocity)) > self.config.contact_detect_threshold:
            self.in_contact = True
            self._settle_velocity = raw_velocity
            return self.velocity

        # Update the observer state using this measurement and the innovation
        position_rate = self.velocity + self.gains[0] * innovation
        velocity_rate = self.gravity + self.disturbance + self.gains[1] * innovation
        disturbance_rate = self.gains[2] * innovation

        # Update the state using Euler integration
        self.position = self.position + self.dt * position_rate
        self.velocity = self.velocity + self.dt * velocity_rate
        self.disturbance = self.disturbance + self.dt * disturbance_rate

        self.innovation = innovation
        return self.velocity
