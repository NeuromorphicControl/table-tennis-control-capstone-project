"""Extended-state observer for the ball (Lecture 4).

The controller never reads ``qvel`` of the ball.  It only gets a position
measurement and has to reconstruct the velocity -- and the unmodelled
acceleration (drag, spin, a mis-calibrated gravity) -- from it.  This is the
classic Luenberger observer with the disturbance appended to the state:

.. math::

    \\dot{\\hat p} &= \\hat v + \\ell_1 (y - \\hat p) \\\\
    \\dot{\\hat v} &= g + \\hat d + \\ell_2 (y - \\hat p) \\\\
    \\dot{\\hat d} &= \\ell_3 (y - \\hat p)

With the three observer poles placed at :math:`-\\omega` the gains become
:math:`\\ell = (3\\omega,\\, 3\\omega^2,\\, \\omega^3)`.  The estimated
disturbance :math:`\\hat d` is exactly the quantity an active-disturbance-
rejection scheme would cancel; here it is fed forward into the ballistic
forward model instead.
"""

from __future__ import annotations

from collections import deque

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
        self._delay_buffer: deque[np.ndarray] = deque(maxlen=max(1, self.config.measurement_delay_steps + 1))

    # ------------------------------------------------------------------ state
    @property
    def acceleration(self) -> np.ndarray:
        """Total acceleration the forward model should use."""
        return self.gravity + self.disturbance

    @property
    def lag_ticks(self) -> int:
        """Ticks that :attr:`position`/:attr:`velocity` currently lag real time by.

        Grows from 0 immediately after a :meth:`reset` up to the configured
        ``measurement_delay_steps`` as the delay buffer fills, then stays
        pinned there: there is no ``measurement_delay_steps``-old sample to
        track before that many ticks have actually passed since the reset,
        so during that window the estimate legitimately still represents the
        reset instant itself (see :meth:`update`) rather than a fixed delay
        that hasn't had time to elapse yet. Callers doing delay compensation
        (:meth:`RallyAgent._estimated_state`) must roll forward by *this*,
        not by the configured delay directly, or they over-compensate right
        after every reset.
        """
        if len(self._delay_buffer) == self._delay_buffer.maxlen:
            return self.config.measurement_delay_steps
        return len(self._delay_buffer)

    def reset(self, position, velocity=None, measurement=None) -> None:
        """Re-initialise the estimate, e.g. after a serve or a paddle impact.

        Args:
            position: Filtered position estimate to seed with
            velocity: Seed velocity
            measurement: Raw measurement to seed the delay buffer with
        
        Note:
            The delay buffer is cleared and the first sample is seeded with
            ``measurement`` (or ``position`` if ``measurement`` is None).  The
            observer then runs on the delayed sample, which is what the
            predictor has to compensate for.  To avoid a one-tick gap in the
            estimate, ``position`` is seeded with a *predicted* position one
            tick ahead of the delayed sample, so that when the next update()
            runs a full tick later than this reset() call, the estimate will
            be exactly one tick behind the real world again.
        """
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float).copy()
        self.disturbance = np.zeros(3)
        self.innovation = np.zeros(3)
        self.in_contact = False
        self._settle_velocity = None
        self._previous_measurement = self.position.copy() if measurement is None else np.asarray(measurement, dtype=float).copy()
        self._delay_buffer.clear()
        self.initialised = True

    # ----------------------------------------------------------------- update
    def update(self, measurement) -> np.ndarray:
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
            measurement: Measured ball position of the *current* step
        
        Returns:
            Estimated ball velocity of the *current* step
        
        Note:
            The returned velocity is always the estimate for the *current*
            step, not the delayed one.  The observer runs on a delayed sample
            (see :attr:`lag_ticks`), but the returned velocity is always the
            one that corresponds to the current measurement, so that the
            controller can use it directly without having to roll it forward
            by the delay itself.  The returned velocity is also the one that
            will be used in the next step's forward model, so that the
            controller can use it to predict the ball's future position and
            plan its own motion accordingly.
        """
        measurement = np.asarray(measurement, dtype=float)
        self._delay_buffer.append(measurement.copy())
        delayed = self._delay_buffer[0]
        delay_settled = len(self._delay_buffer) == self._delay_buffer.maxlen

        # Initialise the observer on the first measurement, or if it was reset
        if not self.initialised:
            self.reset(delayed)
            return self.velocity

        # If the delay buffer isn't full yet, the observer can't run on a delayed sample, so just return the current velocity estimate.
        # The next tick will have a new measurement and the buffer will be one step closer to being full.
        if not delay_settled:
            return self.velocity

        # Compute the raw two-sample velocity from the delayed sample and the previous one, then update the previous sample for the next tick.
        raw_velocity = (delayed - self._previous_measurement) / self.dt
        self._previous_measurement = delayed.copy()

        # If is frozen due to contact, check if velocity has settled again
        if self.in_contact:
            if self._settle_velocity is not None and float(np.linalg.norm(raw_velocity - self._settle_velocity)) <= self.config.contact_clear_threshold:
                position = delayed + raw_velocity * self.dt
                self.reset(position, raw_velocity, measurement=delayed)
            else:
                self._settle_velocity = raw_velocity
            return self.velocity

        # Compute the innovation (measurement error) for the delayed sample
        innovation = delayed - self.position

        # If the innovation is too large, reset the observer to the delayed sample and the raw velocity
        if float(np.linalg.norm(innovation)) > self.config.reset_innovation:
            self.reset(delayed, raw_velocity)
            return self.velocity

        # If the raw velocity deviates too much from the estimated velocity, flag a contact and freeze the estimate until the raw velocity settles again
        if float(np.linalg.norm(raw_velocity - self.velocity)) > self.config.contact_detect_threshold:
            self.in_contact = True
            self._settle_velocity = raw_velocity
            return self.velocity

        # Update the observer state using the delayed sample and the innovation
        position_rate = self.velocity + self.gains[0] * innovation
        velocity_rate = self.gravity + self.disturbance + self.gains[1] * innovation
        disturbance_rate = self.gains[2] * innovation

        # Update the state using Euler integration
        self.position = self.position + self.dt * position_rate
        self.velocity = self.velocity + self.dt * velocity_rate
        self.disturbance = self.disturbance + self.dt * disturbance_rate

        self.innovation = innovation
        return self.velocity
