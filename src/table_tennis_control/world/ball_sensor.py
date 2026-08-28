"""Implements a delayed ball-position sensor that mimics the fixed latency of a real vision system."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..config import SensorConfig
from .ball import Ball


class BallSensor:
    """Reports the ball's position with a fixed sensing latency."""

    def __init__(self, config: SensorConfig | None = None):
        self.config = config or SensorConfig()
        self._buffer: deque[np.ndarray] = deque(maxlen=max(1, self.config.delay_steps + 1))

    @property
    def settled(self) -> bool:
        """Whether the buffer holds a full ``delay_steps``-old sample yet.

        False for the first ``delay_steps`` ticks after every :meth:`reset`
        (there simply hasn't been a sample that old since the reset), during
        which :meth:`measure` still returns the oldest sample available --
        see :attr:`lag_ticks`.
        """
        return len(self._buffer) == self._buffer.maxlen

    @property
    def lag_ticks(self) -> int:
        """Ticks that the most recent :meth:`measure` result lags real time by.

        Grows from 0 immediately after a :meth:`reset` up to the configured
        ``delay_steps`` as the buffer fills, then stays pinned there: there
        is no ``delay_steps``-old sample to report before that many ticks
        have actually passed since the reset, so during that window the
        measurement legitimately still represents the reset instant itself
        rather than a fixed delay that hasn't had time to elapse yet.
        Callers doing delay compensation
        (:meth:`~table_tennis_control.agent.RallyAgent._estimated_state`)
        must roll forward by *this*, not by the configured delay directly,
        or they over-compensate right after every reset.
        """
        if self.settled:
            return self.config.delay_steps
        return len(self._buffer)

    def reset(self) -> None:
        """Discard the buffer, e.g. right after a serve or a paddle impact.

        The next ``delay_steps`` calls to :meth:`measure` will report a
        growing lag as the buffer fills again -- see :attr:`lag_ticks`.
        """
        self._buffer.clear()

    def measure(self, ball: Ball) -> np.ndarray:
        """Delayed position measurement of ``ball``.

        Pushes the ball's true (instantaneous) position into the delay
        buffer and returns the oldest sample in it -- the one exactly
        :attr:`lag_ticks` control steps old.
        """
        self._buffer.append(ball.measure())
        return self._buffer[0].copy()
