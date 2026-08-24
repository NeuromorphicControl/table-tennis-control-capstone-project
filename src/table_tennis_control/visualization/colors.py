"""Every color the overlay and the debug plots use, in one place.

:data:`PHASE_COLOR` is the one palette that has to carry real information at
a glance. It is how both the live overlay and the offline per-serve plots
tell the robot's four stroke phases apart. Everything else in :class:`Color`
and :data:`JOINT_COLORS` is chosen to sit in the gaps between those four
hues, so a trajectory/marker color is never mistaken for a phase and vice
versa, while still reading as one consistent, cohesive set rather than an
unrelated grab-bag.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ..agent import Phase

__all__ = ["Color", "PHASE_COLOR", "JOINT_COLORS", "phase_runs"]


class Color:
    """Palette shared by the overlay elements (never a :data:`PHASE_COLOR` hue)."""

    PREDICTION = (0.48, 0.44, 0.82, 0.85)
    PLANNED = (0.84, 0.33, 0.61, 0.95)
    ACTUAL = (0.20, 0.70, 0.45, 0.45)
    STRIKE = (0.91, 0.20, 0.16, 1.00)
    BOUNCE = (0.61, 0.77, 0.24, 1.00)
    TARGET = (0.91, 0.79, 0.24, 0.95)
    GRID = (1.00, 1.00, 1.00, 0.22)
    TEXT = (1.00, 1.00, 1.00, 1.00)


# Colors for the four phases of the robot's stroke, in RGBA format
PHASE_COLOR: dict[Phase, tuple[float, float, float, float]] = {
    Phase.IDLE: (0.58, 0.58, 0.58, 1.0),
    Phase.TRACK: (0.00, 0.45, 0.70, 1.0),
    Phase.SWING: (0.84, 0.37, 0.00, 1.0),
    Phase.RECOVER: (0.00, 0.62, 0.45, 1.0),
}


# Colors for the robot's joints, in hex format
JOINT_COLORS: tuple[str, ...] = (
    "#4C6EF5",
    "#F76707",
    "#37B24D",
    "#E64980",
    "#1098AD",
    "#F59F00",
    "#7048E8",
    "#495057",
)


def phase_runs(phases: Sequence[Phase]) -> Iterator[tuple[Phase, int, int]]:
    """Yield ``(phase, start, end)`` for each run of consecutive equal entries.

    ``end`` is the *last* index of the run, except that -- so that a caller
    slicing ``[start:end+1]`` gets a segment sharing its boundary point with
    the next run, avoiding a visible gap between them -- the end of every
    run but the last is actually the first index of the *following* run.

    Args:
        phases: A sequence of :class:`Phase` values.
    
    Yields:
        Tuples of ``(phase, start, end)`` for each run of consecutive equal entries in ``phases``.
    """
    if len(phases) == 0:
        return
    
    start = 0
    last = len(phases) - 1
    for index in range(1, len(phases)):
        if phases[index] != phases[start]:
            yield phases[start], start, index
            start = index
    yield phases[start], start, last
