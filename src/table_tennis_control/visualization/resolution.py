"""Keeps overlay line widths and label text legible across render resolutions.

MuJoCo draws user-geom lines (:meth:`SceneOverlay.line`/``polyline``) at a
width given in on-screen *pixels*, and its label font is one of a handful of
discrete presets (:class:`mujoco.mjtFontScale`) -- neither tracks the
requested render resolution on its own. A width picked for 720p all but
disappears at 4K and looks disproportionately fat at 240p. ``REFERENCE_HEIGHT``
is the resolution the hardcoded widths in :mod:`play_overlay` were tuned for;
everything here scales relative to it, while the robot mesh itself (rendered
through the ordinary camera projection) needs no such correction.
"""

from __future__ import annotations

import mujoco

__all__ = ["REFERENCE_HEIGHT", "line_width_scale", "nearest_font_scale"]

REFERENCE_HEIGHT = 720.0

_FONT_SCALES = (
    (50, mujoco.mjtFontScale.mjFONTSCALE_50),   # type: ignore
    (100, mujoco.mjtFontScale.mjFONTSCALE_100), # type: ignore
    (150, mujoco.mjtFontScale.mjFONTSCALE_150), # type: ignore
    (200, mujoco.mjtFontScale.mjFONTSCALE_200), # type: ignore
    (250, mujoco.mjtFontScale.mjFONTSCALE_250), # type: ignore
    (300, mujoco.mjtFontScale.mjFONTSCALE_300), # type: ignore
)


def line_width_scale(height: int) -> float:
    """Multiplier for overlay line widths so they stay legible at ``height``."""
    return max(float(height), 1.0) / REFERENCE_HEIGHT


def nearest_font_scale(height: int):
    """Discrete MuJoCo font preset closest to the continuous resolution scale."""
    target = 150.0 * line_width_scale(height)
    return min(_FONT_SCALES, key=lambda item: abs(item[0] - target))[1]
