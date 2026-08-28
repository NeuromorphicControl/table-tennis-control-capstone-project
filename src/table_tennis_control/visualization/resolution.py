"""Scales overlay line widths and label font presets so they stay legible across different render resolutions."""

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
