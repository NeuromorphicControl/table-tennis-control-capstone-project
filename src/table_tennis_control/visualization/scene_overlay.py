"""Provides drawing primitives for rendering diagnostic overlays directly into the MuJoCo scene as user geometry."""

from __future__ import annotations

import mujoco
import numpy as np

from .colors import Color

__all__ = ["SceneOverlay"]

_IDENTITY = np.eye(3).flatten()


class SceneOverlay:
    """Appends geoms to an :class:`mujoco.MjvScene` and resets them each frame.

    Works both with the interactive viewer (``viewer.user_scn``) and with the
    off-screen ``Renderer`` (``renderer.scene``); the only difference is that
    the renderer's scene already contains the model's own geometry, so the
    overlay must not reset ``ngeom`` there.
    """

    def __init__(self, scene, reset_on_begin: bool = True):
        self.scene = scene
        self.reset_on_begin = reset_on_begin
        self._base = 0

    # ------------------------------------------------------------------ frame
    def begin(self) -> None:
        """Start a new frame."""
        if self.reset_on_begin:
            self.scene.ngeom = 0
        self._base = self.scene.ngeom

    @property
    def capacity_left(self) -> int:
        return int(self.scene.maxgeom - self.scene.ngeom)

    def _next(self):
        if self.capacity_left <= 0:
            return None
        geom = self.scene.geoms[self.scene.ngeom]
        self.scene.ngeom += 1
        return geom

    # ------------------------------------------------------------- primitives
    def sphere(self, position, radius: float, color, label: str = "") -> None:
        geom = self._next()
        if geom is None:
            return
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0.0, 0.0]), np.asarray(position, dtype=float), _IDENTITY, np.asarray(color, dtype=np.float32)) # type: ignore
        geom.label = label

    def line(self, start, end, color, width: float = 3.0) -> None:
        geom = self._next()
        if geom is None:
            return
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), _IDENTITY, np.asarray(color, dtype=np.float32)) # type: ignore
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(start, dtype=float), np.asarray(end, dtype=float)) # type: ignore

    def arrow(self, start, direction, color, width: float = 0.008) -> None:
        end = np.asarray(start, dtype=float) + np.asarray(direction, dtype=float)
        geom = self._next()
        if geom is None:
            return
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), _IDENTITY, np.asarray(color, dtype=np.float32)) # type: ignore
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(start, float), end) # type: ignore

    def polyline(self, points: np.ndarray, color, width: float = 3.0, stride: int = 1) -> None:
        """Draw a sampled curve as a strip of line segments."""
        points = np.asarray(points, dtype=float)
        if points.shape[0] < 2:
            return
        points = points[::max(1, stride)]
        budget = min(points.shape[0] - 1, self.capacity_left)
        for index in range(budget):
            self.line(points[index], points[index + 1], color, width)

    def label(self, position, text: str, color=Color.TEXT) -> None:
        """A text label anchored at a (nearly invisible) marker."""
        geom = self._next()
        if geom is None:
            return
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([1e-4, 0.0, 0.0]), np.asarray(position, dtype=float), _IDENTITY, np.asarray(color, dtype=np.float32)) # type: ignore
        geom.label = text[:99]
