"""The landing target the robot is supposed to hit."""

from __future__ import annotations

import numpy as np

from ..config import BallSpec, TableSpec, TargetConfig


class Target:
    """A site on the opponent's court that marks the desired landing point.

    The site lives in the world body, so writing ``model.site_pos`` is enough
    to move it; ``data.site_xpos`` is recomputed by ``mj_forward`` and must not
    be written directly.
    """

    def __init__(self, model, data, site_name: str = "target"):
        self.model = model
        self.data = data
        self.site_id = int(model.site(site_name).id)

    @property
    def position(self) -> np.ndarray:
        return self.model.site_pos[self.site_id].copy()

    def set_position(self, position) -> None:
        position = np.asarray(position, dtype=float)

        if position.shape != (3,):
            raise ValueError("target position must be a 3-vector")

        self.model.site_pos[self.site_id] = position
        self.data.site_xpos[self.site_id] = position


class TargetSampler:
    """Draws landing targets on the floor behind the opponent's half.

    The sampled distance is always measured *from the table's back edge*
    (:attr:`TableSpec.half_length`), not as an absolute coordinate, so the
    target can never end up so close to the table that reaching it would
    require the return to clip through the tabletop on its way down.
    """

    def __init__(self, table: TableSpec, ball: BallSpec, config: TargetConfig | None = None, generator: np.random.Generator | None = None):
        self.table = table
        self.ball = ball
        self.config = config or TargetConfig()
        self.generator = generator or np.random.default_rng()

    def sample(self, launch_x: float | None = None) -> np.ndarray:
        """Return a random reachable floor point behind the opponent's half.

        When ``launch_x`` is given, the target is sampled on the opposite side 
        of the table from that launch position, to encourage cross-court returns. 
        If ``launch_x`` is None, the target is sampled anywhere in the opponent's 
        half. The returned point is a 3-vector in world coordinates, with the 
        z coordinate set to the ball's radius above the floor so that the ball can 
        be placed there without clipping through the floor.

        Args:
            launch_x: The x position the serve was launched from, if known.
        
        Returns:
            A 3D point on the floor behind the opponent's half, with z set to the ball's radius.
        """
        lo, hi = self.config.floor_x_range
        gap = self.config.cross_court_gap

        if launch_x is None:
            x = self.generator.uniform(lo, hi)
        elif launch_x >= 0.0:
            x = self.generator.uniform(lo, -gap)
        else:
            x = self.generator.uniform(gap, hi)

        margin = self.generator.uniform(*self.config.floor_margin_range)
        y = self.table.opponent_side * (self.table.half_length + margin)
        return np.array([x, y, self.ball.radius])
