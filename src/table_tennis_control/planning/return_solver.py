"""Solves in closed form for the paddle-exit ball velocity that produces a legal one-bounce return to a target landing point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import GRAVITY, BallSpec, TableSpec
from ..physics import bounce_velocity, flight_position, quadratic_roots, time_to_plane

__all__ = ["ReturnSolutions", "solve_bounce_return"]


@dataclass
class ReturnSolutions:
    """A batch of candidate returns, shaped ``(n_strikes, n_options)``."""

    outgoing_velocity: np.ndarray  # (n, k, 3)
    bounce_point: np.ndarray  # (n, k, 3)
    time_to_bounce: np.ndarray  # (n, k)
    time_after_bounce: np.ndarray  # (n, k)
    feasible: np.ndarray  # (n, k) bool

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.outgoing_velocity, axis=-1)

    @property
    def total_time(self) -> np.ndarray:
        """Flight time from the paddle to the landing point."""
        return np.nan_to_num(self.time_to_bounce, nan=0.0) + self.time_after_bounce


def _net_clearance_ok(
    start: np.ndarray,
    velocity: np.ndarray,
    table: TableSpec,
    acceleration: np.ndarray,
    clearance: float,
) -> np.ndarray:
    """Mask of trajectories that pass over the net instead of into it.

    Trajectories that never cross ``y = 0`` while moving towards the opponent
    are rejected as well -- the ball has to end up on the other side.
    """
    velocity_y = velocity[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        time_at_net = -start[..., 1] / velocity_y
        crossing = (velocity_y * table.opponent_side > 0.0) & (time_at_net > 0.0)

        height = (
            start[..., 2]
            + velocity[..., 2] * time_at_net
            + 0.5 * acceleration[2] * time_at_net**2
        )
        lateral = np.abs(start[..., 0] + velocity[..., 0] * time_at_net)
    over_net = height > table.net_height + clearance
    beside_net = lateral > table.net_half_width

    return crossing & (over_net | beside_net)


def solve_bounce_return(
    strike_positions: np.ndarray,
    target: np.ndarray,
    table: TableSpec,
    ball: BallSpec,
    post_bounce_times: np.ndarray,
    bounce_side: int,
    acceleration: np.ndarray | None = None,
    bounce_margin: float = 0.1,
    net_clearance: float = 0.05,
    max_speed: float = 12.0,
    min_time_to_bounce: float = 0.05,
) -> ReturnSolutions:
    """Returns that touch one half of the table before reaching ``target``.

    Args:
        strike_positions: ``(n, 3)`` candidate impact points.
        target: ``(3,)`` landing point on the opponent's side.
        table: Table geometry.
        ball: Ball/​surface properties (restitution).
        post_bounce_times: ``(m,)`` grid of post-bounce flight times.
        bounce_side: ``table.robot_side`` to bounce on the robot's own half
            (the net is then crossed on the second leg, bounce -> target),
            ``table.opponent_side`` for a standard return (the net is crossed
            on the first leg, strike -> bounce, instead).
        acceleration: Constant acceleration, defaults to gravity.
        bounce_margin: Safety margin keeping the bounce off the table edge [m].
        net_clearance: Required height margin over the net [m].
        max_speed: Reject returns faster than this [m/s].
        min_time_to_bounce: Reject solutions that reach the table too quickly.
    """
    acceleration = GRAVITY if acceleration is None else np.asarray(acceleration, float)
    strike_positions = np.atleast_2d(np.asarray(strike_positions, dtype=float))
    target = np.asarray(target, dtype=float)
    post_bounce_times = np.asarray(post_bounce_times, dtype=float).reshape(-1)

    bounce_z = ball.bounce_plane(table)
    restitution = ball.table_restitution
    tangential = ball.table_tangential
    a_z = float(acceleration[2])

    n = strike_positions.shape[0]
    m = post_bounce_times.shape[0]

    t2 = post_bounce_times[None, :]  # (1, m)
    # The instantaneous bounce law used to eliminate v1' below ignores the
    # real (non-zero) time the table impact takes to resolve, during which
    # gravity keeps acting on the ball -- see BallSpec.table_impact_duration.
    # Compensating the target by that same term here is equivalent to
    # solving for the idealised v1' that, once the real impact has added
    # ``a_z * table_impact_duration`` to it, still lands the ball on target.
    target_z = target[2] - a_z * ball.table_impact_duration * t2  # (1, m)
    lead = (target_z - bounce_z - 0.5 * a_z * t2**2) / (restitution * t2)  # (1, m)

    lower, upper = quadratic_roots(
        np.full((n, m), 0.5 * a_z),
        np.broadcast_to(lead, (n, m)),
        np.broadcast_to((bounce_z - strike_positions[:, 2])[:, None], (n, m)),
    )

    # Both roots are physically meaningful (a flat, fast shot and a lofted
    # one); keep them as separate options and let the cost function decide.
    time_to_bounce = np.stack([lower, upper], axis=-1).reshape(n, 2 * m)
    time_after_bounce = np.broadcast_to(post_bounce_times[None, :, None], (n, m, 2)).reshape(n, 2 * m)

    start = np.broadcast_to(strike_positions[:, None, :], (n, 2 * m, 3))

    with np.errstate(invalid="ignore", divide="ignore"):
        weight = time_to_bounce + tangential * time_after_bounce
        bounce_xy = (
            target[None, None, :2] * time_to_bounce[..., None]
            + tangential * time_after_bounce[..., None] * start[..., :2]
        ) / weight[..., None]
        bounce_point = np.concatenate([bounce_xy, np.full((*bounce_xy.shape[:2], 1), bounce_z)], axis=-1)

        outgoing = (bounce_point - start) / time_to_bounce[..., None] - 0.5 * acceleration * time_to_bounce[..., None]
        impact_velocity = outgoing + acceleration * time_to_bounce[..., None]

    # See BallSpec.table_impact_duration.
    rebound = bounce_velocity(impact_velocity, restitution, tangential) + acceleration * ball.table_impact_duration

    feasible = np.isfinite(time_to_bounce) & (time_to_bounce > min_time_to_bounce)
    feasible &= np.all(np.isfinite(outgoing), axis=-1)
    feasible &= table.is_on_half(bounce_point, bounce_side, bounce_margin)
    feasible &= impact_velocity[..., 2] < 0.0  # must be falling onto the table
    feasible &= np.linalg.norm(outgoing, axis=-1) <= max_speed

    # The two roots of the quadratic above are two *independent* candidate
    # shots (different outgoing velocity, same t2), not two crossings of the
    # same flight -- but a candidate's own straight first leg can still clip
    # the table before reaching its own intended bounce_point (typically the
    # "lofted" root of a low strike, which first rises back through table
    # height while still over the table on its way up). Reject those: the
    # earliest time this candidate's own flight reaches the table plane must
    # be the intended bounce itself, not an earlier point over the table.
    with np.errstate(invalid="ignore"):
        earliest_cross = time_to_plane(start, outgoing, bounce_z, acceleration)
        premature = np.isfinite(earliest_cross) & (earliest_cross < time_to_bounce - 1e-4)
        cross_point = flight_position(np.where(premature, earliest_cross, 0.0), start, outgoing, acceleration)
    premature &= table.is_on_half(cross_point, +1) | table.is_on_half(cross_point, -1)
    feasible &= ~premature

    # Only one of the two legs actually crosses the net -- whichever one goes
    # from the robot's side to the opponent's side.  Bouncing on the robot's
    # own half means the *second* leg (bounce -> target) crosses it; bouncing
    # on the opponent's half means the *first* leg (strike -> bounce) does,
    # and the second leg (bounce -> target, both on the opponent's side)
    # never approaches the net at all.
    if bounce_side == table.robot_side:
        feasible &= _net_clearance_ok(bounce_point, rebound, table, acceleration, net_clearance)
    else:
        feasible &= _net_clearance_ok(start, outgoing, table, acceleration, net_clearance)

    return ReturnSolutions(
        outgoing_velocity=outgoing,
        bounce_point=bounce_point,
        time_to_bounce=time_to_bounce,
        time_after_bounce=time_after_bounce.copy(),
        feasible=feasible,
    )
