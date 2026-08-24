"""Analytic ballistics used by the forward model and by the planner.

All functions are written so that they broadcast over leading batch
dimensions, which lets the planner evaluate thousands of candidate strikes in
a couple of vectorised NumPy calls instead of a Python loop.
"""

from __future__ import annotations

import numpy as np

from .config import GRAVITY

__all__ = [
    "flight_position",
    "flight_velocity",
    "time_to_plane",
    "velocity_for_flight_time",
    "bounce_velocity",
    "reflect_off_paddle",
    "paddle_impact_inverse",
    "quadratic_roots",
]


def flight_position(times: np.ndarray, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray | None = None) -> np.ndarray:
    """Position of a point mass after ``times`` seconds of free flight.

    Args:
        times: ``(n,)`` array of times, or any shape broadcastable against the leading dimensions of ``position``.
        position: ``(..., 3)`` start position.
        velocity: ``(..., 3)`` start velocity.
        acceleration: ``(3,)`` constant acceleration, defaults to gravity.

    Returns:
        ``(..., n, 3)`` array of positions.
    """
    acceleration = GRAVITY if acceleration is None else acceleration
    t = np.asarray(times, dtype=float)[..., np.newaxis]
    return position + velocity * t + 0.5 * np.asarray(acceleration) * t**2


def flight_velocity(times: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray | None = None) -> np.ndarray:
    """Velocity of a point mass after ``times`` seconds of free flight.
    
    Args:
        times: ``(n,)`` array of times, or any shape broadcastable against the leading dimensions of ``velocity``.
        velocity: ``(..., 3)`` start velocity.
        acceleration: ``(3,)`` constant acceleration, defaults to gravity.

    Returns:
        ``(..., n, 3)`` array of velocities.
    """
    acceleration = GRAVITY if acceleration is None else acceleration
    t = np.asarray(times, dtype=float)[..., np.newaxis]
    return velocity + np.asarray(acceleration) * t


def quadratic_roots(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both roots of ``a x^2 + b x + c`` with ``nan`` where they are complex.

    The roots are returned in ascending order. ``a`` may contain zeros, in
    which case the single linear root is returned in both outputs.
    """
    a, b, c = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(c, float))
    discriminant = b**2 - 4.0 * a * c

    # Use np.errstate to ignore invalid and divide warnings, since we handle those cases explicitly.
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.sqrt(np.where(discriminant >= 0.0, discriminant, np.nan))
        linear = np.where(np.abs(b) > 1e-12, -c / np.where(np.abs(b) > 1e-12, b, 1.0), np.nan)
        first = np.where(np.abs(a) > 1e-12, (-b - root) / (2.0 * a), linear)
        second = np.where(np.abs(a) > 1e-12, (-b + root) / (2.0 * a), linear)
    return np.minimum(first, second), np.maximum(first, second)


def time_to_plane(position: np.ndarray, velocity: np.ndarray, height: float, acceleration: np.ndarray | None = None, minimum_time: float = 1e-6) -> np.ndarray:
    """Time until the point mass next crosses the horizontal plane ``z=height``.

    Returns ``inf`` where the plane is never reached again.

    Args:
        position: ``(..., 3)`` start position.
        velocity: ``(..., 3)`` start velocity.
        height: Height of the plane.
        acceleration: ``(3,)`` constant acceleration, defaults to gravity.
        minimum_time: Ignore roots that are less than this time.
    
    Returns:
        ``(...)`` array of times until the plane is reached.
    """
    acceleration = GRAVITY if acceleration is None else acceleration
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    # Solve the quadratic equation for the time when the z-coordinate of the position equals the height of the plane.
    lower, upper = quadratic_roots(
        0.5 * np.asarray(acceleration)[2],
        velocity[..., 2],
        position[..., 2] - height,
    )

    # Filter out roots that are less than the minimum time or are NaN, and return the minimum of the remaining roots.
    candidates = np.stack([lower, upper], axis=-1)
    candidates = np.where(candidates > minimum_time, candidates, np.inf)
    candidates = np.where(np.isnan(candidates), np.inf, candidates)
    return np.min(candidates, axis=-1)


def velocity_for_flight_time(start: np.ndarray, goal: np.ndarray, flight_time: np.ndarray, acceleration: np.ndarray | None = None) -> np.ndarray:
    """Launch velocity that moves ``start`` to ``goal`` in ``flight_time``.

    Args:
        start: ``(..., 3)`` start position.
        goal: ``(..., 3)`` goal position.
        flight_time: ``(...)`` time to reach the goal.
        acceleration: ``(3,)`` constant acceleration, defaults to gravity.
    
    Returns:
        ``(..., 3)`` launch velocity.
    """
    acceleration = GRAVITY if acceleration is None else acceleration
    t = np.asarray(flight_time, dtype=float)[..., np.newaxis]
    return (np.asarray(goal, float) - np.asarray(start, float)) / t - 0.5 * np.asarray(acceleration) * t


def bounce_velocity(velocity: np.ndarray, restitution: float, tangential: float = 1.0) -> np.ndarray:
    """Velocity after a bounce off a horizontal surface.

    Args:
        velocity: ``(..., 3)`` incoming velocity.
        restitution: Coefficient of restitution (0 = inelastic, 1 = elastic).
        tangential: Damping factor for the tangential velocity (0 = full damping, 1 = no damping).
    
    Returns:
        ``(..., 3)`` outgoing velocity.
    """
    out = np.asarray(velocity, dtype=float).copy()
    out[..., :2] *= tangential
    out[..., 2] = -restitution * out[..., 2]
    return out


def reflect_off_paddle(incoming: np.ndarray, normal: np.ndarray, paddle_velocity: np.ndarray, restitution: float, tangential: float = 1.0) -> np.ndarray:
    r"""Outgoing ball velocity of a paddle impact.

    In the frame of the paddle the normal component of the relative velocity is
    reversed and scaled by the restitution while the tangential component is
    damped by ``tangential``:

    .. math:: v_\text{out} = u + \mu\,w_t - \varepsilon\, w_n,
              \qquad w = v_\text{in} - u.

    Args:
        incoming: ``(..., 3)`` incoming ball velocity.
        normal: ``(..., 3)`` unit normal of the paddle surface.
        paddle_velocity: ``(..., 3)`` velocity of the paddle.
        restitution: Coefficient of restitution (0 = inelastic, 1 = elastic).
        tangential: Damping factor for the tangential velocity (0 = full damping, 1 = no damping).
    
    Returns:
        ``(..., 3)`` outgoing ball velocity.
    """
    incoming = np.asarray(incoming, float)
    normal = np.asarray(normal, float)
    paddle_velocity = np.asarray(paddle_velocity, float)

    relative = incoming - paddle_velocity
    relative_normal = np.sum(relative * normal, axis=-1, keepdims=True) * normal
    relative_tangential = relative - relative_normal
    return paddle_velocity + tangential * relative_tangential - restitution * relative_normal


def paddle_impact_inverse(incoming: np.ndarray, outgoing: np.ndarray, restitution: float, tangential: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    r"""Paddle normal and normal speed that turn ``incoming`` into ``outgoing``.

    Restricting the paddle velocity to :math:`u = u_n\,\hat n` (the paddle only
    has to move along its own normal, which is the cheapest thing to ask of the
    arm) the forward model collapses to

    .. math:: v_\text{out} = \mu\, v_\text{in}
              + \big[(1+\varepsilon) u_n - (\mu+\varepsilon)(v_\text{in}\!\cdot\!\hat n)\big]\hat n,

    because the *tangential* relative velocity does not depend on :math:`u_n`.
    So :math:`v_\text{out} - \mu v_\text{in}` is parallel to the normal, which
    inverts in closed form -- friction and all.  For :math:`\mu = 1` this
    reduces to the familiar "normal is along the velocity change" rule.

    Args:
        incoming: ``(..., 3)`` incoming ball velocity.
        outgoing: ``(..., 3)`` outgoing ball velocity.
        restitution: Coefficient of restitution (0 = inelastic, 1 = elastic).
        tangential: Damping factor for the tangential velocity (0 = full damping, 1 = no damping).

    Returns:
        ``(normal, normal_speed)`` with ``normal`` of unit length.
    """
    incoming = np.asarray(incoming, float)
    outgoing = np.asarray(outgoing, float)

    direction = outgoing - tangential * incoming
    magnitude = np.linalg.norm(direction, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normal = direction / magnitude

    incoming_normal = np.sum(incoming * normal, axis=-1)
    speed = (magnitude[..., 0] + (tangential + restitution) * incoming_normal) / (1.0 + restitution)
    return normal, speed
