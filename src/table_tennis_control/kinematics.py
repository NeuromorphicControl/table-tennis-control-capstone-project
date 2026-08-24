"""Small rotation helpers shared by the planner and the controller."""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalise",
    "skew",
    "rotation_aligning",
    "orientation_from_normal",
    "axis_alignment_error",
    "slerp_axis",
    "point_in_box",
]


def normalise(vector: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Return ``vector`` scaled to unit length along ``axis``.
    
    Args:
        vector: ``(..., 3)`` vector to normalise.
        axis: Axis along which to normalise.
        eps: Minimum norm to avoid division by zero.
    
    Returns:
        ``(..., 3)`` normalised vector.
    """
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector, axis=axis, keepdims=True)
    return vector / np.maximum(norm, eps)


def skew(vector: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix ``[v]_x`` such that ``[v]_x w == cross(v, w)``.
    
    Args:
        vector: ``(..., 3)`` vector to convert to skew-symmetric matrix.
    
    Returns:
        ``(..., 3, 3)`` skew-symmetric matrix.
    """
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_aligning(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix that maps the unit vector ``source`` onto ``target``.

    Uses Rodrigues' formula on the rotation that sweeps the shortest arc
    between the two directions; the anti-parallel case is handled explicitly.

    Args:
        source: ``(3,)`` unit vector to rotate from.
        target: ``(3,)`` unit vector to rotate to.
    
    Returns:
        ``(3, 3)`` rotation matrix that maps ``source`` to ``target``.
    """
    source = normalise(source)
    target = normalise(target)

    axis = np.cross(source, target)
    sin_angle = float(np.linalg.norm(axis))
    cos_angle = float(np.clip(np.dot(source, target), -1.0, 1.0))

    # Handle the parallel and anti-parallel cases explicitly to avoid numerical issues.
    if sin_angle < 1e-9:
        if cos_angle > 0.0:
            # parallel: no rotation needed
            return np.eye(3)

        # Anti-parallel: rotate by pi about any axis perpendicular to source.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, source))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])

        axis = normalise(np.cross(source, helper))
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    # Rodrigues' formula for the rotation matrix.
    axis = axis / sin_angle
    cross = skew(axis)
    angle = float(np.arctan2(sin_angle, cos_angle))
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def orientation_from_normal(normal: np.ndarray, reference_rotation: np.ndarray, normal_axis: int = 1) -> np.ndarray:
    """Rotation matrix whose ``normal_axis`` column points along ``normal``.

    The remaining degree of freedom (the spin of the paddle about its own
    normal) is resolved by staying as close as possible to
    ``reference_rotation``, which makes the command deterministic and free of
    the drift a "rotate from the current pose" rule would accumulate.

    Args:
        normal: ``(..., 3)`` unit vector to align the paddle normal with.
        reference_rotation: ``(..., 3, 3)`` rotation matrix to stay close to.
        normal_axis: Axis of the paddle rotation matrix that should point along ``normal``.
    
    Returns:
        ``(..., 3, 3)`` rotation matrix with the specified normal.
    """
    reference_normal = reference_rotation[:, normal_axis]
    return rotation_aligning(reference_normal, normal) @ reference_rotation


def axis_alignment_error(current_rotation: np.ndarray, desired_normal: np.ndarray, normal_axis: int = 1) -> np.ndarray:
    """Rotation vector that aligns the paddle normal with ``desired_normal``.

    The result has no component along the paddle normal: the spin about the
    face is deliberately left uncontrolled so that the redundancy is available
    to the null-space tasks.

    Args:
        current_rotation: ``(..., 3, 3)`` current paddle rotation matrix.
        desired_normal: ``(..., 3)`` unit vector to align the paddle normal with.
        normal_axis: Axis of the paddle rotation matrix that should point along ``desired_normal``.
    
    Returns:
        ``(..., 3)`` rotation vector that aligns the paddle normal with ``desired_normal``.
    """
    current_normal = normalise(current_rotation[:, normal_axis])
    desired_normal = normalise(desired_normal)

    axis = np.cross(current_normal, desired_normal)
    sin_angle = float(np.linalg.norm(axis))
    cos_angle = float(np.clip(np.dot(current_normal, desired_normal), -1.0, 1.0))

    # Handle the parallel and anti-parallel cases explicitly to avoid numerical issues.
    if sin_angle < 1e-9:
        if cos_angle > 0.0:
            # parallel: no rotation needed
            return np.zeros(3)

        # Anti-parallel: rotate by pi about any axis perpendicular to current_normal.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, current_normal))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        
        return np.pi * normalise(np.cross(current_normal, helper))

    return (axis / sin_angle) * float(np.arctan2(sin_angle, cos_angle))


def slerp_axis(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Spherical interpolation between two unit direction vectors.
    
    Args:
        start: ``(3,)`` unit vector to interpolate from.
        end: ``(3,)`` unit vector to interpolate to.
        fraction: Interpolation fraction in [0, 1].
        
    Returns:
        ``(3,)`` unit vector interpolated between ``start`` and ``end``.
    """
    start = normalise(start)
    end = normalise(end)
    
    cos_angle = float(np.clip(np.dot(start, end), -1.0, 1.0))

    angle = float(np.arccos(cos_angle))
    if angle < 1e-8:
        return end

    sin_angle = np.sin(angle)

    # Use spherical linear interpolation formula to compute the interpolated vector
    a = np.sin((1.0 - fraction) * angle) / sin_angle
    b = np.sin(fraction * angle) / sin_angle
    return normalise(a * start + b * end)


def point_in_box(points: np.ndarray, bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> np.ndarray:
    """Boolean mask of the ``(..., 3)`` points that lie inside an axis-aligned box.

    Args:
        points: Array of shape ``(..., 3)``.
        bounds: ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``.

    Returns:
        Boolean mask of shape ``(...)``.
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
