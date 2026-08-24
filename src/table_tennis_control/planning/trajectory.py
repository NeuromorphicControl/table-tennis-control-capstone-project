"""Smooth task-space reference trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..kinematics import normalise, slerp_axis

__all__ = ["TaskState", "QuinticSegment", "SwingTrajectory", "GoToTrajectory"]


@dataclass
class TaskState:
    """A full reference for the end effector at one instant."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    normal: np.ndarray
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))


class QuinticSegment:
    """Minimum-jerk polynomial between two boundary states.

    Solves for the quintic that matches position, velocity and acceleration at both ends.
    """

    def __init__(self, start_position: np.ndarray, start_velocity: np.ndarray, start_acceleration: np.ndarray, end_position: np.ndarray, end_velocity: np.ndarray, end_acceleration: np.ndarray, duration: float):
        """Construct a quintic segment from boundary conditions.
        
        Args:
            start_position: The starting position of the segment.
            start_velocity: The starting velocity of the segment.
            start_acceleration: The starting acceleration of the segment.
            end_position: The ending position of the segment.
            end_velocity: The ending velocity of the segment.
            end_acceleration: The ending acceleration of the segment.
            duration: The duration of the segment in seconds.
        """
        self.duration = max(float(duration), 1e-3)
        T = self.duration

        p0 = np.asarray(start_position, dtype=float)
        v0 = np.asarray(start_velocity, dtype=float)
        a0 = np.asarray(start_acceleration, dtype=float)
        p1 = np.asarray(end_position, dtype=float)
        v1 = np.asarray(end_velocity, dtype=float)
        a1 = np.asarray(end_acceleration, dtype=float)

        # Compute the coefficients of the quintic polynomial that satisfies the boundary conditions
        delta = p1 - p0
        self.c = np.stack([
            p0,
            v0,
            0.5 * a0,
            (20.0 * delta - (8.0 * v1 + 12.0 * v0) * T - (3.0 * a0 - a1) * T**2) / (2.0 * T**3),
            (-30.0 * delta + (14.0 * v1 + 16.0 * v0) * T + (3.0 * a0 - 2.0 * a1) * T**2) / (2.0 * T**4),
            (12.0 * delta - 6.0 * (v1 + v0) * T - (a0 - a1) * T**2) / (2.0 * T**5),
        ])

    def evaluate(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position, velocity and acceleration at ``time`` (clamped to the segment).
        
        Args:
            time: The time at which to evaluate the segment, in seconds.
        
        Returns:
            A tuple containing the position, velocity and acceleration at the given time.
        """
        t = float(np.clip(time, 0.0, self.duration))

        # Compute powers of t for the polynomial evaluation
        powers = np.array([1.0, t, t**2, t**3, t**4, t**5])
        d_powers = np.array([0.0, 1.0, 2.0 * t, 3.0 * t**2, 4.0 * t**3, 5.0 * t**4])
        dd_powers = np.array([0.0, 0.0, 2.0, 6.0 * t, 12.0 * t**2, 20.0 * t**3])

        # Compute position, velocity and acceleration using the polynomial coefficients
        return powers @ self.c, d_powers @ self.c, dd_powers @ self.c


class _OrientationSchedule:
    """Rotates the paddle normal from ``start`` to ``end`` about a single fixed axis.

    Boundary-matched to the *incoming* angular velocity/acceleration exactly
    the way :class:`QuinticSegment` matches incoming linear velocity/
    acceleration for position.
    """

    def __init__(self, start_normal: np.ndarray, end_normal: np.ndarray, duration: float, start_angular_velocity: np.ndarray | None = None, start_angular_acceleration: np.ndarray | None = None):
        """Construct a single-axis rotation schedule from boundary conditions.
        
        Args:
            start_normal: The starting normal vector of the paddle.
            end_normal: The ending normal vector of the paddle.
            duration: The duration of the rotation in seconds.
            start_angular_velocity: The starting angular velocity of the paddle (optional).
            start_angular_acceleration: The starting angular acceleration of the paddle (optional).
        """
        self.start = normalise(start_normal)
        self.end = normalise(end_normal)
        self.duration = max(float(duration), 1e-3)

        # Compute the rotation axis and angle between the start and end normals
        cross = np.cross(self.start, self.end)
        sin_angle = float(np.linalg.norm(cross))
        cos_angle = float(np.clip(np.dot(self.start, self.end), -1.0, 1.0))
        self._angle = float(np.arctan2(sin_angle, cos_angle))

        # Handle special cases for the rotation axis when the angle is very small or when the normals are antiparallel
        if sin_angle < 1e-9:
            if cos_angle > 0.0:
                # Start and end are the same: no rotation needed, axis is arbitrary
                self._axis = np.zeros(3)
            else:
                # Antiparallel: choose an arbitrary axis perpendicular to the start normal
                helper = np.array([1.0, 0.0, 0.0])
                if abs(float(np.dot(helper, self.start))) > 0.9:
                    helper = np.array([0.0, 1.0, 0.0])
                self._axis = normalise(np.cross(self.start, helper))
        else:
            self._axis = cross / sin_angle

        angular_velocity = np.zeros(3) if start_angular_velocity is None else np.asarray(start_angular_velocity, dtype=float)
        angular_acceleration = np.zeros(3) if start_angular_acceleration is None else np.asarray(start_angular_acceleration, dtype=float)

        # Project the incoming angular velocity and acceleration onto the rotation axis to get the effective initial conditions for the single-axis rotation
        omega0 = float(np.dot(angular_velocity, self._axis))
        alpha0 = float(np.dot(angular_acceleration, self._axis))

        # Create a quintic segment for the rotation angle that matches the initial angular velocity and acceleration, and ends at the target angle with zero velocity and acceleration
        self._theta = QuinticSegment(
            np.array([0.0]),
            np.array([omega0]),
            np.array([alpha0]),
            np.array([self._angle]),
            np.array([0.0]),
            np.array([0.0]),
            self.duration,
        )

    def _theta_at(self, time: float) -> tuple[float, float, float]:
        """Angle swept, its rate and its acceleration, clamped to ``[0, self._angle]``."""
        theta, theta_dot, theta_ddot = self._theta.evaluate(time)
        theta = float(theta[0])

        # Clamp the angle and its derivatives to ensure they stay within the valid range of rotation
        if theta <= 0.0:
            return 0.0, 0.0, 0.0
        if theta >= self._angle:
            return self._angle, 0.0, 0.0

        # Return the angle, angular velocity, and angular acceleration along the rotation axis
        return theta, float(theta_dot[0]), float(theta_ddot[0])

    def normal(self, time: float) -> np.ndarray:
        """Paddle normal at ``time``."""
        if self._angle < 1e-9:
            return self.end
        
        # Get the angle swept at the given time and compute the corresponding normal using spherical linear interpolation (slerp) between the start and end normals
        theta, _, _ = self._theta_at(time)
        return slerp_axis(self.start, self.end, theta / self._angle)

    def evaluate(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normal, angular velocity and angular acceleration at ``time``."""
        if self._angle < 1e-9:
            return self.end, np.zeros(3), np.zeros(3)

        # Get the angle swept, angular velocity, and angular acceleration at the given time
        theta, theta_dot, theta_ddot = self._theta_at(time)
        normal = slerp_axis(self.start, self.end, theta / self._angle)

        # Return the normal, angular velocity, and angular acceleration along the rotation axis
        return normal, theta_dot * self._axis, theta_ddot * self._axis


class GoToTrajectory:
    """Move to a pose and stay there (used for the ready/idle posture)."""

    def __init__(self, start: TaskState, goal_position: np.ndarray, goal_normal: np.ndarray, duration: float, start_time: float = 0.0):
        """Construct a trajectory that moves from the start state to the goal position and normal over the specified duration.
        
        Args:
            start: The starting state of the trajectory.
            goal_position: The target position to reach.
            goal_normal: The target normal to reach.
            duration: The duration of the trajectory in seconds.
            start_time: The time at which the trajectory starts (default is 0.0).
        """
        self.start_time = float(start_time)
        self.duration = max(float(duration), 1e-3)

        self._segment = QuinticSegment(
            start.position,
            start.velocity,
            start.acceleration,
            np.asarray(goal_position, dtype=float),
            np.zeros(3),
            np.zeros(3),
            self.duration,
        )

        self._orientation = _OrientationSchedule(
            start.normal, 
            goal_normal, 
            self.duration, 
            start.angular_velocity, 
            start.angular_acceleration,
        )

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def evaluate(self, time: float) -> TaskState:
        """Position, velocity, acceleration, normal, angular velocity and angular acceleration at ``time``."""
        local = float(np.clip(time - self.start_time, 0.0, self.duration))

        # Evaluate the position, velocity, and acceleration from the quintic segment, and the normal, angular velocity, and angular acceleration from the orientation schedule at the local time
        position, velocity, acceleration = self._segment.evaluate(local)
        normal, angular_velocity, angular_acceleration = self._orientation.evaluate(local)

        # If the time exceeds the duration of the trajectory, set the velocity, acceleration, angular velocity, and angular acceleration to zero
        if local >= self.duration:
            velocity = np.zeros(3)
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)
            angular_acceleration = np.zeros(3)
        return TaskState(position, velocity, acceleration, normal, angular_velocity, angular_acceleration)


class SwingTrajectory:
    """Approach -- impact -- follow-through.

    The approach segment ends *with the required paddle velocity*, so the
    paddle is already swinging when it meets the ball; the follow-through
    segment decelerates smoothly instead of stopping dead on impact.
    """

    def __init__(self, start: TaskState, start_time: float, impact_time: float, impact_position: np.ndarray, impact_velocity: np.ndarray, impact_normal: np.ndarray, follow_through_time: float = 0.18, follow_through_distance: float = 0.1, orientation_lead: float = 0.75):
        """Construct a swing trajectory from boundary conditions.
        
        Args:
            start: The starting state of the trajectory.
            start_time: The time at which the approach segment starts.
            impact_time: The time at which the impact occurs.
            impact_position: The position of the paddle at impact.
            impact_velocity: The velocity of the paddle at impact.
            impact_normal: The normal of the paddle at impact.
            follow_through_time: The duration of the follow-through segment after impact (default is 0.18 seconds).
            follow_through_distance: The distance to travel during the follow-through segment (default is 0.1 meters).
            orientation_lead: The fraction of the approach duration to complete the orientation change before impact (default is 0.75).
        """
        self.start_time = float(start_time)
        self.impact_time = float(impact_time)
        self.approach_duration = max(self.impact_time - self.start_time, 1e-3)
        self.follow_through_time = max(float(follow_through_time), 1e-3)

        impact_position = np.asarray(impact_position, dtype=float)
        impact_velocity = np.asarray(impact_velocity, dtype=float)

        self.impact_position = impact_position
        self.impact_velocity = impact_velocity
        self.impact_normal = normalise(impact_normal)

        self._approach = QuinticSegment(
            start.position,
            start.velocity,
            start.acceleration,
            impact_position,
            impact_velocity,
            np.zeros(3),
            self.approach_duration,
        )

        speed = float(np.linalg.norm(impact_velocity))
        direction = impact_velocity / speed if speed > 1e-6 else np.zeros(3)

        self._follow_through = QuinticSegment(
            impact_position,
            impact_velocity,
            np.zeros(3),
            impact_position + direction * float(follow_through_distance),
            np.zeros(3),
            np.zeros(3),
            self.follow_through_time,
        )

        # Be done turning the paddle before the ball arrives.
        self._orientation = _OrientationSchedule(
            start.normal,
            self.impact_normal,
            max(self.approach_duration * orientation_lead, 1e-3),
            start.angular_velocity,
            start.angular_acceleration,
        )

    @property
    def end_time(self) -> float:
        return self.impact_time + self.follow_through_time

    def evaluate(self, time: float) -> TaskState:
        """Position, velocity, acceleration, normal, angular velocity and angular acceleration at ``time``."""
        angular_acceleration = np.zeros(3)


        if time <= self.impact_time:
            local = float(np.clip(time - self.start_time, 0.0, self.approach_duration))

            # Evaluate the position, velocity, and acceleration from the approach segment, and the normal, angular velocity, and angular acceleration from the orientation schedule at the local time
            position, velocity, acceleration = self._approach.evaluate(local)
            normal, angular_velocity, angular_acceleration = self._orientation.evaluate(local)
        else:
            local = float(np.clip(time - self.impact_time, 0.0, self.follow_through_time))

            # Evaluate the position, velocity, and acceleration from the follow-through segment, and use the impact normal and zero angular velocity for the follow-through phase
            position, velocity, acceleration = self._follow_through.evaluate(local)
            normal, angular_velocity = self.impact_normal, np.zeros(3)

            # If the time exceeds the follow-through duration, set the velocity and acceleration to zero
            if local >= self.follow_through_time:
                velocity = np.zeros(3)
                acceleration = np.zeros(3)

        # Return the task state at the given time, including position, velocity, acceleration, normal, angular velocity, and angular acceleration
        return TaskState(position, velocity, acceleration, normal, angular_velocity, angular_acceleration)
