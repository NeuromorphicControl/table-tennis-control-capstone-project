import numpy as np


STANDARD_GRAVITY = np.array([0, 0, -9.81])


def calc_time_to_floor(start_position: np.ndarray, start_velocity: np.ndarray, gravity_vector: np.ndarray | None = None) -> float:
    """Calculate the time it takes for the ball to hit the floor (z=0) based on its initial position and velocity.

    Args:
        start_position (np.ndarray): Array of shape (3,) containing the initial position of the ball.
        start_velocity (np.ndarray): Array of shape (3,) containing the initial velocity of the ball.
        gravity_vector (np.ndarray | None): Array of shape (3,) containing the gravitational acceleration. If None, uses the standard gravity.

    Returns:
        float: The time it takes for the ball to hit the floor. Returns np.inf if the ball will not hit the floor.
    """
    if gravity_vector is None:
        gravity_vector = STANDARD_GRAVITY

    a = 0.5 * gravity_vector[2]
    b = start_velocity[2]
    c = start_position[2]

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return np.inf  # No real solution, ball will not hit the floor

    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    # Return the positive time value
    return max(t1, t2) if max(t1, t2) > 0 else np.inf


def predict_position(times: np.ndarray, start_positions: np.ndarray, start_velocities: np.ndarray, gravity_vector: np.ndarray | None = None) -> np.ndarray:
    """Calculate the position of the ball at each time step using the equations of motion.

    Args:
        times (np.ndarray): Array of time steps.
        start_positions (np.ndarray): Array of shape (n, 3) containing the initial positions of the ball at each time step.
        start_velocities (np.ndarray): Array of shape (n, 3) containing the initial velocities of the ball at each time step.
        gravity_vector (np.ndarray | None): Array of shape (3,) containing the gravitational acceleration. If None, uses the standard gravity.

    Returns:
        np.ndarray: Array of shape (n, 3) containing the positions of the ball at each time step.
    """
    if gravity_vector is None:
        gravity_vector = STANDARD_GRAVITY

    positions = start_positions + start_velocities * times[:, np.newaxis] + 0.5 * gravity_vector * times[:, np.newaxis]**2
    return positions


def predict_velocity(times: np.ndarray, start_velocities: np.ndarray, gravity_vector: np.ndarray | None = None) -> np.ndarray:
    """Calculate the velocity of the ball at each time step using the equations of motion.

    Args:
        times (np.ndarray): Array of time steps.
        start_velocities (np.ndarray): Array of shape (n, 3) containing the initial velocities of the ball at each time step.
        gravity_vector (np.ndarray | None): Array of shape (3,) containing the gravitational acceleration. If None, uses the standard gravity.

    Returns:
        np.ndarray: Array of shape (n, 3) containing the velocities of the ball at each time step.
    """
    if gravity_vector is None:
        gravity_vector = STANDARD_GRAVITY
    
    velocities = start_velocities + gravity_vector * times[:, np.newaxis]
    return velocities


def calc_paddle_normal(v_in: np.ndarray, v_out: np.ndarray) -> np.ndarray:
    """Calculate the normal vector of the paddle based on the incoming and outgoing velocities of the ball.

    Args:
        v_in (np.ndarray): Array of shape (3,) containing the incoming velocity of the ball.
        v_out (np.ndarray): Array of shape (3,) containing the outgoing velocity of the ball.

    Returns:
        np.ndarray: Array of shape (3,) containing the normal vector of the paddle.
    """
    n = v_out - v_in
    n /= np.linalg.norm(n)
    return n