import numpy as np

def calculate_optimal_target_position(positions, last_target):
    """
    Calculate the optimal target position for the robot arm to intercept the ball.

    Args:
        times (np.ndarray): Array of time steps.
        positions (np.ndarray): Array of shape (n, 3) containing the positions of the ball at each time step.
        velocities (np.ndarray): Array of shape (n, 3) containing the velocities of the ball at each time step.
        last_target (np.ndarray): The last target position of the robot arm.
    
    Returns:
        np.ndarray: The optimal target position for the robot arm to intercept the ball.
    """
    # Squared distances to all trajectory points
    d2 = np.sum((positions - last_target)**2, axis=1)

    # Find the index of the point with the minimum distance to the last target position
    idx = np.argmin(d2)

    return positions[idx]