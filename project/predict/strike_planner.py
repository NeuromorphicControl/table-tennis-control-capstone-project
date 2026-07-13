import numpy as np

from .physics import STANDARD_GRAVITY

class StrikePlanner:
    """Class to plan the strike of a ball towards a target position based on its predicted trajectory and the working area of the robot arm."""
    
    def __init__(self, target, trajectory_predictor, working_area_bounds, gravity_vector=None):
        """Initialize the StrikePlanner with the target position, trajectory predictor, working area bounds, and optional gravity vector.
        
        Args:
            target (np.ndarray): Array of shape (3,) containing the target position for the ball.
            trajectory_predictor (TrajectoryPredictor): An instance of the TrajectoryPredictor class to predict the ball's trajectory.
            working_area_bounds (tuple): A tuple containing the bounds of the working area for the robot arm in the format ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
            gravity_vector (np.ndarray | None): Array of shape (3,) containing the gravitational acceleration. If None, uses the standard gravity.
        """

        self.target = target
        self.trajectory_predictor = trajectory_predictor
        self.gravity_vector = gravity_vector or STANDARD_GRAVITY
        self.working_area_bounds = working_area_bounds


    def get_possible_strike_positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the possible strike positions for the robot arm to intercept the ball.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuples containing the times, positions, and velocities of the possible strike positions.
        """
        # Get the predicted trajectory of the ball
        times, positions, velocities = self.trajectory_predictor.get_trajectory()

        # Reduce the number of positions to only those within the working area of the robot arm
        valid_positions_mask = (np.logical_and.reduce((
            positions[:, 0] >= self.working_area_bounds[0][0], positions[:, 0] <= self.working_area_bounds[0][1],
            positions[:, 1] >= self.working_area_bounds[1][0], positions[:, 1] <= self.working_area_bounds[1][1],
            positions[:, 2] >= self.working_area_bounds[2][0], positions[:, 2] <= self.working_area_bounds[2][1]
        )))

        return times[valid_positions_mask], positions[valid_positions_mask], velocities[valid_positions_mask]
    

    def calculate_flight_time(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """Calculate the flight time for the ball to reach the target position from each of the given positions and speeds.
        
        Args:
            positions (np.ndarray): Array of shape (n, 3) containing the positions of the ball at each time step.
            velocities (np.ndarray): Array of shape (n, 3) containing the velocities of the ball at each time step.
        """
        d = self.target - positions
        s = np.linalg.norm(velocities, axis=1)

        c = np.sum(d * d, axis=1)
        b = -(d @ self.gravity_vector + s**2)
        a = 0.25 * self.gravity_vector.dot(self.gravity_vector)

        discriminant = b**2 - 4*a*c

        # Calculate the two possible flight times using the quadratic formula
        flight_time1 = (-b + np.sqrt(discriminant)) / (2*a)
        flight_time2 = (-b - np.sqrt(discriminant)) / (2*a)

        # Replace negative flight times with NaN to indicate that they are not valid
        flight_times1 = np.where(flight_time1 > 0, flight_time1, np.nan)
        flight_times2 = np.where(flight_time2 > 0, flight_time2, np.nan)    

        # take the square root of the flight times to get the actual flight times
        flight_times1 = np.sqrt(flight_times1)
        flight_times2 = np.sqrt(flight_times2)    

        # Stack the two columns and return the result
        return np.column_stack((flight_times1, flight_times2))
    

    def calculate_strike_angle(self, flight_times: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Calculate the strike angle for the ball to reach the target position from each of the given positions and speeds.
        
        Args:
            flight_times (np.ndarray): Array of shape (n,) containing the selected flight times for each position and velocity.
            positions (np.ndarray): Array of shape (n, 3) containing the positions of the ball at each time step.
        """
        return (self.target - positions -0.5 * self.gravity_vector * flight_times**2) / flight_times
    

    def calculate_paddle_normal(self, strike_angles: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """Calculate the paddle normal for the ball to reach the target position from each of the given positions and speeds.
        
        Args:
            strike_angles (np.ndarray): Array of shape (n, 3) containing the strike angles for each position and velocity.
            velocities (np.ndarray): Array of shape (n, 3) containing the velocities of the ball at each time step.
        """
        diff = strike_angles - velocities
        return diff / np.linalg.norm(diff, axis=1, keepdims=True)
