import numpy as np

from .physics import calc_time_to_floor, predict_position, predict_velocity


class TrajectoryPredictor:
    def __init__(self, ball, dt=0.05, t_max=5, gravity_vector=None):
        """Initialize the trajectory predictor.
        
        Args:
            ball: An instance of the Ball class representing the ball in the simulation.
            dt (float): Time step for prediction.
            t_max (float): Maximum time for prediction (seconds).
            gravity_vector (np.ndarray | None): Gravitational acceleration vector. If None, uses standard gravity.
        """
        self.ball = ball

        self.dt = dt
        self.t_max = t_max

        self.gravity_vector = gravity_vector

        self._calculate_trajectory()


    def _calculate_trajectory(self):
        """Calculate the predicted trajectory of the ball based on its current state."""
        time_to_floor = calc_time_to_floor(self.ball.get_position(), self.ball.get_velocity(), self.gravity_vector)
        self.times = np.arange(0, min(self.t_max, time_to_floor), self.dt)

        self.positions = predict_position(self.times, self.ball.get_position(), self.ball.get_velocity(), self.gravity_vector)
        self.velocities = predict_velocity(self.times, self.ball.get_velocity(), self.gravity_vector)


    def get_trajectory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the predicted trajectory of the ball.
        
        Returns:
            tuple: A tuple containing the time steps, predicted positions, and predicted velocities of the ball.
        """
        return self.times, self.positions, self.velocities

    def update(self):
        """Update the predicted trajectory based on the current state of the ball."""
        self._calculate_trajectory()
