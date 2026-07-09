import mujoco

import numpy as np

MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")

class Ball:
    """A class representing a ball in the simulation."""
    def __init__(self, model, data, ball_joint_name="ball_free"):
        """
        Initialize the Ball instance.

        Args:
            model (MjModel): The MuJoCo model.
            data (MjData): The MuJoCo data.
            ball_joint_name (str): The name of the ball's free joint.
            start_pos (tuple): The initial position of the ball.
        """
        self.model = model
        self.data = data

        self.ball_joint_id = model.joint(ball_joint_name).id

        self.last_position = self.get_position()
        self.velocity = self._calculate_velocity()

    
    def get_position(self) -> np.ndarray:
        """Get the current position of the ball.

        Returns:
            np.ndarray: The current position of the ball as a 3D coordinate (x, y, z).
        """
        return self.data.qpos[self.ball_joint_id:self.ball_joint_id + 3]


    def get_velocity(self) -> np.ndarray:
        """Get the current velocity of the ball.

        Returns:
            np.ndarray: The current velocity of the ball as a 3D vector (vx, vy, vz).
        """
        return self.velocity
    

    def set_position(self, position: tuple | np.ndarray) -> None:
        """
        Set the position of the ball.

        Args:
            position (tuple | np.ndarray): The new position of the ball.

        Raises:
            ValueError: If the position is not a 3D coordinate.
        """
        position = np.array(position)  # Ensure position is a numpy array
        if position.shape != (3,):
            raise ValueError("Position must be a 3D coordinate (x, y, z).")
        
        self.data.qpos[self.ball_joint_id:self.ball_joint_id + 3] = position

    
    def set_velocity(self, velocity: tuple | np.ndarray) -> None:
        """
        Set the velocity of the ball.

        Args:
            velocity (tuple | np.ndarray): The new velocity of the ball.

        Raises:
            ValueError: If the velocity is not a 3D vector.
        """
        velocity = np.array(velocity)  # Ensure velocity is a numpy array
        if velocity.shape != (3,):
            raise ValueError("Velocity must be a 3D vector (vx, vy, vz).")
        
        self.data.qvel[self.ball_joint_id:self.ball_joint_id + 3] = velocity
    

    def reset_position(self, position: tuple | np.ndarray) -> None:
        """
        Reset the position of the ball to a specified position and reset its velocity.

        Args:
            position (tuple | np.ndarray): The position to reset the ball to.
        
        Raises:
            ValueError: If the position is not a 3D coordinate.
        """
        self.last_position = np.array(position)

        self.set_position(position)
        self.set_velocity((0.0, 0.0, 0.0))
    

    def _calculate_velocity(self, dt: float | None = None) -> np.ndarray:
        """Calculate the current velocity of the ball based on its position change over time.

        Args:
            dt (float | None): The time step over which to calculate the velocity. If None, the model's timestep will be used.

        Returns:
            np.ndarray: The current velocity of the ball as a 3D vector (vx, vy, vz).

        Raises:
            ValueError: If the time step (dt) is zero, which would lead to division by zero.
        """
        if dt is None:
            dt = self.model.opt.timestep  # Use the model's timestep if dt is not provided

        if dt == 0:
            raise ValueError("Time step (dt) cannot be zero.")

        current_position = self.get_position()
        velocity = (current_position - self.last_position) / dt
        self.last_position = current_position.copy()
        return velocity


    def update(self, dt: float | None = None) -> None:
        """Update the ball's state.

        Args:
            dt (float | None): The time step over which to update the ball's state. If None, the model's timestep will be used.
        """
        self.velocity = self._calculate_velocity(dt)