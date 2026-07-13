import mujoco

import numpy as np

MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")

class Target:
    """A class representing a target in the simulation."""
    def __init__(self, model, data, target_site_name="target"):
        """
        Initialize the Target instance.

        Args:
            model (MjModel): The MuJoCo model.
            data (MjData): The MuJoCo data.
            target_site_name (str): The name of the target site.
        """
        self.model = model
        self.data = data

        self.target_site_id = model.site(target_site_name).id

    
    def get_position(self) -> np.ndarray:
        """Get the current position of the ball.

        Returns:
            np.ndarray: The current position of the ball as a 3D coordinate (x, y, z).
        """
        return self.data.site_xpos[self.target_site_id]


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
        
        self.data.site_xpos[self.target_site_id] = position