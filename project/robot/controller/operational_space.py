import mujoco

import numpy as np

from .controller import ControllerBase
from .utils import calc_orientation_error

mj_forward = getattr(mujoco, "mj_forward")
mj_jacSite = getattr(mujoco, "mj_jacSite")
mj_fullM = getattr(mujoco, "mj_fullM")

class OperationalSpaceController(ControllerBase):
    """Operational Space controller class for controlling multiple joints in task space (position and orientation)."""

    def __init__(self, model, data, joint_names, target_site_name, u_max=100.0, dt=0.005) -> None:
        """Constructor method for the OperationalSpaceController class.

        Args:
            model (MjModel): The MuJoCo model object.
            data (MjData): The MuJoCo data object.
            joint_names (list): List of joint names to be controlled.
            target_site_name (str): Name of the target site for control.
            u_max (float or np.ndarray, optional): Maximum control signal output. Can be a single float for uniform limits or an array for individual joint limits. Defaults to 100.0.
            dt (float, optional): Time step for the controller updates. Defaults to 0.005.
        """
        super().__init__(model, data, joint_names, target_site_name, u_max, dt)

        # Set default gains for position and orientation control
        self.position_kp = np.array([80.0, 80.0, 80.0], dtype=float)
        self.position_kd = np.array([22.0, 22.0, 22.0], dtype=float)

        self.orientation_kp = np.array([70.0, 70.0, 70.0], dtype=float)
        self.orientation_kd = np.array([18.0, 18.0, 18.0], dtype=float)

        mj_forward(self.model, self.data)

        # Initialize target position and rotation
        self.target_position = self._get_site_position()
        self.target_rotation = self._get_site_rotation()


    def _get_site_rotation(self) -> np.ndarray:
        """Get the current rotation of the target site as a 3x3 rotation matrix."""
        rotation = self.data.site_xmat[self.target_site_id].reshape(3, 3)
        return rotation
    

    def _get_site_position(self) -> np.ndarray:
        """Get the current position of the target site."""
        return self.data.site_xpos[self.target_site_id]


    def get_site_pose(self) -> dict:
        """Get the current pose (position and rotation) of the target site.

        Returns:
            dict: A dictionary containing the position and rotation of the target site.
                  - "position": A numpy array representing the position (x, y, z).
                  - "rotation": A 3x3 numpy array representing the rotation matrix.
        """
        return {
            "position": self.data.site_xpos[self.target_site_id].copy(),
            "rotation": self._get_site_rotation(),
        }

    def set_gains(self, position_kp=None, position_kd=None, orientation_kp=None, orientation_kd=None) -> None:
        """Set the gains for position and orientation control.

        Args:
            position_kp (list or np.ndarray, optional): Proportional gains for position control. Defaults to None.
            position_kd (list or np.ndarray, optional): Derivative gains for position control. Defaults to None.
            orientation_kp (list or np.ndarray, optional): Proportional gains for orientation control. Defaults to None.
            orientation_kd (list or np.ndarray, optional): Derivative gains for orientation control. Defaults to None.
        """
        if position_kp is not None:
            self.position_kp = np.array(position_kp, dtype=float)
        if position_kd is not None:
            self.position_kd = np.array(position_kd, dtype=float)
        if orientation_kp is not None:
            self.orientation_kp = np.array(orientation_kp, dtype=float)
        if orientation_kd is not None:
            self.orientation_kd = np.array(orientation_kd, dtype=float)
    

    def set_target_pose(self, position, rotation) -> None:
        """Set the target pose (position and rotation) for the controller.

        Args:
            position (list or np.ndarray): Target position as a 3D vector (x, y, z).
            rotation (list or np.ndarray): Target rotation as a 3x3 rotation matrix.
        """
        self.set_target_position(position)
        self.set_target_rotation(rotation)


    def set_target_position(self, position) -> None:
        """Set the target position for the controller.

        Args:
            position (list or np.ndarray): Target position as a 3D vector (x, y, z).
        """
        position = np.asarray(position, dtype=float)
        if position.shape != (3,):
            raise ValueError("Target position must be a 3D vector.")
        self.target_position = np.array(position, dtype=float)
    

    def set_target_rotation(self, rotation) -> None:
        """Set the target rotation for the controller.

        Args:
            rotation (list or np.ndarray): Target rotation as a 3x3 rotation matrix.
        """
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("Target rotation must be a 3x3 matrix.")
        self.target_rotation = rotation
    
    
    def _compute_site_jacobians(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Jacobian matrices for the target site."""
        jacobian_position = np.zeros((3, self.model.nv), dtype=float)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=float)

        # Compute the Jacobian for the target site using MuJoCo's mj_jacSite function
        mj_jacSite(self.model, self.data, jacobian_position, jacobian_rotation, self.target_site_id)

        # Reduce the Jacobians to only include the controlled joints
        jacobian_position = jacobian_position[:, self.dof_ids]
        jacobian_rotation = jacobian_rotation[:, self.dof_ids]

        return jacobian_position, jacobian_rotation
    

    def _compute_mass_matrix(self) -> np.ndarray:
        """Compute the mass matrix for the system."""
        mass_matrix = np.zeros((self.model.nv, self.model.nv), dtype=float)
        
        # Compute the mass matrix using MuJoCo's mj_fullM function
        mj_fullM(self.model, mass_matrix, self.data.qM)

        # Reduce the mass matrix to only include the controlled joints
        mass_matrix = mass_matrix[np.ix_(self.dof_ids, self.dof_ids)]
        return mass_matrix
    

    def update(self):
        """Update the controller and compute the control signal based on the current state and target pose.
        
        Returns:
            dict: A dictionary containing the position and orientation errors, and the computed joint torques.
                  - "position_error": A numpy array representing the position error (x, y, z).
                  - "orientation_error": A numpy array representing the orientation error (roll, pitch, yaw).
                  - "joint_torques": A numpy array representing the computed joint torques for the controlled joints.
        """
        # Calculate position and orientation errors
        current_position = self._get_site_position()
        pos_error = self.target_position - current_position

        current_rotation = self._get_site_rotation()
        rot_error = calc_orientation_error(current_rotation, self.target_rotation, return_as_rotvec=True)

        # Calculate task error and their jacobians
        task_error = np.concatenate([pos_error, rot_error])
        task_jacobians = np.vstack(self._compute_site_jacobians())

        # Compute the current task space velocity
        velocity = self.data.qvel[self.dof_ids]
        task_velocity = task_jacobians @ velocity

        # Compute the mass matrix and its inverse
        mass_matrix = self._compute_mass_matrix()
        mass_matrix_inverse = np.linalg.inv(mass_matrix)

        # Compute the operational space inertia matrix
        operational_inertia_inverse = task_jacobians @ mass_matrix_inverse @ task_jacobians.T
        operational_inertia_inverse += 1e-6 * np.eye(task_jacobians.shape[0])  # Regularization for numerical stability
        operational_inertia = np.linalg.pinv(operational_inertia_inverse)

        # Concatenate task stiffness and damping
        task_stiffness = np.concatenate([self.position_kp, self.orientation_kp])
        task_damping = np.concatenate([self.position_kd, self.orientation_kd])
        
        # Compute task acceleration based on stiffness, damping, and errors (basic PD control)
        task_acceleration = task_stiffness * task_error - task_damping * task_velocity

        # Compute force in operational space and map it to joint torques
        task_force = operational_inertia @ task_acceleration
        joint_torque = task_jacobians.T @ task_force

        # Cancel out the Coriolis and gravitational forces to get the final control signal
        joint_torque += self.data.qfrc_bias[self.dof_ids]

        self.apply_control(joint_torque)

        # Return the position and orientation errors and torques for monitoring or logging purposes

        return {
            "position_error": pos_error,
            "orientation_error": rot_error,
            "joint_torques": joint_torque,
        }
