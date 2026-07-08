import mujoco

import numpy as np

from .controller import ControllerBase

mj_forward = getattr(mujoco, "mj_forward")

class PIDController(ControllerBase):
    """Proportional-Integral-Derivative (PID) controller class for multiple joints."""

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

        self.kp = np.array([20.0, 20.0], dtype=float)
        self.ki = np.array([0.0, 0.0], dtype=float)
        self.kd = np.array([12.0, 12.0], dtype=float)

        self.e_last = np.zeros_like(self.kp)
        self.y_last = np.zeros_like(self.kp)

        self.e_integral = np.zeros_like(self.kp)
        self.e_derivative = np.zeros_like(self.kp)

        mj_forward(self.model, self.data)

        # Initialize target position and rotation
        self.target_position = self._get_site_position()

        # Calculate joint postion offset from initial target position
        self.joint_position_offset = self.data.qpos[self.qpos_ids] - self.target_position[:len(self.joint_ids)]


    def set_gains(self, kp=None, ki=None, kd=None) -> None:
        """Set the PID gains.

        Args:
            kp (list or np.ndarray, optional): Proportional gains for each joint. Defaults to None.
            ki (list or np.ndarray, optional): Integral gains for each joint. Defaults to None.
            kd (list or np.ndarray, optional): Derivative gains for each joint. Defaults to None.
        """
        if kp is not None:
            self.kp = np.array(kp, dtype=float)
        if ki is not None:
            self.ki = np.array(ki, dtype=float)
        if kd is not None:
            self.kd = np.array(kd, dtype=float)

    
    def set_target_position(self, position) -> None:
        """Set the target position for the PID controller.

        Args:
            position (list or np.ndarray): The desired target position for the controlled joints.
        """
        position = np.array(position, dtype=float)
        if position.shape != (len(self.target_position),):
            raise ValueError(f"Target position must have shape {self.target_position.shape}, but got {position.shape}.")
        self.target_position = position
    

    def _get_site_position(self) -> np.ndarray:
        """Get the current position of the target site.

        Returns:
            np.ndarray: A numpy array representing the position of the target site.
        """
        return self.data.site_xpos[self.target_site_id]
    

    def _compute_inverse_kinematics(self) -> np.ndarray:
        """Compute the inverse kinematics to translate target site positions to joint positions.

        Returns:
            np.ndarray: A numpy array representing the target joint positions.
        """
        # Placeholder for inverse kinematics computation
        # For now only translate the target position to slider positions (assuming a simple mapping)
        
        target_joint_positions = self.target_position[:len(self.joint_ids)] + self.joint_position_offset
        
        return target_joint_positions
    

    def update(self) -> dict:
        """Compute the control signal based on the current state and target.

        Returns:
            dict: A dictionary containing the position and orientation errors, and the computed joint torques.
                  - "position_error": A numpy array representing the position error (x, y, z).
                  - "joint_torques": A numpy array representing the computed joint torques for the controlled joints.
        """
        # Compute the target joint positions using inverse kinematics
        target_joint_positions = self._compute_inverse_kinematics()

        # Get current joint positions and velocities
        current_joint_positions = self.data.qpos[self.qpos_ids]
        current_joint_velocities = self.data.qvel[self.dof_ids]

        # Compute the position error
        position_error = target_joint_positions - current_joint_positions

        # Compute the derivative of the error (negative of current joint velocities)
        self.e_derivative = -current_joint_velocities

        # Update the integral of the error
        self.e_integral += position_error * self.dt

        # Compute the control signal using PID formula
        control_signal = (
            self.kp * position_error +
            self.ki * self.e_integral +
            self.kd * self.e_derivative
        )

        # Apply the computed control signal to the actuators
        self.apply_control(control_signal)

        return {
            "position_error": position_error,
            "joint_torques": control_signal,
        }