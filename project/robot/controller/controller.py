import numpy as np

# Base Class for all controllers
class ControllerBase:
    """Base class for all controllers in the system.
    
    This class provides a common interface and shared functionality for different types of controllers.
    """

    def __init__(self, model, data, joint_names: list, target_name: str, u_max=100.0, dt=0.005) -> None:
        """Constructor method.
        
        Args:
            model (MjModel): The MuJoCo model object.
            data (MjData): The MuJoCo data object.
            joint_names (list): List of joint names to be controlled.
            target_name (str): Name of the target site for control.
            u_max (float or np.ndarray, optional): Maximum control signal output. Can be a single float for uniform limits or an array for individual joint limits. Defaults to 100.0.
            dt (float, optional): Time step for the controller updates. Defaults to 0.005.
        """

        self.model = model
        self.data = data

        self.joint_names = list(joint_names)
        self.target_name = target_name

        if isinstance(u_max, (float, int)):
            self.u_max = np.full(len(joint_names), float(u_max))
        elif isinstance(u_max, (list, np.ndarray)):
            self.u_max = np.array(u_max, dtype=float)
        else:
            raise ValueError("u_max must be a float, int, list, or np.ndarray.")
        
        self.dt = float(dt)

        # Get joint IDs based on the provided joint names
        self.joint_ids = np.array(
            [self.model.joint(name).id for name in self.joint_names],
            dtype=int,
        )

        # Get qpos and dof IDs for the specified joints
        self.qpos_ids = np.array([self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=int)
        self.dof_ids = np.array([self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids], dtype=int)

        # Get actuator IDs for the specified joints
        self.motor_ids = np.array(
            [self.model.actuator(name).id for name in self.joint_names],
            dtype=int,
        )

        # Get the target site ID based on the provided target name
        self.target_site_id = self.model.site(self.target_name).id
    

    def apply_control(self, control_signal):
        """Apply the computed control signal to the actuators.

        Args:
            control_signal (np.ndarray): The control signal to be applied to the actuators.
        """
        # Check if the control signal is a numpy array and has the correct shape
        if not isinstance(control_signal, np.ndarray):
            raise ValueError("Control signal must be a numpy array.")
        
        if control_signal.shape != (len(self.motor_ids),):
            raise ValueError(f"Control signal must have shape ({len(self.motor_ids)},), but got {control_signal.shape}.")

        # Ensure the control signal is within the specified limits
        control_signal = np.clip(control_signal, -self.u_max, self.u_max)

        # Apply the control signal to the actuators
        self.data.ctrl[self.motor_ids] = control_signal

    
    def update(self):
        """Update the controller state. This method should be overridden by subclasses to implement specific control logic."""
        raise NotImplementedError("The update method must be implemented by subclasses.")