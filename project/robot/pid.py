import numpy as np

class PIDController:
    """Proportional-Integral-Derivative (PID) controller class. For multiple joints at once."""

    def __init__(self, k_P, k_I, k_D, u_max, dt):
        """Constructor method.

        Args:
            k_P (list): Proportional gains for each joint.
            k_I (list): Integral gains for each joint.
            k_D (list): Derivative gains for each joint.
            u_max (float): Maximum control signal output.
            dt (float): Time step for the controller updates.
        """

        self.k_P = np.array(k_P)
        self.k_I = np.array(k_I)
        self.k_D = np.array(k_D)
        self.e_last = np.zeros_like(self.k_P)
        self.y_last = np.zeros_like(self.k_P)
        self.e_integral = np.zeros_like(self.k_P)
        self.e_derivative = np.zeros_like(self.k_P)
        self.u_max = u_max
        self.dt = dt
    
    def set_gains(self, k_P, k_I, k_D):
        """Set the PID gains.

        Args:
            k_P (list): Proportional gains for each joint.
            k_I (list): Integral gains for each joint.
            k_D (list): Derivative gains for each joint.
        """
        self.k_P = np.array(k_P)
        self.k_I = np.array(k_I)
        self.k_D = np.array(k_D)

    def update(self, pos, vel, targets):
        """Compute the controller output.

        Args:
            pos (np.ndarray): The current joint positions.
            vel (np.ndarray): The current joint velocities.
            targets (np.ndarray): The desired target positions.

        Returns:
            np.ndarray: The control signal outputs from the PID controller for each joint.
        """

        # Compute the error
        e = targets - pos

        # Compute the (first-order) derivative of the error
        self.e_derivative = -vel

        # Compute the control signal
        u = self.k_P * e + self.k_I * self.e_integral + self.k_D * self.e_derivative

        # Check for control signal saturation
        u_saturated = np.clip(u, -self.u_max, self.u_max)

        # Only integrate if control signal is not saturated
        self.e_integral += self.dt * e * (u == u_saturated)

        # Save the latest error
        self.e_last = e

        return u_saturated