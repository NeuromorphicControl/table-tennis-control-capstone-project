r"""Operational-space control of the paddle (Lecture 5/6).

The joint torque is assembled from three clearly separated blocks, exactly the
decomposition used in the lecture:

.. math::

    u = \underbrace{J^{\mathsf T}\Lambda\,(f^* - \dot J \dot q)}_{u_\text{task}}
      + \underbrace{C(q,\dot q)\dot q + G(q)}_{u_\text{FL}}
      + \underbrace{N^{\mathsf T} u_0}_{\text{null space}}

``u_FL`` is the feedback-linearisation term that cancels gravity and the
Coriolis/centrifugal forces (MuJoCo hands it to us as ``qfrc_bias``).  What is
left behind is a unit mass floating in task space, so the virtual force

.. math:: f^* = \ddot x_\text{ref} + K_p (x_\text{ref}-x) + K_d(\dot x_\text{ref}-\dot x)

is a plain feed-forward + PD law whose gains directly set the closed-loop
natural frequency.  Whatever motion is left over is used by the null-space
term for posture and collision avoidance.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..config import ArmSpec, ControlConfig
from ..kinematics import axis_alignment_error, normalise
from ..planning.trajectory import TaskState

__all__ = ["OperationalSpaceController", "ControllerDiagnostics"]


def _clamp_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    """Scale ``vector`` down so that its norm does not exceed ``limit``."""
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit or norm < 1e-12 else vector * (limit / norm)


@dataclass
class ControllerDiagnostics:
    """Signals the HUD and the logs are interested in."""

    position_error: np.ndarray
    orientation_error: np.ndarray
    task_force: np.ndarray
    torque: np.ndarray
    saturation: float


class OperationalSpaceController:
    """Torque-level tracking controller for the paddle."""

    def __init__(self, model, data, arm: ArmSpec, config: ControlConfig | None = None):
        """Initialise the controller with the MuJoCo model and data, the arm spec, and the control config.
        
        Args:
            model: MuJoCo model.
            data: MuJoCo data.
            arm: Arm spec.
            config: Control config. If None, a default config is used.
        """
        self.model = model
        self.data = data
        self.arm = arm
        self.config = config or ControlConfig()

        # Get the joint, qpos, dof, and actuator IDs for the arm's joints, as well as the site and body IDs for the paddle
        self.joint_ids = np.array([model.joint(name).id for name in arm.joint_names], dtype=int)
        self.qpos_ids = np.array([model.jnt_qposadr[j] for j in self.joint_ids], dtype=int)
        self.dof_ids = np.array([model.jnt_dofadr[j] for j in self.joint_ids], dtype=int)
        self.actuator_ids = np.array([model.actuator(name).id for name in arm.joint_names], dtype=int)
        self.site_id = int(model.site(arm.site_name).id)
        self.body_id = int(model.site_bodyid[self.site_id])
        self.n_dof = self.dof_ids.size

        # Set up the controller gains and limits
        self.torque_limits = np.asarray(arm.torque_limits, dtype=float)
        if self.torque_limits.size == 1:
            self.torque_limits = np.full(self.n_dof, float(self.torque_limits))

        # Set up the controller gains and limits
        self.position_kp = np.asarray(self.config.position_kp, dtype=float)
        self.position_kd = np.asarray(self.config.position_kd, dtype=float)
        self.orientation_kp = np.asarray(self.config.orientation_kp, dtype=float)
        self.orientation_kd = np.asarray(self.config.orientation_kd, dtype=float)

        # Set up the home configuration and posture weights for the null-space control
        self.home_configuration = np.asarray(arm.neutral_qpos, dtype=float)
        self.posture_weights = np.ones(self.n_dof)
        self.posture_weights[:2] = self.config.base_posture_weight

        # Scratch buffers so the control loop allocates nothing
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._jacp_dot = np.zeros((3, model.nv))
        self._jacr_dot = np.zeros((3, model.nv))
        self._mass = np.zeros((model.nv, model.nv))

        # Cache the Jacobian and its time to avoid recomputing it if the state hasn't changed
        self._jac_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._jac_cache_time: float | None = None

    # ------------------------------------------------------------------ state
    @property
    def paddle_position(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    @property
    def paddle_rotation(self) -> np.ndarray:
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    @property
    def paddle_normal(self) -> np.ndarray:
        return normalise(self.paddle_rotation[:, self.arm.normal_axis])

    def jacobians(self) -> tuple[np.ndarray, np.ndarray]:
        """Translational and rotational site Jacobian restricted to the arm.
        
        If the state hasn't changed since the last call, return the cached Jacobian instead of recomputing it.

        Returns:
            jacp: Translational Jacobian, shape (3, n_dof).
            jacr: Rotational Jacobian, shape (3, n_dof).
        """
        if self._jac_cache is not None and self._jac_cache_time == self.data.time:
            return self._jac_cache

        # Compute the Jacobian for the paddle site using MuJoCo's built-in function
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id) # type: ignore

        # Cache the Jacobian restricted to the arm's degrees of freedom and the current time
        self._jac_cache = (self._jacp[:, self.dof_ids], self._jacr[:, self.dof_ids])
        self._jac_cache_time = self.data.time
        return self._jac_cache

    def measure(self) -> TaskState:
        """Current *measured* task state of the paddle.
        
        Returns:
            TaskState: Current position, velocity, acceleration, normal, and angular velocity of the paddle
        """
        jacp, jacr = self.jacobians()
        qvel = self.data.qvel[self.dof_ids]
        return TaskState(
            position=self.paddle_position,
            velocity=jacp @ qvel,
            acceleration=np.zeros(3),
            normal=self.paddle_normal,
            angular_velocity=jacr @ qvel,
        )

    # ------------------------------------------------------------------ update
    def _jacobian_dot(self) -> np.ndarray:
        """Time derivative of the task Jacobian, ``(6, n_dof)``."""
        if not self.config.use_jacobian_dot:
            return np.zeros((6, self.n_dof))
        mujoco.mj_jacDot(self.model, self.data, self._jacp_dot, self._jacr_dot, self.data.site_xpos[self.site_id], self.body_id) # type: ignore
        return np.vstack([self._jacp_dot[:, self.dof_ids], self._jacr_dot[:, self.dof_ids]])

    def compute(
        self,
        reference: TaskState,
        extra_joint_torque: np.ndarray | None = None,
    ) -> ControllerDiagnostics:
        """Compute and apply the actuator torques for one control step.

        Args:
            reference: Desired task state produced by the planner.
            extra_joint_torque: Additional joint torque to be added to the controller output, e.g. for collision avoidance.
        
        Returns:
            ControllerDiagnostics: Diagnostics information about the controller's performance, including position and orientation errors, 
                task force, applied torque, and saturation level.
        """
        jacp, jacr = self.jacobians()
        jacobian = np.vstack([jacp, jacr])
        qvel = self.data.qvel[self.dof_ids]

        # Compute the mass matrix and its inverse for the arm's degrees of freedom
        mujoco.mj_fullM(self.model, self._mass, self.data.qM) # type: ignore
        mass = self._mass[np.ix_(self.dof_ids, self.dof_ids)]
        mass_inv = np.linalg.inv(mass)

        # Compute the operational-space inertia matrix and its inverse, with regularisation to avoid singularities
        inertia_inv = jacobian @ mass_inv @ jacobian.T
        inertia_inv += self.config.inertia_regularisation * np.eye(6)
        inertia = np.linalg.pinv(inertia_inv)

        # --- task errors -------------------------------------------------
        position = self.paddle_position
        velocity = jacp @ qvel
        angular_velocity = jacr @ qvel

        # Compute the errors between the reference and measured task states
        position_error = reference.position - position
        velocity_error = reference.velocity - velocity
        orientation_error = axis_alignment_error(self.paddle_rotation, reference.normal, self.arm.normal_axis)
        angular_error = reference.angular_velocity - angular_velocity

        # Clamp the errors to avoid excessive control commands
        linear_command = reference.acceleration + self.position_kp * _clamp_norm(position_error, self.config.max_position_error) + self.position_kd * velocity_error
        angular_command = reference.angular_acceleration + self.orientation_kp * _clamp_norm(orientation_error, self.config.max_orientation_error) + self.orientation_kd * angular_error
        command = np.concatenate([linear_command, angular_command])

        # --- operational-space force -------------------------------------
        bias = self._jacobian_dot() @ qvel
        task_force = _clamp_norm(inertia @ (command - bias), self.config.max_task_force)

        # Convert the operational-space force into joint torques using the Jacobian transpose
        torque = jacobian.T @ task_force

        # --- feedback linearisation --------------------------------------
        torque += self.data.qfrc_bias[self.dof_ids]

        # --- null-space posture ------------------------------------------
        # Compute the null-space projector to ensure that the posture control does not interfere with the task-space control
        null_projector = np.eye(self.n_dof) - jacobian.T @ inertia @ jacobian @ mass_inv
        posture_error = self.home_configuration - self.data.qpos[self.qpos_ids]

        # Compute the posture control torque using PD control, weighted by the posture weights, and project it into the null space
        posture_torque = self.posture_weights * (self.config.posture_kp * posture_error - self.config.posture_kd * qvel)
        torque += null_projector @ posture_torque

        if extra_joint_torque is not None:
            torque += extra_joint_torque

        # Apply the computed torque to the actuators, ensuring that it does not exceed the actuator limits
        limited = np.clip(torque, -self.torque_limits, self.torque_limits)
        self.data.ctrl[self.actuator_ids] = limited

        # Compute the saturation level of the applied torque relative to the actuator limits and return the controller diagnostics
        saturation = float(np.max(np.abs(limited) / self.torque_limits))
        return ControllerDiagnostics(
            position_error=position_error,
            orientation_error=orientation_error,
            task_force=task_force,
            torque=limited,
            saturation=saturation,
        )
