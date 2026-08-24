"""The controlled arm: operational-space control plus collision avoidance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ArmSpec, CollisionConfig, ControlConfig
from ..planning.trajectory import TaskState
from .collision_avoidance import CollisionAvoider, CollisionState
from .operational_space import ControllerDiagnostics, OperationalSpaceController

__all__ = ["RobotArm", "ArmDiagnostics"]


@dataclass
class ArmDiagnostics:
    """Everything the HUD and the logs want to know about one control step."""

    controller: ControllerDiagnostics
    collision: CollisionState

    @property
    def position_error(self) -> float:
        return float(np.linalg.norm(self.controller.position_error))

    @property
    def orientation_error(self) -> float:
        return float(np.linalg.norm(self.controller.orientation_error))


class RobotArm:
    """Facade combining the low-level control blocks of the arm.

    The arm only ever follows a *reference task state*.  Deciding what that
    reference should be is the planner's job, which keeps the split between
    the (fast, linear) inner loop and the (slow, non-linear) outer loop from
    the lecture intact.
    """

    def __init__(self, model, data, arm: ArmSpec | None = None, control: ControlConfig | None = None, collision: CollisionConfig | None = None):
        """Create a new :class:`RobotArm` for the given MuJoCo model and data.
        
        Args:
            model: The MuJoCo model.
            data: The MuJoCo data.
            arm: The arm specification (joint limits, etc.).  If ``None``, a default :class:`ArmSpec` is used.
            control: The control configuration (gains, etc.).  If ``None``, a default :class:`ControlConfig` is used.
            collision: The collision avoidance configuration.  If ``None``, a default :class:`:CollisionConfig` is used.
        """
        self.model = model
        self.data = data
        self.spec = arm or ArmSpec()
        self.controller = OperationalSpaceController(model, data, self.spec, control)
        self.avoider = CollisionAvoider(model, data, self.controller.dof_ids, collision)
        self.reference = self.measure()

    # ------------------------------------------------------------------ state
    def measure(self) -> TaskState:
        """Measured task state of the paddle."""
        return self.controller.measure()

    @property
    def position(self) -> np.ndarray:
        return self.controller.paddle_position

    @property
    def normal(self) -> np.ndarray:
        return self.controller.paddle_normal

    def hold(self) -> TaskState:
        """Freeze the reference at the current measured pose."""
        measured = self.measure()
        self.reference = TaskState(
            position=measured.position,
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            normal=measured.normal,
            angular_velocity=np.zeros(3),
        )
        return self.reference

    # ----------------------------------------------------------------- control
    def update(self, reference: TaskState | None = None) -> ArmDiagnostics:
        """Run one control step against ``reference`` (or the stored one)."""
        if reference is not None:
            self.reference = reference

        collision = self.avoider.compute()
        diagnostics = self.controller.compute(self.reference, extra_joint_torque=collision.torque)
        return ArmDiagnostics(controller=diagnostics, collision=collision)

    def limit_velocity(self) -> None:
        """Hard safety cap on joint speed; call once after every physics step."""
        dof_ids = self.controller.dof_ids
        qvel = self.data.qvel[dof_ids]
        speed = float(np.linalg.norm(qvel))

        limit = self.spec.max_joint_speed
        if speed > limit:
            self.data.qvel[dof_ids] = qvel * (limit / speed)
