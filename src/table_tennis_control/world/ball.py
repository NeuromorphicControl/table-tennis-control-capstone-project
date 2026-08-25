"""Access to the ball's state inside the MuJoCo simulation."""

from __future__ import annotations

import numpy as np


class Ball:
    """Thin wrapper around the free joint of the ball body.

    The class only exposes the *plant* state.  Everything the controller is
    allowed to use goes through :meth:`measure` and the delayed sensor built
    on top of it (:class:`~table_tennis_control.world.ball_sensor.BallSensor`)
    on the way to the observer in :mod:`table_tennis_control.estimation`,
    which mirrors the "the controller only sees the measurement" structure of
    the lecture.
    """

    def __init__(self, model, data, joint_name: str = "ball_free", geom_name: str = "ball_geom"):
        self.model = model
        self.data = data
        self.joint_name = joint_name

        joint_id = model.joint(joint_name).id
        self.qpos_address = int(model.jnt_qposadr[joint_id])
        self.dof_address = int(model.jnt_dofadr[joint_id])
        self.geom_id = int(model.geom(geom_name).id)
        self.radius = float(model.geom_size[self.geom_id][0])

    @property
    def position(self) -> np.ndarray:
        """Current position of the ball centre (a view into ``data.qpos``)."""
        return self.data.qpos[self.qpos_address : self.qpos_address + 3]

    @property
    def velocity(self) -> np.ndarray:
        """Current linear velocity (a view into ``data.qvel``)."""
        return self.data.qvel[self.dof_address : self.dof_address + 3]

    def set_state(self, position, velocity=(0.0, 0.0, 0.0)) -> None:
        """Teleport the ball and set its linear velocity."""
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("position and velocity must both be 3-vectors")

        self.data.qpos[self.qpos_address : self.qpos_address + 3] = position
        self.data.qpos[self.qpos_address + 3 : self.qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[self.dof_address : self.dof_address + 3] = velocity
        self.data.qvel[self.dof_address + 3 : self.dof_address + 6] = 0.0

    def measure(self) -> np.ndarray:
        """Instantaneous, true position measurement."""
        return self.position.copy()
