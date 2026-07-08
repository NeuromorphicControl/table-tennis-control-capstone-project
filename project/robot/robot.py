import numpy as np

from .controller import OperationalSpaceController
from .controller import PIDController

class RobotArm:
    def __init__(self, model, data, joint_names, target_site_name="paddle_site", base_site_name="base_site", base_offset=None, u_max=100, dt=0.01):
        self.arm_controller = OperationalSpaceController(
            model=model,
            data=data,
            joint_names=joint_names[2:],
            target_site_name=target_site_name,
            u_max=u_max,
            dt=dt,
        )

        self.base_controller = PIDController(
            model=model,
            data=data,
            joint_names=joint_names[:2],
            target_site_name=base_site_name,
            u_max=u_max,
            dt=dt,
        )
        if base_offset is not None and len(base_offset) != 2:
            raise ValueError("base_offset must be a 3D vector (x, y).")
        elif base_offset is None:
            base_offset = np.zeros(2, dtype=float)
        
        # Append 3rd dimension (z) to base_offset to make it a 3D vector
        self.base_offset = np.array([base_offset[0], base_offset[1], 0.0], dtype=float)


    def set_target_pose(self, position, rotation):
        self.arm_controller.set_target_pose(position, rotation)
        self.base_controller.set_target_position(position - self.base_offset)

    def set_target_position(self, position):
        self.arm_controller.set_target_position(position)
        self.base_controller.set_target_position(position - self.base_offset)

    def set_target_rotation(self, rotation):
        self.arm_controller.set_target_rotation(rotation)


    def get_target_site_pose(self):
        return self.arm_controller.get_site_pose()

    def update(self):
        data = []
        data.append(self.base_controller.update())
        data.append(self.arm_controller.update())
        return data