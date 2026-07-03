from .operational_space import OperationalSpaceController

class RobotArm:
    def __init__(self, model, data, joint_names, site_name="paddle_site", return_home=False, base_pos=None, u_max=100, dt=0.01):
        self.controller = OperationalSpaceController(
            model=model,
            data=data,
            joint_names=joint_names,
            site_name=site_name,
            return_home=return_home,
            base_pos=base_pos,
            u_max=u_max,
            dt=dt,
        )

        self.data = data
        self.model = model
        self.joint_names = joint_names
        self.return_home = return_home
        self.base_pos = base_pos or (0, 0)
        self.u_max = u_max

        self.joint_ids = self.controller.qpos_ids
        self.motor_ids = self.controller.motor_ids
        self.target_position = self.controller.target_position
        self.target_rotation = self.controller.target_rotation
    

    def set_task_gains(self, position_kp=None, position_kd=None, orientation_kp=None, orientation_kd=None, null_kp=None, null_kd=None):
        self.controller.set_task_gains(position_kp, position_kd, orientation_kp, orientation_kd, null_kp, null_kd)

    def set_target_pose(self, position, rotation):
        self.controller.set_target_pose(position, rotation)
        self.target_position = self.controller.target_position
        self.target_rotation = self.controller.target_rotation

    def set_target_position(self, position):
        self.controller.set_target_position(position)
        self.target_position = self.controller.target_position

    def set_target_rotation(self, rotation):
        self.controller.set_target_rotation(rotation)
        self.target_rotation = self.controller.target_rotation

    def set_base_position(self, base_pos):
        self.controller.set_base_position(base_pos)
        self.base_pos = self.controller.base_pos

    def get_site_pose(self):
        return self.controller.get_site_pose()
    
    def update(self):
        return self.controller.update()