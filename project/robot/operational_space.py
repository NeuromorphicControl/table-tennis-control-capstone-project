import numpy as np
import mujoco

from scipy.spatial.transform import Rotation

mj_forward = getattr(mujoco, "mj_forward")
mj_jacSite = getattr(mujoco, "mj_jacSite")
mj_fullM = getattr(mujoco, "mj_fullM")

class OperationalSpaceController:
    def __init__(
        self,
        model,
        data,
        joint_names,
        site_name,
        return_home=False,
        base_pos=None,
        u_max=100.0,
        dt=0.005,
    ):
        self.model = model
        self.data = data
        self.joint_names = list(joint_names)
        self.site_name = site_name
        self.return_home = return_home
        self.base_pos = base_pos or (0.0, 0.0)
        self.u_max = float(u_max)
        self.dt = float(dt)

        self.joint_ids = np.array(
            [self.model.joint(name).id for name in self.joint_names],
            dtype=int,
        )
        self.qpos_ids = np.array([self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=int)
        self.dof_ids = np.array([self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids], dtype=int)
        self.motor_ids = np.array(
            [self.model.actuator(name).id for name in self.joint_names],
            dtype=int,
        )
        self.site_id = self.model.site(self.site_name).id

        self.position_kp = np.array([80.0, 80.0, 80.0], dtype=float)
        self.position_kd = np.array([22.0, 22.0, 22.0], dtype=float)
        self.orientation_kp = np.array([70.0, 70.0, 70.0], dtype=float)
        self.orientation_kd = np.array([18.0, 18.0, 18.0], dtype=float)
        self.null_kp = 8.0
        self.null_kd = 2.0
        self.normal_axis = 1

        self.data.qpos[self.qpos_ids[0:2]] = self.base_pos[0:2]

        mj_forward(self.model, self.data)
            
        if self.return_home and len(self.qpos_ids) >= 2:
            self.q_home = self.data.qpos[self.qpos_ids[0:2]].copy()
        
        self.target_position = self.data.site_xpos[self.site_id].copy()
        self.target_rotation = self._current_rotation()

    def _current_rotation(self):
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        return rotation
    
    def _compute_orientation_error(self, q_curr, q_des):
        """Compute the orientation error between current and desired rotations.
        current_rotation: current rotation matrix (3x3)
        desired_rotation: desired rotation matrix (3x3)
        """
        # Create rotation objects
        rot_curr = Rotation.from_matrix(q_curr)
        rot_des = Rotation.from_matrix(q_des)
        
        # Compute the relative rotation: R_err = R_des * inv(R_curr)
        relative_rot = rot_des * rot_curr.inv()
        
        return relative_rot.as_rotvec() 

    def get_site_pose(self):
        return {
            "position": self.data.site_xpos[self.site_id].copy(),
            "rotation": self._current_rotation(),
        }

    def set_task_gains(self, position_kp=None, position_kd=None, orientation_kp=None, orientation_kd=None, null_kp=None, null_kd=None):
        if position_kp is not None:
            self.position_kp = np.broadcast_to(np.asarray(position_kp, dtype=float), (3,)).copy()
        if position_kd is not None:
            self.position_kd = np.broadcast_to(np.asarray(position_kd, dtype=float), (3,)).copy()
        if orientation_kp is not None:
            self.orientation_kp = np.broadcast_to(np.asarray(orientation_kp, dtype=float), (3,)).copy()
        if orientation_kd is not None:
            self.orientation_kd = np.broadcast_to(np.asarray(orientation_kd, dtype=float), (3,)).copy()
        if null_kp is not None:
            self.null_kp = float(null_kp)
        if null_kd is not None:
            self.null_kd = float(null_kd)

    def set_target_pose(self, position, rotation):
        self.set_target_position(position)
        self.set_target_rotation(rotation)

    def set_target_position(self, position):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,):
            raise ValueError("Target position must be a 3D vector.")
        self.target_position = position

    def set_target_rotation(self, rotation):
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3,):
            raise ValueError("Target rotation must be a 3D xyz euler vector.")
        self.target_rotation = Rotation.from_euler("xyz", rotation).as_matrix()
    
    def set_base_position(self, base_pos):
        base_pos = np.asarray(base_pos, dtype=float)
        if base_pos.shape != (2,):
            raise ValueError("Base position must be a 2D vector.")
        self.base_pos = base_pos

    def update(self):
        mj_forward(self.model, self.data)

        current_position = self.data.site_xpos[self.site_id].copy()

        pos_error = self.target_position - current_position
        ori_error = self._compute_orientation_error(
            self._current_rotation(),
            self.target_rotation,
        )
        task_error = np.concatenate([pos_error, ori_error])

        jacobian_position = np.zeros((3, self.model.nv), dtype=float)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=float)
        mj_jacSite(self.model, self.data, jacobian_position, jacobian_rotation, self.site_id)
        task_jacobian = np.vstack([jacobian_position, jacobian_rotation])

        qvel = self.data.qvel.copy()
        task_velocity = task_jacobian @ qvel

        mass_matrix = np.zeros((self.model.nv, self.model.nv), dtype=float)
        mj_fullM(self.model, mass_matrix, self.data.qM)
        mass_matrix_inv = np.linalg.inv(mass_matrix)

        operational_inertia_inv = task_jacobian @ mass_matrix_inv @ task_jacobian.T
        operational_inertia_inv += 1e-6 * np.eye(task_jacobian.shape[0])
        operational_inertia = np.linalg.pinv(operational_inertia_inv)

        task_stiffness = np.concatenate([self.position_kp, self.orientation_kp])
        task_damping = np.concatenate([self.position_kd, self.orientation_kd])
        task_acceleration = task_stiffness * task_error - task_damping * task_velocity

        task_force = operational_inertia @ task_acceleration
        joint_torque = task_jacobian.T @ task_force

        nullspace_projector = np.eye(self.model.nv) - (mass_matrix_inv @ task_jacobian.T @ operational_inertia @ task_jacobian)

        # Cancel bias forces if requested
        joint_torque += self.data.qfrc_bias.copy()

        if self.return_home and len(self.qpos_ids) >= 2:
            posture_error = self.base_pos - self.data.qpos[self.qpos_ids[0:2]]
            posture_torque = self.null_kp * posture_error - self.null_kd * self.data.qvel[self.dof_ids[0:2]]
            
            # Apply posture torque only to the base joints (first two joints) by filling the rest with zeros
            full_posture_torque = np.zeros(self.model.nv, dtype=float)
            full_posture_torque[self.dof_ids[0:2]] = posture_torque

            joint_torque += nullspace_projector.T @ full_posture_torque
        
        self.data.ctrl[self.motor_ids] = np.clip(joint_torque[self.dof_ids], -self.u_max, self.u_max)

        return {
            "position_error": pos_error,
            "orientation_error": ori_error,
            "joint_torque": joint_torque,
        }