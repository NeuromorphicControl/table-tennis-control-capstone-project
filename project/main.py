import time

import numpy as np
import mujoco
import mujoco.viewer

from robot.robot import RobotArm

from scipy.spatial.transform import Rotation

MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")
mj_step = getattr(mujoco, "mj_step")

# List of demo Poses for the robot arm, each defined by a target position and rotation (in Euler angles)
demo_poses = [
    {
        "position": np.array([0.0, 0.6, 0.25]),      # Target position in 3D space (x, y, z)
        "rotation": np.array([np.pi / 4, 0.0, 0.0]), # Euler angles (roll, pitch, yaw)
    },
    {
        "position": np.array([0.4, -0.2, 0.0]),
        "rotation": np.array([np.pi / 4, 0.0, np.pi / 4]),  # Euler angles (roll, pitch, yaw)
    },
    {
        "position": np.array([-0.4, -0.2, 0.5]),
        "rotation": np.array([0.0, np.pi / 4, np.pi/2]),  # Euler angles (roll, pitch, yaw)
    },
]


if __name__ == "__main__":
    # Load the MuJoCo model and create a data object
    model = MjModel.from_xml_path("project/world.xml")
    data = MjData(model)

    # Create a RobotArm instance with the loaded data and joint names
    joint_names = ["base_x", "base_y", "rotator1", "rotator2", "arm1", "arm2", "paddle_rotator", "paddle"]
    robot_arm = RobotArm(model, data, joint_names, target_site_name="paddle_site", base_site_name="base_site", base_offset=(0, 0), dt=model.opt.timestep)

    # Set PID gains for position and orientation control of the robot arm
    robot_arm.arm_controller.set_gains(
            position_kp=[120.0, 120.0, 120.0],
            position_kd=[38.0, 38.0, 38.0],
            orientation_kp=[120.0, 120.0, 120.0],
            orientation_kd=[38.0, 38.0, 38.0],
    )

    # Set PID gains for the base controller of the robot arm
    robot_arm.base_controller.set_gains(
            kp=[0, 0],
            ki=[0.0, 0.0],
            kd=[38.0, 38.0],
    )

    # Get the initial pose of the robot arm's target site (paddle) to use as a reference for demo poses
    initial_pose = robot_arm.get_target_site_pose()

    # Store the initial pose to reset the robot arm to its starting position after each demo pose
    initial_pose = {
        "position": initial_pose["position"].copy(),
        "rotation": initial_pose["rotation"].copy(),
    }

    robot_arm.set_target_pose(
            initial_pose["position"],
            initial_pose["rotation"],
    )

    counter = 0
    index = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if counter >= 10 / model.opt.timestep:  # 10 seconds in simulation time
                index += 1
                if index >= len(demo_poses):
                    index = 0
                counter = 0
                robot_arm.set_target_pose(
                    initial_pose["position"] + demo_poses[index]["position"],
                    initial_pose["rotation"] @ Rotation.from_euler("xyz", demo_poses[index]["rotation"]).as_matrix(),
                )
                print(f"Switching to demo pose {index + 1}: Position {demo_poses[index]['position']}, Rotation {demo_poses[index]['rotation']}")
            counter += 1

            robot_arm.update()
            mj_step(model, data)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)