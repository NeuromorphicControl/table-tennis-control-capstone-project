from pathlib import Path

import numpy as np
import mujoco

from mujocohelper import Renderer
from world.robot import RobotArm

from scipy.spatial.transform import Rotation

MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")
mj_step = getattr(mujoco, "mj_step")



# List of demo Poses for the robot arm, each defined by a target position and rotation (in Euler angles)
demo_poses = [
    {
        "position": np.array([0.0, 0.6, 0.25]),
        "rotation": np.array([np.pi / 4, 0.0, 0.0]),  # Euler angles (roll, pitch, yaw)
        "ball_velocity": np.array([0.0, 0.0, 0.0]),  # Initial velocity of the ball
    },
    {
        "position": np.array([0.4, -0.2, 0.0]),
        "rotation": np.array([np.pi / 4, 0.0, np.pi / 4]),  # Euler angles (roll, pitch, yaw)
        "ball_velocity": np.array([0.0, 0.0, 0.0]),  # Initial velocity of the ball
    },
    {
        "position": np.array([-0.4, -0.2, 0.5]),
        "rotation": np.array([0.0, np.pi / 4, np.pi/2]),  # Euler angles (roll, pitch, yaw)
        "ball_velocity": np.array([0.0, 0.0, 0.0]),  # Initial velocity of the ball
    },
]


if __name__ == "__main__":
    # Load the MuJoCo model and create a data object
    model = MjModel.from_xml_path("project/world.xml")
    data = MjData(model)

    # Create a RobotArm instance with the loaded data and joint names
    joint_names = ["base_x", "base_y", "rotator1", "rotator2", "arm1", "arm2", "paddle_rotator", "paddle"]
    robot_arm = RobotArm(model, data, joint_names, site_name="paddle_site", return_home=True, base_pos=(0, 0), dt=model.opt.timestep)

    robot_arm.set_task_gains(
            position_kp=[120.0, 120.0, 120.0],
            position_kd=[38.0, 38.0, 38.0],
            orientation_kp=[120.0, 120.0, 120.0],
            orientation_kd=[38.0, 38.0, 38.0],
            null_kp=0.0,
            null_kd=12, # Damping for the base joints to prevent oscillations
    )

    initial_pose = robot_arm.get_site_pose()

    # Store the initial pose to reset the robot arm to its starting position after each demo pose
    initial_pose = {
        "position": initial_pose["position"].copy(),
        "rotation": initial_pose["rotation"].copy(),
    }

    robot_arm.set_target_pose(
            initial_pose["position"],
            Rotation.from_matrix(initial_pose["rotation"]).as_euler("xyz"),
    )

    # Set the initial position and velocity of the ball
    ball_initial_position = np.array([0.0, -1.0, 1.5])  # Initial position of the ball
    ball_initial_velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity of the ball

    # Get joint address for the ball's free joint
    ball_joint_id = model.joint("ball_free").id

    # Set the ball's position and velocity in the simulation
    data.qpos[ball_joint_id:ball_joint_id + 3] = ball_initial_position
    data.qvel[ball_joint_id:ball_joint_id + 3] = ball_initial_velocity

    counter = 0
    index = 0

    duration = 60
    framerate = 24
    frame_idx = 0

    videopath = Path(".") / "output"
    videopath.mkdir(parents=True, exist_ok=True)
    
    with Renderer(model, height=1080, width=1920) as renderer:
        renderer.init_video(videopath / "presentation.mp4", framerate=framerate)
        while data.time < duration:

            if counter >= 5 / model.opt.timestep:  # 5 seconds in simulation time
                index += 1
                if index >= len(demo_poses):
                    index = 0
                counter = 0
                robot_arm.set_target_pose(
                    initial_pose["position"] + demo_poses[index]["position"],
                    Rotation.from_matrix(initial_pose["rotation"]).as_euler("xyz") + demo_poses[index]["rotation"],
                )

                # Reset the ball's position and velocity for the new demo pose
                data.qpos[ball_joint_id:ball_joint_id + 3] = ball_initial_position
                data.qvel[ball_joint_id:ball_joint_id + 3] = ball_initial_velocity
            counter += 1

            robot_arm.update()
            mj_step(model, data)

            if frame_idx < data.time * framerate:
                renderer.update_scene(data, camera="render_cam")
                renderer.render_frame()
                frame_idx += 1