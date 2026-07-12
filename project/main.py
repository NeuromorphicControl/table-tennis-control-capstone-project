from pathlib import Path
import time

import numpy as np
import mujoco
import mujoco.viewer

from predict import TrajectoryPredictor
from world import Ball
from robot import RobotArm

from predict import calculate_path, calculate_path_numba, calculate_optimal_target_position

from scipy.spatial.transform import Rotation

MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")
mj_step = getattr(mujoco, "mj_step")

def key_callback(keycode):
    """Callback function for key events in the MuJoCo viewer.

    Args:
        keycode (int): The keycode of the pressed key.
    """
    if keycode == 32:
        # Reset the ball's position and velocity when the Space key is pressed
        ball.reset_position((0, 0, 2))
        
        vel = (0, -3, 2)
        vel += np.random.uniform(-0.5, 0.5, size=3)
        ball.set_velocity(vel)

if __name__ == "__main__":
    # Load the MuJoCo model and create a data object
    model = MjModel.from_xml_path("project/world.xml")
    data = MjData(model)

    # Create a Ball instance with the loaded data and joint name
    ball = Ball(model, data, ball_joint_name="ball_free")

    # Create a RobotArm instance with the loaded data and joint names
    joint_names = ["base_x", "base_y", "rotator1", "rotator2", "arm1", "arm2", "paddle_rotator", "paddle"]
    robot_arm = RobotArm(model, data, joint_names, site_name="paddle_site", return_home=True, base_pos=(0.2, 0), dt=model.opt.timestep)

    robot_arm.set_task_gains(
            position_kp=[300.0, 300.0, 300.0],
            position_kd=[29.6, 29.6, 29.6],
            orientation_kp=[680.0, 680.0, 680.0],
            orientation_kd=[23.0, 23.0, 23.0],
            null_kp=80.0,
            null_kd=60.0, # Damping for the base joints to prevent oscillations
    )

    initial_pose = robot_arm.get_site_pose()

    # Store the initial pose to reset the robot arm to its starting position after each demo pose
    initial_pose = {
        "position": initial_pose["position"].copy(),
        "rotation": initial_pose["rotation"].copy(),
    }

    # TODO: Write method to check working area of the robot arm
    working_x_min, working_x_max = initial_pose["position"][0] - 0.8, initial_pose["position"][0] + 0.8
    working_y_min, working_y_max = initial_pose["position"][1] - 0.8, initial_pose["position"][1] + 0.8
    working_z_min, working_z_max = 0.4, 1.5

    robot_arm.set_target_pose(
            initial_pose["position"],
            Rotation.from_matrix(initial_pose["rotation"]).as_euler("xyz"),
    )

    # TODO: Move target site movement to a separate function that can be called from the main loop
    data.site("target").xpos = initial_pose["position"]

    # Set the ball to its initial position and velocity
    ball.reset_position((0, 0, 2))

    vel = (0, -3, 2)
    vel += np.random.uniform(-0.5, 0.5, size=3)
    ball.set_velocity(vel)

    # Initialize TrajectoryPredictor with the ball instance
    trajectory_predictor = TrajectoryPredictor(ball, dt=model.opt.timestep, t_max=10)

    state = 0

    tolerance = 2  # Tolerance for considering the ball to be at the target position

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Update the ball's position and velocity, and recalculate the predicted trajectory
            ball.update()
            trajectory_predictor.update()

            # Get the predicted trajectory of the ball
            times, positions, velocities = trajectory_predictor.get_trajectory()
            
            # Reduce the number of positions to only those within the working area of the robot arm
            valid_positions_mask = (np.logical_and.reduce((positions[:, 0] >= working_x_min, positions[:, 0] <= working_x_max,
                                                        positions[:, 1] >= working_y_min, positions[:, 1] <= working_y_max,
                                                        positions[:, 2] >= working_z_min, positions[:, 2] <= working_z_max)))
            
            # If there are valid positions and the robot arm is in the waiting state or the target position has changed, calculate the optimal target position and set the robot arm's target pose
            if np.any(valid_positions_mask) and (state == 0 or np.linalg.norm(positions[valid_positions_mask][-1] - robot_arm.target_position) > tolerance):
                target_position = calculate_optimal_target_position(positions[valid_positions_mask], robot_arm.target_position)
                robot_arm.set_target_pose(
                    target_position,
                    Rotation.from_matrix(initial_pose["rotation"]).as_euler("xyz"),
                )
                state = 1

            # If there are no valid positions and the robot arm is in the moving state, reset the robot arm to its initial pose
            elif not np.any(valid_positions_mask) and state == 1:
                robot_arm.set_target_pose(
                    initial_pose["position"],
                    Rotation.from_matrix(initial_pose["rotation"]).as_euler("xyz"),
                )
                state = 0

            robot_arm.update()
            mj_step(model, data)

            data.site("target").xpos = robot_arm.target_position
            model.site_rgba[data.site("target").id] = np.array([0, 1, 0, 1]) if state == 1 else np.array([1, 0, 0, 1])

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)