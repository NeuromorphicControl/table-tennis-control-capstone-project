from pathlib import Path
import time

import numpy as np
import mujoco
import mujoco.viewer

from predict import TrajectoryPredictor, StrikePlanner
from plotting import PlotManager, TrajectoryPlot
from world import Ball, Target
from robot import RobotArm

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
        vel += np.random.uniform(-1, 1, size=3)
        ball.set_velocity(vel)

if __name__ == "__main__":
    # Load the MuJoCo model and create a data object
    model = MjModel.from_xml_path("project/world.xml")
    data = MjData(model)

    # Create Ball and Target instances with the loaded data and joint name
    ball = Ball(model, data, ball_joint_name="ball_free")
    target = Target(model, data, target_site_name="target")

    # Create a RobotArm instance with the loaded data and joint names
    joint_names = ["base_x", "base_y", "rotator1", "rotator2", "arm1", "arm2", "paddle_rotator", "paddle"]
    robot_arm = RobotArm(model, data, joint_names, site_name="paddle_site", return_home=True, base_pos=(0.2, 0), dt=model.opt.timestep)

    robot_arm.set_task_gains(
            position_kp=[300.0, 300.0, 300.0],
            position_kd=[29.6, 29.6, 29.6],
            orientation_kp=[680.0, 680.0, 680.0],
            orientation_kd=[23.0, 23.0, 23.0],
            null_kp=60.0,
            null_kd=50.0, # Damping for the base joints to prevent oscillations
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
    data.site("target_pose").xpos = initial_pose["position"]

    # Set the ball to its initial position and velocity
    ball.reset_position((0, 0, 2))

    vel = (0, -3, 2)
    vel += np.random.uniform(-1, 1, size=3)
    ball.set_velocity(vel)

    # Create working area bounds for the robot arm based on its initial position and a predefined range
    working_area_bounds = ((working_x_min, working_x_max), (working_y_min, working_y_max), (working_z_min, working_z_max))

    # Initialize the PlotManager
    plot_manager = PlotManager(update_interval=0.5)
    plot_manager.add(TrajectoryPlot(ball, target, history_length=20))

    plot_data = {
        "dt": model.opt.timestep,
        "gravity_vector": model.opt.gravity,
        "p_start": None,
        "v_start": None,
        "p_paddle": None,
        "v_paddle": None,
        "pre_time": None,
        "post_time": None,
        "update_predictions": False,
    }

    # Initialize the TrajectoryPredictor and StrikePlanner with the ball, target, and working area bounds
    trajectory_predictor = TrajectoryPredictor(ball, dt=model.opt.timestep, t_max=10)
    strike_planner = StrikePlanner(target, trajectory_predictor, working_area_bounds=working_area_bounds)

    state = 0
    tolerance = 0.005 # Tolerance for when to update the robot arm's target position based on the ball's predicted trajectory

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Update the ball's position and velocity, and recalculate the predicted trajectory
            ball.update()
            trajectory_predictor.update()

            # Get the predicted trajectory of the ball
            times, positions, velocities = strike_planner.get_possible_strike_positions()

            # Calculate flight times for the ball to reach the target position from each of the valid positions and velocities
            flight_times = np.nanmax(strike_planner.calculate_flight_time(positions, velocities), axis=1)

            # Calculate the strike and paddle angles for the ball to reach the target position from each of the valid positions and velocities
            strike_angles = strike_planner.calculate_strike_angle(flight_times, positions)
            paddle_normals = strike_planner.calculate_paddle_normal(strike_angles, velocities)

            # Calculate the shortest flight time for the ball to reach the target position
            total_flight_time = times + flight_times

            # Select the arm position and velocity corresponding to the shortest flight time
            optimal_index = np.argmin(total_flight_time) if len(total_flight_time) > 0 and np.isfinite(np.max(total_flight_time)) else None
            
            optimal_flight_time = flight_times[optimal_index]
            optimal_position = positions[optimal_index]
            optimal_velocity = velocities[optimal_index]
            optimal_paddle_normal = paddle_normals[optimal_index]
            
            # A boolean to check if the ball is flying towards the robot arm
            ball_flying_towards_arm = ball.get_velocity()[1] < 0 if optimal_index is not None else False

            plot_data["time"] = data.time

            # If there are valid positions, the ball flies towards the arm and the robot arm is in the waiting state or the target position has changed, calculate the optimal target position and set the robot arm's target pose
            if optimal_index is not None and ball_flying_towards_arm and (state == 0 or np.linalg.norm(optimal_position - robot_arm.target_position) > tolerance):
                # Save current ball position (as start for prediction) and velocity (as start for prediction) to the data dictionary
                plot_data["p_start"] = ball.get_position().copy()
                plot_data["v_start"] = ball.get_velocity().copy()
                plot_data["p_paddle"] = optimal_position.copy()
                plot_data["v_paddle"] = strike_angles[optimal_index].copy()
                plot_data["pre_time"] = times[optimal_index]
                plot_data["post_time"] = optimal_flight_time

                # Paddle's normal in its local frame (adjust if your model uses a different axis)
                n_local = np.array([0.0, 1.0, 0.0])

                # Current orientation
                R_current = Rotation.from_matrix(robot_arm.get_site_pose()["rotation"])

                # Current normal in world coordinates
                n_current = R_current.apply(n_local)
                n_current /= np.linalg.norm(n_current)

                # Compute smallest rotation from current normal to target normal
                axis = np.cross(n_current, optimal_paddle_normal)
                axis_norm = np.linalg.norm(axis)
                dot = np.clip(np.dot(n_current, optimal_paddle_normal), -1.0, 1.0)

                if axis_norm < 1e-8:
                    if dot > 0.999999:
                        # Already aligned
                        R_delta = Rotation.identity()
                    else:
                        # Normals are opposite; rotate 180° about any perpendicular axis
                        perp = np.array([1.0, 0.0, 0.0])
                        if abs(np.dot(perp, n_current)) > 0.9:
                            perp = np.array([0.0, 1.0, 0.0])
                        axis = np.cross(n_current, perp)
                        axis /= np.linalg.norm(axis)
                        R_delta = Rotation.from_rotvec(np.pi * axis)
                else:
                    axis /= axis_norm
                    angle = np.arccos(dot)
                    R_delta = Rotation.from_rotvec(angle * axis)

                # Apply the rotation
                R_new = R_delta * R_current

                # Convert back to Euler angles (xyz convention)
                euler_new = R_new.as_euler("xyz", degrees=False)
                
                # Update the robot arm's target position based on the ball's predicted trajectory
                robot_arm.set_target_pose(
                    optimal_position,
                    euler_new,
                )
                state = 1

            # # If there are no valid positions and the robot arm is in the moving state, reset the robot arm to its initial pose
            # elif (optimal_index is None or not ball_flying_towards_arm) and state == 1:
            #     robot_arm.set_target_pose(
            #         initial_pose["position"],
            #         Rotation.from_matrix(initial_pose["rotation"]).as_euler("xyz"),
            #     )
            #     state = 0

            robot_arm.update()
            mj_step(model, data)

            data.site("target_pose").xpos = robot_arm.target_position
            model.site_rgba[data.site("target_pose").id] = np.array([0, 1, 0, 1]) if state == 1 else np.array([1, 0, 0, 1])

            # Update the plots with the latest data
            plot_manager.update(plot_data)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)