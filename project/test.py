import mujoco
import mujoco.viewer
import numpy as np
import time
from utils import (
    get_robot_workspace_mujoco,
    print_workspace_analysis,
    create_telemetry_storage,
    plot_robot_telemetry,
    plot_pid_controller_analysis
)
from orientation_controller import OrientationController

# Gewünschte Orientierung:
# Das ist die Einheitsquaternion: (1, 0, 0, 0)
target_quat = np.array([1.0, 0.0, 0.0, 0.0])
orient_ctrl = OrientationController(
    Kp_orient=[10.0, 10.0, 10.0],
    Kd_orient=[5.0, 5.0, 5.0],
    target_quat=target_quat
)

# Telemetrie
telemetry = create_telemetry_storage()

# Modell laden
xml_file = "world_no_table.xml"
model = mujoco.MjModel.from_xml_path(xml_file)
data = mujoco.MjData(model)

# Schwerkraft aktivieren
model.opt.gravity = np.array([0, 0, -9.81])

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'paddle')
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')

# Zielposition (erreichbar)
mujoco.mj_forward(model, data)
initial_paddle = data.xpos[body_id].copy()
print(f"Anfangsposition des Schlägers: {initial_paddle}")


workspace = get_robot_workspace_mujoco(xml_file)

# Definiere deine gewünschte Zielposition
raw_target = np.array([0.0, -3.0, 1.0]) 
# Clippe sie auf den Arbeitsraum
x_target = np.clip(raw_target,
                   [workspace['Total_Min_X'], workspace['Total_Min_Y'], workspace['Min_Z']],
                   [workspace['Total_Max_X'], workspace['Total_Max_Y'], workspace['Max_Z']]) # [0.000, -1.674, 1.010]
x_dot_target = np.array([0.0, 0.0, 0.0])

print(f"Raw Ziel: {raw_target}")
print(f"Erreichbares Ziel: {x_target}")

# PID-Gains
Kp = np.array([10.0, 10.0, 10.0])
Kd = np.array([5.0, 5.0, 5.0])
Ki = np.array([0.0, 0.0, 0.0])       # Integral aus
K_null = np.array([5, 5, 2, 2, 1, 1, 0.5, 0.5])

integral_error = np.zeros(3)
integral_limit = 2.0

# Arbeitsrauminfo (optional)
workspace = get_robot_workspace_mujoco(xml_file)
print_workspace_analysis(workspace, initial_paddle)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 1. Aktuelle Zustände
        mujoco.mj_forward(model, data)
        x_curr = data.xpos[body_id]
        jac_p = np.zeros((3, model.nv))
        jac_r = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jac_p, jac_r, body_id)
        x_dot_curr = jac_p @ data.qvel

        # 2. Fehler
        error_pos = x_target - x_curr
        error_vel = x_dot_target - x_dot_curr

        # 3. PID (nur PD)
        P_force = Kp * error_pos
        D_force = Kd * error_vel
        F_virtual = P_force + D_force
        F_virtual = np.clip(F_virtual, -200, 200)

        # 4. Task-Momente
        tau_task = jac_p.T @ F_virtual
        
        orient_ctrl.set_target_horizontal_from_current(data.xmat[body_id])
        tau_orient = orient_ctrl.compute_torque(data.xmat[body_id], jac_r, data.qvel)
        tau_task += tau_orient
        
        # 5. Gravitationskompensation mit mj_inverse
        #    Zuerst setzen wir die Beschleunigung auf Null (für statische Kompensation)
        data.qacc = np.zeros(model.nv)   # Beschleunigung = 0
        #    Rufe mj_inverse auf, um die benötigten Kräfte (inkl. Gravitation) zu berechnen
        mujoco.mj_inverse(model, data)
        #    Die inverse Dynamik liefert in data.qfrc_inverse die generalisierten Kräfte,
        #    die nötig sind, um die gewünschte Beschleunigung (hier 0) zu erreichen.
        #    Diese enthalten Gravitations-, Coriolis- und Zentrifugalkräfte.
        tau_gravity_comp = data.qfrc_inverse

        # 6. Nullraum-Dämpfung
        tau_null = -K_null * data.qvel
        
        # Deaktiviere paddle_joint (Index 7)
        #tau_task[7] = 0.0
        #tau_gravity_comp[7] = 0.0
        #tau_null[7] = 0.0
        
        # 7. Gesamtmoment
        tau_total = tau_task + tau_gravity_comp + tau_null

        # Begrenzung
        max_torque = np.array([200, 200, 100, 100, 50, 50, 20, 20])
        tau_total = np.clip(tau_total, -max_torque, max_torque)

        # Ansteuern
        data.ctrl[:] = tau_total

        # Telemetrie
        distance = np.linalg.norm(error_pos)
        telemetry["time"].append(data.time)
        telemetry["distance"].append(distance)
        telemetry["ctrl_signals"].append(np.copy(data.ctrl))
        telemetry["grav_comp"].append(np.copy(tau_gravity_comp))
        telemetry["task_torques"].append(np.copy(tau_task))
        telemetry["P_norm"].append(np.linalg.norm(P_force))
        telemetry["I_norm"].append(0.0)
        telemetry["D_norm"].append(np.linalg.norm(D_force))

        # Schritt
        mujoco.mj_step(model, data)
        viewer.sync()

        # Echtzeit-Sync
        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

# Nach Simulation
plot_robot_telemetry(telemetry)
plot_pid_controller_analysis(telemetry)