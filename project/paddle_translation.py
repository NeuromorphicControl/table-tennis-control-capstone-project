import mujoco
import mujoco.viewer
import numpy as np
import time
from utils import get_robot_workspace_mujoco, print_workspace_analysis

# 1. Modell laden
xml_file = "world.xml"
model = mujoco.MjModel.from_xml_path(xml_file)
data = mujoco.MjData(model)

# 2. IDs aus dem Modell auslesen
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'paddle')

# 3. Tuning-Parameter
Kp = np.array([100.0, 100.0, 100.0])  # Proportional
Kd = np.array([10.0, 10.0, 10.0])    # Derivativ

# 4. Deine gewünschte Zielposition im Raum [X, Y, Z]
x_target = np.array([0.6, 0.3, 0.4]) 
x_dot_target = np.array([0.0, 0.0, 0.0]) # Stoppen am Ziel

# 5. Simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # --- INFOS ---
    workspace = get_robot_workspace_mujoco(xml_file, is_string=False)
    paddel_pos = data.xpos[body_id]
    
    print_workspace_analysis(workspace=workspace, paddel_pos=paddel_pos)

    
    while viewer.is_running():
        step_start = time.time()

        # --- 1. AKTUELLE WERTE AUSLESEN ---
        x_curr = data.xpos[body_id] # Aktuelle 3D-Position des Schlägers
        
        # Jacobi-Matrizen für Gelenke (8) initialisieren (Form: 3 Zeilen x 8 Spalten)
        jac_p = np.zeros((3, model.nv)) # translation ((x,y,z),8)
        jac_r = np.zeros((3, model.nv)) # rotation ((x,y,z),8)
        
        # Berechnung der Jacobi-Matrix (3x8) für die Schläger-Position
        mujoco.mj_jacBody(model, data, jac_p, jac_r, body_id)
        
        # Aktuelle Geschwindigkeit des Schlägers berechnen
        x_dot_curr = jac_p @ data.qvel # (3, 8) ⨯ (8,) --> (3,)

        # --- 2. VIRTUELLE KRAFT BERECHNEN (Arbeitsraum-PD) ---
        error_pos = x_target - x_curr      # (3,)
        error_vel = x_dot_target - x_dot_curr  # (3,)
        F_virtual = Kp * error_pos + Kd * error_vel  # (3,)

        # --- 3. KRAFT IN MOMENTE UMRECHNEN (Transponierte Jacobi) ---
        tau_task = jac_p.T @ F_virtual 

        # --- 4. GRAVITATION AUSGLEICHEN ---
        # Berechnet alle passiven Kräfte (Schwerkraft & Zentrifugalkräfte)
        mujoco.mj_forward(model, data)
        tau_gravity = data.qfrc_gravcomp

        # --- 5. MOTOREN ANSTEUERN ---
        # Gesamtes Drehmoment = Bewegungskraft + Schwerkraft-Ausgleich
        tau_total = tau_task + tau_gravity
        
        # Da du 8 Motoren für 8 Gelenke hast, passt das Array direkt (8 Elemente)
        data.ctrl[:] = tau_total

        # Schritt in der Physik-Engine ausführen
        mujoco.mj_step(model, data)

        # Viewer updaten
        viewer.sync()

        # Timing synchronisieren, damit die Simulation in Echtzeit läuft
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
