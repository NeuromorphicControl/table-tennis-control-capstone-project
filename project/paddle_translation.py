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

# Setup Analyse
telemetry = create_telemetry_storage()


# 1. Modell laden
xml_file = "world.xml"
model = mujoco.MjModel.from_xml_path(xml_file)
data = mujoco.MjData(model)

# 2. IDs aus dem Modell auslesen
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'paddle')
current_pos = data.xpos[body_id]
print(f"Aktuelle Position: {current_pos}")

# 3. Tuning-Parameter
Kp = np.array([10.0, 10.0, 10.0])  # Proportional
Kd = np.array([5.0, 5.0, 5.0])    # Derivativ
Ki = np.array([0.0, 0.0, 0.0])      # Integral
K_null = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 500.0, 500.0])
Kd_rot = np.array([100.0, 100.0, 100.0]) 

# Integral Speicher
integral_error = np.zeros(3)            
integral_limit = 2.0  

# 4. Deine gewünschte Zielposition im Raum [X, Y, Z]

# --- INFOS ---
workspace = get_robot_workspace_mujoco(xml_file, is_string=False)
paddel_pos = data.xpos[body_id]

# Aktuelle Position als Ziel setzen (Test)
min_x, max_x = workspace['Total_Min_X'], workspace['Total_Max_X']
min_y, max_y = workspace['Total_Min_Y'], workspace['Total_Max_Y']
min_z, max_z = workspace['Min_Z'], workspace['Max_Z']

x_target = np.array([
    (min_x + max_x) / 2,
    (min_y + max_y) / 2,
    (min_z + max_z) / 2
])# Ziel = aktuelle Position  #[0.8, -1.0, 0.6]
print(f"Ziel in der Mitte des Arbeitsraums: {x_target}")
x_dot_target = np.array([0.0, 0.0, 0.0]) # Stoppen am Ziel

# 5. Simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    print_workspace_analysis(workspace=workspace, paddel_pos=paddel_pos)

    
    while viewer.is_running():
        step_start = time.time()
        
        # --- PHYSIK-ZUSTAND AKTUALISIEREN & GRAVITATION ERMITTELN ---
        mujoco.mj_forward(model, data)
        tau_gravity = data.qfrc_gravcomp
        print(f"Gravitation: {tau_gravity}")

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
        
        P_force = Kp * error_pos
        D_force = Kd * error_vel
        dt = model.opt.timestep  # Zeitschritt der Simulation
        
        # Aktiviere das Integral erst, wenn der Fehler klein ist (< 0.1 m)
        if np.linalg.norm(error_pos) < 0.1:
            integral_error += error_pos * dt
        else:
            integral_error = np.zeros(3)  # Integral zurücksetzen bei großen Fehlern
        
        integral_error = np.clip(integral_error, -integral_limit, integral_limit) #Begrenze den akkumulierten Fehler, damit das Integral nicht explodiert
        I_force = Ki * integral_error
        
        # --- GESAMTE VIRTUELLE KRAFT ---
        F_virtual = P_force + I_force + D_force  # (3,)
        
        max_force = 10.0
        F_virtual = np.clip(F_virtual, -max_force, max_force)

        # --- 3. KRAFT IN MOMENTE UMRECHNEN (Transponierte Jacobi) ---
        tau_task = jac_p.T @ F_virtual 

        # --- ORIENTIERUNGSDÄMPFUNG (gegen das Drehen) --- (hier dann vollstädneige invere kinematik für orientiertung einbauen)
        ang_vel_curr = jac_r @ data.qvel
        torque_virtual_rot = -Kd_rot * ang_vel_curr
        tau_rot = jac_r.T @ torque_virtual_rot
        
        tau_task += tau_rot
        
        # --- 4. MOTOREN ANSTEUERN ---
        tau_null = -K_null * data.qvel   # data.qvel ist (8,)
        # Gesamtes Drehmoment = Bewegungskraft + Schwerkraft-Ausgleich
        tau_total = tau_task + tau_gravity + tau_null
        
        # Motorkraft begrenzen
        max_torque = np.array([500.0, 500.0, 200.0, 200.0, 100.0, 100.0, 50.0, 50.0]) # np.array([50.0, 50.0, 20.0, 20.0, 10.0, 10.0, 5.0, 5.0])
        print(f"Clip limits: {max_torque}")  # Muss einmalig erscheinen
        tau_total = np.clip(tau_total, -max_torque, max_torque)
        if np.max(np.abs(tau_total)) > 1.0:  # nur bei relevanten Werten
            print(f"tau_total max (nach Clip): {np.max(np.abs(tau_total)):.2f}")
        
        # Wir holen uns die Getriebewerte (gear) aller aktiven Motoren aus dem Modell
        # model.actuator_gear[:, 0] enthält die primären Faktoren
        gears = model.actuator_gear[:, 0]
        data.ctrl[:] = tau_total / gears
        
        # --- 📈 DATEN FÜR DIE ANALYSE SPEICHERN ---
        abstand = np.linalg.norm(error_pos)
        
        telemetry["time"].append(data.time)
        telemetry["distance"].append(abstand)
        telemetry["ctrl_signals"].append(np.copy(data.ctrl))
        telemetry["grav_comp"].append(np.copy(tau_gravity))
        telemetry["task_torques"].append(np.copy(tau_task))
        telemetry["P_norm"].append(np.linalg.norm(P_force))
        telemetry["I_norm"].append(np.linalg.norm(I_force))
        telemetry["D_norm"].append(np.linalg.norm(D_force))

        # Schritt in der Physik-Engine ausführen
        mujoco.mj_step(model, data)

        # Viewer updaten
        viewer.sync()

        # Timing synchronisieren, damit die Simulation in Echtzeit läuft
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


# NACHDEM DER VIEWER GESCHLOSSEN WURDE: Diagramm anzeigen
print("\n[Info] Simulation beendet. Generiere Plots zur Fehleranalyse...")
plot_robot_telemetry(telemetry)
print("\n[Info] Simulation beendet. Generiere PID-Analyse...")
plot_pid_controller_analysis(telemetry)