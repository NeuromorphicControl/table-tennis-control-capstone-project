import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import mujoco


def get_robot_workspace_mujoco(xml_path_or_string, is_string=False):
    """
    Nutzt die MuJoCo API, um die Gelenkgrenzen der Basis sowie 
    den maximalen Rotationsradius des Arms (bis zum Paddle) zu berechnen.
    """
    # 1. Modell über MuJoCo laden
    if is_string:
        model = mujoco.MjModel.from_xml_string(xml_path_or_string)
    else:
        model = mujoco.MjModel.from_xml_path(xml_path_or_string)
    
    data = mujoco.MjData(model)
    
    # 2. Berechne die Kinematik einmalig im Standardzustand (Arme gestreckt/Nullstellung)
    mujoco.mj_forward(model, data)
    
    # 3. Positionen der Basis und des Paddle auslesen
    # MuJoCo liefert uns hier direkt die globalen 3D-Koordinaten (xpos)
    base_xyz = data.body('base').xpos
    paddle_xyz = data.body('paddle').xpos
    
    # 3D-Gesamtlänge des Arms (Abstand von Basis zum Schläger im Raum)
    arm_length_3d = np.linalg.norm(paddle_xyz - base_xyz)
    # Der maximale Rotationsradius ist die euklidische Distanz in der X-Y Ebene
    arm_length_xy = np.linalg.norm(paddle_xyz[:2] - base_xyz[:2])
    
    # 4. Lineare Grenzen der Basis-Gelenke auslesen
    # Wir suchen nach den IDs der Schiebegelenke
    slide_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'base_slide_x')
    slide_y_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'base_slide_y')
    
    # Grenzen (range) aus dem Modell auslesen [min, max]
    range_slide_x = model.jnt_range[slide_x_id]
    range_slide_y = model.jnt_range[slide_y_id]
    
    # Da slide_x auf "0 1 0" (Y) wirkt und slide_y auf "1 0 0" (X):
    min_x = base_xyz[0] + range_slide_y[0]
    max_x = base_xyz[0] + range_slide_y[1]
    
    min_y = base_xyz[1] + range_slide_x[0]
    max_y = base_xyz[1] + range_slide_x[1]
    
    max_z = base_xyz[2] + arm_length_3d
    min_z = base_xyz[2] - arm_length_3d
    
    return {
        "Base_X_Range": (min_x, max_x),
        "Base_Y_Range": (min_y, max_y),
        "Max_Arm_Radius": arm_length_xy,
        "Total_Max_X": max_x + arm_length_xy,
        "Total_Min_X": min_x - arm_length_xy,
        "Total_Max_Y": max_y + arm_length_xy,
        "Total_Min_Y": min_y - arm_length_xy,
        "Min_Z": min_z,
        "Max_Z": max_z
    }


def print_workspace_analysis(workspace, paddel_pos):
    """
    Gibt die berechneten Reichweiten und die aktuelle Position des Roboters
    sauber und strukturiert in der Konsole aus.
    """
    print("=" * 55)
    print(" ROBOTER ARBEITSRAUM ANALYSE ".center(55, "="))
    print("=" * 55)

    print("\n📍 AKTUELLE POSITION:")
    print(f"  • Schläger-Anfangsposition (X, Y, Z): [{paddel_pos[0]:.3f}, {paddel_pos[1]:.3f}, {paddel_pos[2]:.3f}]")

    print("\n📏 KINEMATIK & REICHWEITE:")
    print(f"  • Maximaler Rotationsradius (Arm):   {workspace['Max_Arm_Radius']:.3f} Einheiten")

    print("\n🌐 ABSOLUTE GRENZEN (Arbeitsraum):")
    print(f"  • X-Achse (Min / Max):              [{workspace['Total_Min_X']:.3f} / {workspace['Total_Max_X']:.3f}]")
    print(f"  • Y-Achse (Min / Max):              [{workspace['Total_Min_Y']:.3f} / {workspace['Total_Max_Y']:.3f}]")
    print(f"  • Z-Achse (Min / Max Höhe):         [{workspace['Min_Z']:.3f} / {workspace['Max_Z']:.3f}]")

    print("\n📊 DIMENSIONEN DES ARBEITSRAUMS:")
    x_spanne = workspace['Total_Max_X'] - workspace['Total_Min_X']
    y_spanne = workspace['Total_Max_Y'] - workspace['Total_Min_Y']
    z_spanne = workspace['Max_Z'] - workspace['Min_Z']
    print(f"  • Gesamtbreite in X-Richtung:        {x_spanne:.3f} Einheiten")
    print(f"  • Gesamttiefe in Y-Richtung:         {y_spanne:.3f} Einheiten")
    print(f"  • Maximale Höhenflexibilität (Z):    {z_spanne:.3f} Einheiten")

    print("=" * 55)


def create_telemetry_storage():
    """Erstellt ein leeres Dictionary, um die Simulationsdaten zu sammeln."""
    return {
        "time": [],
        "distance": [],
        "P_norm": [],  
        "D_norm": [],
        "I_norm": [],  
        "target_pos": [],
        "current_pos": [],
        "ctrl_signals": [],
        "grav_comp": [],
        "task_torques": []
    }


def plot_robot_telemetry(data_log, dateiname="fehleranalyse_roboter.png"):
    """
    Verbesserte Darstellung der gesammelten Telemetriedaten.
    Zeigt:
      - Abstand zum Ziel über der Zeit
      - Alle Ctrl-Signale (Motorbefehle) überlagert
      - Task- und Gravitationsanteile für ausgewählte Gelenke
    """
    time_arr = np.array(data_log["time"])
    ctrl_arr = np.array(data_log["ctrl_signals"])    # (n_steps, 8)
    grav_arr = np.array(data_log["grav_comp"])       # (n_steps, 8)
    task_arr = np.array(data_log["task_torques"])    # (n_steps, 8)
    dist_arr = np.array(data_log["distance"])

    # Figure mit 3 übereinanderliegenden Subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("Robot Control Telemetry", fontsize=16)

    # --- Plot 1: Abstand zum Ziel ---
    ax1 = axes[0]
    ax1.plot(time_arr, dist_arr, color='red', linewidth=2, label='Abstand zum Ziel')
    ax1.set_ylabel('Distanz [m]')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    ax1.set_title('Regelfehler über der Zeit')

    # --- Plot 2: Alle Ctrl-Signale ---
    ax2 = axes[1]
    # Verwende eine schöne Farbpalette für 8 Gelenke
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i in range(8):
        ax2.plot(time_arr, ctrl_arr[:, i], color=colors[i], linewidth=1.5, label=f'Gelenk {i}')
    ax2.set_ylabel('Ctrl-Signal')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)  # Legende außerhalb
    ax2.set_title('Alle Motorsignale (ctrl)')

    # --- Plot 3: Task- und Gravitationsanteile für zwei wichtige Gelenke ---
    ax3 = axes[2]
    # Wähle Gelenk 2 (Arm1) und Gelenk 6 (Paddle-Rotator) – repräsentativ
    joints_to_show = [2, 6]
    for j in joints_to_show:
        ax3.plot(time_arr, task_arr[:, j], linestyle='-', linewidth=1.5,
                 label=f'Task Gelenk {j}')
        ax3.plot(time_arr, grav_arr[:, j], linestyle='--', linewidth=1.5,
                 label=f'Grav Gelenk {j}')
    ax3.set_xlabel('Zeit [s]')
    ax3.set_ylabel('Drehmoment / Kraft')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax3.set_title('Task‑ vs. Gravitationsanteile für ausgewählte Gelenke')

    # Layout anpassen, damit nichts überlappt
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Platz für Legenden

    # Speichern
    dateiname = "output/" + dateiname
    plt.savefig(dateiname, dpi=300, bbox_inches='tight')
    print(f"\n[Erfolg] Verbesserte Grafik gespeichert als '{dateiname}'")
    plt.close()
    
    
def plot_pid_controller_analysis(data_log, dateiname="pid_controller_analysis.png"):
    """
    Zeigt den kompletten PID-Regler in Aktion:
    - Oben: Abstand zum Ziel (Regelfehler)
    - Mitte: P-, I- und D-Anteil im Vergleich (als Norm)
    - Unten: Gesamte virtuelle Kraft (F_virtual) mit allen Anteilen
    """
    time_arr = np.array(data_log["time"])
    dist_arr = np.array(data_log["distance"])
    P_arr = np.array(data_log["P_norm"])
    D_arr = np.array(data_log["D_norm"])
    I_arr = np.array(data_log["I_norm"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("PID-Regler Analyse – P, I und D im Zusammenspiel", fontsize=16)

    # --- Plot 1: Abstand zum Ziel ---
    ax1 = axes[0]
    ax1.plot(time_arr, dist_arr, color='red', linewidth=2)
    ax1.set_ylabel('Abstand [m]')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title('Regelfehler (soll → 0)')

    # --- Plot 2: P-, I- und D-Anteil ---
    ax2 = axes[1]
    ax2.plot(time_arr, P_arr, label='P-Anteil (Feder)', color='blue', linewidth=2)
    ax2.plot(time_arr, I_arr, label='I-Anteil (Ausgleich)', color='green', linewidth=2)
    ax2.plot(time_arr, D_arr, label='D-Anteil (Dämpfer)', color='orange', linewidth=2)
    ax2.set_ylabel('Kraft / Moment [N]')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    ax2.set_title('PID-Regleranteile (Norm der Vektoren)')

    # --- Plot 3: Summe F_virtual (P + I + D) ---
    F_arr = P_arr + I_arr + D_arr
    ax3 = axes[2]
    ax3.plot(time_arr, F_arr, label='F_virtual (P + I + D)', color='purple', linewidth=2)
    ax3.plot(time_arr, P_arr, label='P', color='blue', linestyle='--', alpha=0.5)
    ax3.plot(time_arr, I_arr, label='I', color='green', linestyle='--', alpha=0.5)
    ax3.plot(time_arr, D_arr, label='D', color='orange', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Zeit [s]')
    ax3.set_ylabel('Kraft [N]')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper right')
    ax3.set_title('Gesamte virtuelle Kraft = P + I + D')

    plt.tight_layout()
    dateiname = "output/" + dateiname
    plt.savefig(dateiname, dpi=300)
    print(f"\n[Erfolg] PID-Analyse gespeichert als '{dateiname}'")
    plt.close()