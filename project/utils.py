import numpy as np
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
