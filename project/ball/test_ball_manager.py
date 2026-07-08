import mujoco
import time
import os
import gc
from ballmanager import BallManager

    """Dieses Script wurde weitestgehend von AI gebaut, um die ballmanager klasse zu testen.
    """

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    xml_path = os.path.join(parent_dir, "world.xml")

    manager = BallManager.from_file(xml_path)
    model, data = manager.get_model_data()

    viewer = mujoco.viewer.launch_passive(model, data)

    action_times = [5.0, 8.0, 11.0, 14.0]
    action_index = 0
    print("Simulation gestartet. Drücke Strg+C zum Beenden.")

    # Hilfsfunktion zum Ausgeben der Ballposition (über Body)
    def print_ball_position(body_name="ball_body_0"):
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            pos = data.xpos[body_id]   # (3,) Array
            print(f"Ballposition (x,y,z): {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")
        except Exception as e:
            print(f"Fehler beim Auslesen der Position: {e}")

    steps_per_frame = 500   # 500 * 1e-5 = 0.005 s pro Frame

    try:
        while viewer.is_running():
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data, nstep=1)
            viewer.sync()
            current_time = data.time

            if int(current_time * 2) % 2 == 0:
                print_ball_position()

            if action_index < len(action_times) and current_time >= action_times[action_index]:
                if action_index == 0:
                    manager.add_ball(pos=[0.0, 0.0, 3.0], vel=[0.5, 0.0, 0.0])
                    print(f"[{current_time:.2f}s] Ball 1 hinzugefügt")
                    model, data = manager.get_model_data()
                    viewer.close()
                    del viewer
                    gc.collect()
                    time.sleep(0.5)   # Warten, bis der Viewer geschlossen ist
                    viewer = mujoco.viewer.launch_passive(model, data)
                elif action_index == 1:
                    manager.add_ball(pos=[0.5, 0.0, 3.0], vel=[-0.5, 0.2, 0.0])
                    print(f"[{current_time:.2f}s] Ball 2 hinzugefügt")
                    model, data = manager.get_model_data()
                    viewer.close()
                    del viewer
                    gc.collect()
                    time.sleep(0.5)
                    viewer = mujoco.viewer.launch_passive(model, data)
                elif action_index == 2:
                    manager.remove_last_ball()
                    print(f"[{current_time:.2f}s] Ball 2 entfernt")
                    model, data = manager.get_model_data()
                    viewer.close()
                    del viewer
                    gc.collect()
                    time.sleep(0.5)
                    viewer = mujoco.viewer.launch_passive(model, data)
                elif action_index == 3:
                    manager.remove_last_ball()
                    print(f"[{current_time:.2f}s] Ball 1 entfernt")
                    model, data = manager.get_model_data()
                    viewer.close()
                    del viewer
                    gc.collect()
                    time.sleep(0.5)
                    viewer = mujoco.viewer.launch_passive(model, data)
                action_index += 1

            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nBeendet.")
    finally:
        viewer.close()

if __name__ == "__main__":
    main()