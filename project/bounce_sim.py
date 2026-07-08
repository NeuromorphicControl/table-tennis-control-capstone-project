import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mujoco

xml = """
<mujoco model="tabletennis">
  <option cone="elliptic" timestep="1e-5" integrator="implicitfast" gravity="0 0 -9.81"/>

  <visual>
    <global elevation="30" azimuth="-45"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
    <texture name="ball_tex" type="cube" builtin="checker" rgb1=".9 .9 .9" rgb2="1 .6 .1" width="300" height="300"/>
    <material name="ball_mat" texture="ball_tex" texrepeat="4 4"/>
    <texture name="table_tex" type="2d" builtin="checker" rgb1=".1 .3 .1" rgb2=".2 .4 .2" width="300" height="300"/>
    <material name="table_mat" texture="table_tex" texrepeat="4 4" reflectance=".3"/>
    <material name="floor_mat" reflectance="0.2" rgba="0.2 0.3 0.4 1"/>
  </asset>

  <default>
    <!-- Weicherer Kontakt: 10 ms Zeitkonstante, 0.001 Dämpfung -->
    <pair solref="0.01 0.001"/>
  </default>

  <worldbody>
    <!-- Boden: weit unten, nur als Sicherheit -->
    <geom name="floor" type="box" size="5 5 0.5" pos="0 0 -0.6" material="floor_mat" friction="0.5"/>

    <!-- Tischplatte mit realistischer Dicke (0.025 m) -->
    <geom name="table" type="box" size="1.37 0.7625 0.0125" pos="0 0 0.78"
          material="table_mat" friction="0.8 0.01 0.001"/>

    <!-- Ball: leichter seitlicher Versatz (0.05 m), Höhe 1.0 m -->
    <body name="ball" pos="0.05 0 1.0">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.02" material="ball_mat" mass="0.0027"/>
    </body>

    <light pos="0 0 1.5" directional="false"/>
    <light pos="-1 -1 1" dir="1 1 -1" directional="true"/>
  </worldbody>

  <contact>
    <pair geom1="table" geom2="ball_geom"/>
    <pair geom1="floor" geom2="ball_geom"/>
  </contact>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Initialbedingungen
data.qpos[:] = [0.05, 0, 1.78, 1, 0, 0, 0]
data.qvel[:] = 0

# Simulation: 3 Sekunden sollten reichen
duration = 3.0
n_steps = int(duration / model.opt.timestep)
times = np.zeros(n_steps)
heights = np.zeros(n_steps)
velocities = np.zeros(n_steps)

ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

for i in range(n_steps):
    mujoco.mj_step(model, data)
    times[i] = data.time
    heights[i] = data.xpos[ball_body_id][2]
    velocities[i] = data.qvel[2]

    # Ausgabe alle 5000 Schritte
    if i % 5000 == 0:
        print(f"t={data.time:.4f}s  h={heights[i]:.6f}m  v={velocities[i]:.3f}m/s")

# Nach der Simulation prüfen, ob der Ball auf dem Tisch liegt
final_height = heights[-1]
print(f"\nEndhöhe: {final_height:.6f} m (Tischhöhe: 0.78 m)")
if abs(final_height - 0.78) < 0.001:
    print("✅ Ball liegt korrekt auf dem Tisch.")
else:
    print("⚠️ Ball schwebt oder ist abgewichen – Kontaktparameter optimieren.")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(times, heights, linewidth=1.5)
plt.xlabel("Zeit [s]")
plt.ylabel("Höhe über Boden [m]")
plt.title("Tischtennisball – Fall und abklingendes Bouncen")
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(y=0.78, color='r', linestyle=':', label='Tischhöhe (0.78 m)')
plt.legend()
plt.tight_layout()
plt.savefig("bounce_plot.png", dpi=150)
print("✅ Plot gespeichert als 'bounce_plot.png'")