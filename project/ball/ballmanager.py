import mujoco
import numpy as np
from typing import List, Optional, Tuple

class BallManager:
    def __init__(self, spec: mujoco.MjSpec, collision_geoms: Optional[List[str]] = None):
        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.ball_id = 0
        if collision_geoms is None:
            collision_geoms = ["table_collider_v", "table_collider_h", "ground"]
        self.collision_geoms = collision_geoms
        self._balls = []

    @classmethod
    def from_file(cls, xml_path: str, collision_geoms=None):
        spec = mujoco.MjSpec.from_file(xml_path)
        return cls(spec, collision_geoms)

    def create_ball_id(self) -> int:
        current_id = self.ball_id
        self.ball_id += 1
        return current_id
        
    def add_ball(self, pos: List[float], vel: List[float]) -> None:
        pos = pos or [0.0, 0.0, 2.0]
        vel = vel or [0.0, 0.0, 0.0]

        ball_idx = self.create_ball_id()
        body_name = f"ball_body_{ball_idx}"
        geom_name = f"ball_geom_{ball_idx}"
        joint_name = f"ball_joint_{ball_idx}"

        body = self.spec.worldbody.add_body(
            name=body_name,
            pos=pos,
            quat=[0.0, 0.0, 0.0, 0.0],
            mass=0.00167
        )
        geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            name=geom_name,
            size=[0.02, 0.0, 0.0],
            rgba=[1.0, 0.5, 0.0, 1.0]
        )
        joint = body.add_joint(
            name=joint_name,
            type=mujoco.mjtJoint.mjJNT_FREE,
            damping=0.0
        )

        pair_objects = []
        for g1 in self.collision_geoms:
            try:
                pair = self.spec.add_pair()
                pair.name = f"pair_{ball_idx}_{g1}"
                pair.geomname1 = g1
                pair.geomname2 = geom_name
                pair_objects.append(pair)
            except Exception:
                print(f"Warnung: Kollisionspartner '{g1}' nicht gefunden – Paar übersprungen.")

        self.model, self.data = self.spec.recompile(self.model, self.data)

        #
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qveladr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[qveladr:qveladr + 3] = vel

        self._balls.append({
            "body": body,
            "geom": geom,
            "joint": joint,
            "pairs": pair_objects,
            "body_name": body_name,
            "geom_name": geom_name,
            "joint_name": joint_name,
            "pos": pos,
            "vel": vel
        })

    def remove_last_ball(self) -> bool:
            if not self._balls:
                print("Kein Ball vorhanden.")
                return False
            ball_info = self._balls.pop()
            # Ball deaktivieren (nicht löschen) -- muss mir genau anschauen wie das geht
            joint_name = ball_info["joint_name"]
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qposadr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qposadr:qposadr+3] = [0.0, 0.0, -10.0]   # weit unter den Tisch
                qveladr = self.model.jnt_dofadr[joint_id]
                self.data.qvel[qveladr:qveladr+3] = [0.0, 0.0, 0.0]
            except Exception as e:
                print(f"Fehler beim Deaktivieren des Balls: {e}")
            # Geometrie unsichtbar machen
            # geom = ball_info["geom"]
            # geom.rgba = [0.0, 0.0, 0.0, 0.0]   # transparent
            return True

    def remove_last_ball_01(self) -> bool:
        if not self._balls:
            print("Kein Ball vorhanden.")
            return False
        ball_info = self._balls.pop()

        body_name = ball_info["body_name"]
        geom_name = ball_info["geom_name"]
        joint_name = ball_info["joint_name"]

        # IDs im kompilierten Modell ermitteln
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        except Exception as e:
            print(f"Fehler beim Finden der IDs: {e}")
            return False

        # 1. Teleportieren (weit weg + Geschwindigkeit = 0)
        qposadr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qposadr:qposadr + 3] = [0.0, 0.0, -100.0]
        qveladr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[qveladr:qveladr + 3] = [0.0, 0.0, 0.0]

        # 2. Unsichtbar machen (rgba)
        self.model.geom_rgba[geom_id, 0] = 0.0
        self.model.geom_rgba[geom_id, 1] = 0.0
        self.model.geom_rgba[geom_id, 2] = 0.0
        self.model.geom_rgba[geom_id, 3] = 0.0

        # 3. Kollision deaktivieren
        self.model.geom_contype[geom_id] = 0
        self.model.geom_conaffinity[geom_id] = 0

        # 4. Masse auf Null setzen (physikalisch irrelevant)
        self.model.body_mass[body_id] = 0.0
        
        return True

    def get_model_data(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        return self.model, self.data