import numpy as np
import mujoco

class OrientationController:
    """
    PD-Regler für die Orientierung eines Körpers im Raum.
    Verwendet Quaternionen zur Fehlerberechnung.
    """
    def __init__(self, Kp_orient, Kd_orient, target_quat=None):
        self.Kp = np.asarray(Kp_orient)
        self.Kd = np.asarray(Kd_orient)
        self.target_quat = target_quat

    def set_target_orientation(self, quat):
        """Setzt die gewünschte Quaternion (w, x, y, z)."""
        self.target_quat = np.asarray(quat)

    def set_target_horizontal_from_current(self, xmat):
        """
        Setzt die Zielquaternion so, dass die lokale z-Achse des Körpers
        auf die Welt-z-Achse (0,0,1) zeigt. Die Drehung um die z-Achse bleibt erhalten.
        """
        rotmat = xmat.reshape(3, 3)
        # lokale z-Achse im Weltkoordinatensystem (dritte Spalte der Rotationsmatrix)
        local_z = rotmat[:, 2]
        target_z = np.array([0, 0, 1])
        axis = np.cross(local_z, target_z)
        norm = np.linalg.norm(axis)
        if norm > 1e-8:
            axis = axis / norm
            angle = np.arccos(np.clip(np.dot(local_z, target_z), -1, 1))
            q = np.array([np.cos(angle/2),
                          axis[0]*np.sin(angle/2),
                          axis[1]*np.sin(angle/2),
                          axis[2]*np.sin(angle/2)])
        else:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        self.target_quat = q

    def compute_torque(self, xmat, jac_r, qvel):
        """
        Berechnet das Gelenkmoment, das den Körper in die Zielorientierung bringt.

        Args:
            xmat (np.ndarray): (9,) – Rotationsmatrix des Körpers (row-major, wie von MuJoCo).
            jac_r (np.ndarray): (3, nv) – Rotations-Jacobi-Matrix.
            qvel (np.ndarray): (nv,) – aktuelle Gelenkgeschwindigkeiten.

        Returns:
            np.ndarray: (nv,) – Gelenkmomente für die Orientierungsregelung.
        """
        if self.target_quat is None:
            raise ValueError("Zielquaternion nicht gesetzt. Verwende set_target_orientation() oder set_target_horizontal_from_current().")

        # Aktuelle Orientierung aus der Rotationsmatrix (3x3) in Quaternion umwandeln
        rotmat = xmat.reshape(3, 3)
        current_quat = self._mat2quat(rotmat)  # (w, x, y, z)

        # Fehlerquaternion: q_err = q_target * inv(q_current)
        q_inv = np.array([current_quat[0], -current_quat[1], -current_quat[2], -current_quat[3]])
        q_err = self._quat_multiply(self.target_quat, q_inv)

        # Vektor-Teil des Fehlerquaternions (für kleine Winkel: ≈ 2 * Vektor)
        error_orient = 2 * q_err[1:]  # (3,)

        # Winkelgeschwindigkeit des Körpers
        ang_vel = jac_r @ qvel  # (3,)

        # PD-Regelung im Arbeitsraum (Drehmoment)
        torque_world = self.Kp * error_orient - self.Kd * ang_vel

        # Rücktransformation in Gelenkkoordinaten
        tau_orient = jac_r.T @ torque_world
        return tau_orient

    @staticmethod
    def _quat_multiply(q1, q2):
        """Multipliziert zwei Quaternionen (w, x, y, z)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def _mat2quat(mat):
        """
        Konvertiert eine 3x3-Rotationsmatrix in eine Quaternion (w, x, y, z).
        """
        # Siehe https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        trace = np.trace(mat)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (mat[2,1] - mat[1,2]) * s
            y = (mat[0,2] - mat[2,0]) * s
            z = (mat[1,0] - mat[0,1]) * s
        else:
            # Bestimme die größte Diagonale
            if mat[0,0] > mat[1,1] and mat[0,0] > mat[2,2]:
                s = 2.0 * np.sqrt(1.0 + mat[0,0] - mat[1,1] - mat[2,2])
                w = (mat[2,1] - mat[1,2]) / s
                x = 0.25 * s
                y = (mat[0,1] + mat[1,0]) / s
                z = (mat[0,2] + mat[2,0]) / s
            elif mat[1,1] > mat[2,2]:
                s = 2.0 * np.sqrt(1.0 + mat[1,1] - mat[0,0] - mat[2,2])
                w = (mat[0,2] - mat[2,0]) / s
                x = (mat[0,1] + mat[1,0]) / s
                y = 0.25 * s
                z = (mat[1,2] + mat[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + mat[2,2] - mat[0,0] - mat[1,1])
                w = (mat[1,0] - mat[0,1]) / s
                x = (mat[0,2] + mat[2,0]) / s
                y = (mat[1,2] + mat[2,1]) / s
                z = 0.25 * s
        # Normalisieren (sollte bereits normalisiert sein, aber sicherheitshalber)
        norm = np.linalg.norm([w, x, y, z])
        return np.array([w, x, y, z]) / norm