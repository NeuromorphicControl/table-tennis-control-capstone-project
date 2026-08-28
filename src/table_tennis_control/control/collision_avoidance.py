"""Implements a damped Khatib-style repulsive potential field that keeps the arm's links off the table and off each other."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..config import CollisionConfig

__all__ = ["CollisionAvoider", "CollisionState"]


@dataclass
class CollisionState:
    """Diagnostics of the last evaluation."""

    minimum_distance: float
    active_pairs: int
    torque: np.ndarray
    closest_point: np.ndarray | None = None


class CollisionAvoider:
    """Repulsive potential field between the arm links and the environment."""

    def __init__(self, model, data, dof_ids: np.ndarray, config: CollisionConfig | None = None):
        self.model = model
        self.data = data
        self.dof_ids = np.asarray(dof_ids, dtype=int)
        self.config = config or CollisionConfig()

        self.link_geoms = np.array(
            [model.geom(name).id for name in self.config.link_geoms if _has_geom(model, name)],
            dtype=int,
        )
        self.obstacle_geoms = np.array(
            [model.geom(name).id for name in self.config.obstacle_geoms if _has_geom(model, name)],
            dtype=int,
        )
        self.self_pairs = _self_pairs(model, self.link_geoms, self.config.self_collision_excludes)

        self._fromto = np.zeros(6)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._counter = 0
        self._cached = CollisionState(np.inf, 0, np.zeros(self.dof_ids.size))
        self._smoothed_torque = np.zeros(self.dof_ids.size)

    def _point_jacobian(self, point: np.ndarray, body_id: int) -> np.ndarray:
        """Jacobian of the given point on the given body, expressed in world frame."""
        mujoco.mj_jacPointAxis(self.model, self.data, self._jacp, self._jacr, point, np.array([0.0, 0.0, 1.0]), body_id) # type: ignore
        return self._jacp[:, self.dof_ids]

    def _repulsion(self, geom_a: int, geom_b: int, influence: float):
        """Distance and outward (b -> a) repulsion direction, or ``None`` if out of range."""
        distance = mujoco.mj_geomDistance(self.model, self.data, geom_a, geom_b, influence, self._fromto) # type: ignore
        if distance >= influence:
            return None

        on_a = self._fromto[0:3].copy()
        on_b = self._fromto[3:6].copy()
        direction = on_a - on_b
        norm = float(np.linalg.norm(direction))

        # If the two witness points are exactly coincident, we choose to push along the world z-axis instead of returning a NaN direction
        if norm < 1e-9:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction = direction / norm

        # Swap the direction if the witness point on the link is inside the obstacle (negative distance)
        if distance < 0.0:
            direction = -direction
        return distance, on_a, on_b, direction

    def _magnitude(self, distance: float, approach_rate: float) -> float:
        """Repulsion magnitude for the given distance and approach rate."""
        saturation = self.config.saturation_distance
        clamped = max(distance, saturation)

        magnitude = self.config.gain * (1.0 / clamped - 1.0 / self.config.influence_distance) / clamped**2

        if approach_rate < 0.0:
            magnitude -= self.config.damping * approach_rate
        return float(np.clip(magnitude, 0.0, self.config.max_force))

    def compute(self) -> CollisionState:
        """Repulsive joint torque for the current configuration.

        The (comparatively expensive) distance queries are decimated and the
        *target* torque of the last evaluation held in between; what this
        returns is a first-order blend towards that target, recomputed every
        call, so a newly (de)activated pair ramps in over roughly one
        decimation period instead of stepping.
        """
        if not self.config.enabled or self.link_geoms.size == 0:
            self._cached = CollisionState(np.inf, 0, np.zeros(self.dof_ids.size))
            return self._blend()

        self._counter += 1
        if self._counter % max(1, self.config.decimation) != 0:
            return self._blend()

        torque = np.zeros(self.dof_ids.size)
        qvel = self.data.qvel[self.dof_ids]
        influence = self.config.influence_distance

        minimum = np.inf
        active = 0
        closest_point = None

        # Check all link-obstacle pairs and all self-collision pairs, summing the resulting torques
        for link in self.link_geoms:
            body_id = int(self.model.geom_bodyid[link])
            for obstacle in self.obstacle_geoms:
                # Compute the repulsion between the link and the obstacle, if any
                hit = self._repulsion(int(link), int(obstacle), influence)
                if hit is None:
                    continue
                distance, on_link, _on_obstacle, direction = hit

                # If this is the closest pair so far, remember it for diagnostics.
                if distance < minimum:
                    minimum = distance
                    closest_point = on_link
                active += 1

                # Compute the approach rate of the link witness point along the repulsion direction
                jacobian = self._point_jacobian(on_link, body_id)
                approach_rate = float(direction @ (jacobian @ qvel))

                # Compute the repulsion magnitude and convert it to joint torque via the Jacobian transpose
                magnitude = self._magnitude(distance, approach_rate)
                torque += jacobian.T @ (magnitude * direction)

        # Check all link-link pairs for self-collision, summing the resulting torques
        for link_a, link_b in self.self_pairs:
            # Compute the repulsion between the two links, if any
            hit = self._repulsion(int(link_a), int(link_b), influence)
            if hit is None:
                continue
            distance, on_a, on_b, direction = hit

            # If this is the closest pair so far, remember it for diagnostics.
            if distance < minimum:
                minimum = distance
                closest_point = on_a
            active += 1

            # Compute the approach rate of the two witness points along the repulsion direction
            body_a = int(self.model.geom_bodyid[link_a])
            body_b = int(self.model.geom_bodyid[link_b])
            jacobian_a = self._point_jacobian(on_a, body_a)
            jacobian_b = self._point_jacobian(on_b, body_b)
            approach_rate = float(direction @ (jacobian_a @ qvel - jacobian_b @ qvel))

            # Compute the repulsion magnitude and convert it to joint torque via the Jacobian transpose
            magnitude = self._magnitude(distance, approach_rate)
            torque += jacobian_a.T @ (magnitude * direction)
            torque -= jacobian_b.T @ (magnitude * direction)

        # Clamp the total torque to avoid exceeding the actuator limits
        torque_norm = float(np.linalg.norm(torque))
        if torque_norm > self.config.max_total_torque:
            torque *= self.config.max_total_torque / torque_norm

        # Cache the result for the next decimation period, and return a first-order blend towards it
        self._cached = CollisionState(float(minimum), active, torque, closest_point)
        return self._blend()

    def _blend(self) -> CollisionState:
        """Blend the last decimated torque towards the current target."""
        blend = 1.0 / max(1, self.config.decimation)
        self._smoothed_torque = self._smoothed_torque + (self._cached.torque - self._smoothed_torque) * blend
        return CollisionState(
            self._cached.minimum_distance, 
            self._cached.active_pairs, 
            self._smoothed_torque, 
            self._cached.closest_point
        )


def _has_geom(model, name: str) -> bool:
    """Check if the model has a geom with the given name."""
    try:
        model.geom(name)
    except KeyError:
        return False
    return True


def _self_pairs(model, link_geoms: np.ndarray, excludes: tuple[tuple[str, str], ...]) -> list[tuple[int, int]]:
    """All ``(geom_a, geom_b)`` link pairs worth a repulsion check.

    Skips a geom against itself, links whose bodies are directly joined (their
    proxies are sized to meet at the shared joint, so that pair would be in
    constant spurious contact -- mirrors the ``<exclude>`` list in
    ``world.xml``), and any pair named in ``excludes``.
    """
    # Build a set of excluded geom pairs from the names in ``excludes``.
    excluded_ids = set()
    for name_a, name_b in excludes:
        if _has_geom(model, name_a) and _has_geom(model, name_b):
            excluded_ids.add(frozenset((model.geom(name_a).id, model.geom(name_b).id)))

    # Build the list of all link-link pairs that are not excluded.
    pairs = []
    for i, geom_a in enumerate(link_geoms):
        body_a = int(model.geom_bodyid[geom_a])
        for geom_b in link_geoms[i + 1 :]:
            body_b = int(model.geom_bodyid[geom_b])
            if body_a == body_b:
                continue
            if model.body_parentid[body_a] == body_b or model.body_parentid[body_b] == body_a:
                continue
            if frozenset((int(geom_a), int(geom_b))) in excluded_ids:
                continue
            pairs.append((int(geom_a), int(geom_b)))
    return pairs
