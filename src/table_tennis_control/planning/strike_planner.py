"""Picks *where*, *when* and *how* to hit the incoming ball.

This is the high-level planner of the hierarchy from Lecture 6: it searches a
discrete set of candidate actions (impact point x return type), scores them
with a cost function and hands the winner down to the low-level
operational-space controller as a smooth reference trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import GRAVITY, ArmSpec, BallSpec, PlannerConfig, TableSpec
from ..estimation.predictor import Trajectory
from ..kinematics import normalise, point_in_box
from ..physics import bounce_velocity, paddle_impact_inverse
from .return_solver import ReturnSolutions, solve_bounce_return
from .trajectory import TaskState

__all__ = ["StrikePlan", "StrikePlanner"]


@dataclass
class StrikePlan:
    """A fully specified plan for one stroke."""

    impact_time: float  # absolute simulation time of the impact
    position: np.ndarray  # predicted ball centre at the impact
    normal: np.ndarray  # paddle face normal at the impact
    paddle_velocity: np.ndarray  # paddle velocity at the impact
    incoming_velocity: np.ndarray  # ball velocity just before the impact
    outgoing_velocity: np.ndarray  # ball velocity just after the impact
    bounce_point: np.ndarray | None  # first table contact of the return
    landing_point: np.ndarray  # intended landing point
    move_time: float  # estimated time the arm needs for the motion
    cost: float
    bounces_before: int = 0 # Number of table bounces the ball still makes before this impact.
    paddle_position: np.ndarray = field(default_factory=lambda: np.zeros(3)) # Where the paddle *site* has to be at the impact
    time_to_bounce: float = float("nan") # Time from the impact to the first table bounce of the return
    time_after_bounce: float = float("nan") # Time from the first table bounce of the return to the landing point
    acceleration: np.ndarray = field(default_factory=lambda: GRAVITY.copy()) # Acceleration used for the ballistic model of the return

    @property
    def paddle_speed(self) -> float:
        return float(np.linalg.norm(self.paddle_velocity))

    @property
    def is_committable(self) -> bool:
        return self.bounces_before == 0

    def rebound_velocity(self, ball: BallSpec, acceleration: np.ndarray | None = None) -> np.ndarray:
        """Ball velocity just after the intermediate bounce of the return."""
        acceleration = self.acceleration if acceleration is None else acceleration
        acceleration = np.asarray(acceleration)

        # The ball's velocity just before the bounce is the outgoing velocity plus the effect of gravity during the flight to the bounce.
        impact = self.outgoing_velocity + acceleration * self.time_to_bounce
        rebound = bounce_velocity(impact, ball.table_restitution, ball.table_tangential)

        # The ball's velocity after the bounce is the rebound plus the effect of gravity during the contact duration.
        return rebound + acceleration * ball.table_impact_duration


class StrikePlanner:
    """Searches the predicted ball trajectory for the best stroke."""

    def __init__(self, table: TableSpec, ball: BallSpec, arm: ArmSpec, config: PlannerConfig | None = None):
        """Initialise a planner for one robot arm, one table and one ball.
        
        Args:
            table: The table specification.
            ball: The ball specification.
            arm: The arm specification.
            config: The planner configuration.
        """
        self.table = table
        self.ball = ball
        self.arm = arm
        self.config = config or PlannerConfig()

    # ------------------------------------------------------------- candidates
    def _candidate_mask(self, trajectory: Trajectory) -> np.ndarray:
        """Points of the prediction that the arm may legally intercept."""
        positions, velocities, times = trajectory.positions, trajectory.velocities, trajectory.times

        inside = point_in_box(positions, self.arm.workspace) # type: ignore
        in_time = (times >= self.config.min_time_to_impact) & (times <= self.config.max_time_to_impact)
        approaching = velocities[:, 1] * self.table.robot_side > 0.0
        return inside & in_time & approaching

    def anticipation_point(self, trajectory: Trajectory) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Where to wait for the ball while no stroke is committable yet.

        Returns the first predicted position inside the working volume,
        together with the ball's predicted velocity there and the time until
        the ball gets there.
        """
        mask = self._candidate_mask(trajectory)
        indices = np.flatnonzero(mask)

        if indices.size == 0:
            return None

        # Pick a point in the first third of the candidates, so that the arm has time to pre-position and turn towards the impact normal before the ball arrives
        chosen = indices[min(indices.size - 1, indices.size // 3)]
        return (
            trajectory.positions[chosen].copy(),
            trajectory.velocities[chosen].copy(),
            float(trajectory.times[chosen]),
        )

    def anticipated_normal(self, paddle: TaskState, position: np.ndarray, velocity: np.ndarray, horizon: float, target: np.ndarray, acceleration: np.ndarray | None = None) -> np.ndarray | None:
        """Best-guess paddle normal for a point the arm is only pre-positioning towards.

        Args:
            paddle: Current *reference* state of the paddle (not the measured one -- planning from the reference keeps the motion continuous).
            position: Predicted ball position at the impact.
            velocity: Predicted ball velocity at the impact.
            horizon: Time until the impact.
            target: Desired landing point.
            acceleration: Acceleration used for the ballistic model.
        
        Returns:
            The paddle normal that would produce the requested outgoing velocity, or ``None`` if the arm cannot reach it in time.
        """
        acceleration = GRAVITY if acceleration is None else np.asarray(acceleration, dtype=float)
        positions = np.asarray(position, dtype=float)[None, :]
        incoming = np.asarray(velocity, dtype=float)[None, :]
        times = np.array([horizon])

        # Solve the return problem for this single candidate point, ignoring the arm's reachability.
        solutions = self._solve_returns(positions, target, acceleration)
        if not np.any(solutions.feasible):
            return None

        # Compute the paddle normal and speed that would produce the requested outgoing velocity, and check whether the arm can reach it in time.
        normals, speeds, paddle_velocity, feasible = self._impact_geometry(incoming, solutions)
        if not np.any(feasible):
            return None

        # Compute the time the arm would need to reach each candidate normal, and discard those that are unreachable.
        required = self.movement_time(paddle, positions[:, None, :], paddle_velocity, normals)
        cost = self._cost(positions, times, required, speeds, solutions.time_to_bounce)
        cost = np.where(feasible, cost, np.inf)

        # Pick the cheapest candidate and return its normal.
        k = int(np.argmin(cost[0]))
        if not np.isfinite(cost[0, k]):
            return None

        return normals[0, k].copy()

    # ----------------------------------------------------------------- returns
    def _solve_returns(self, positions: np.ndarray, target: np.ndarray, acceleration: np.ndarray) -> ReturnSolutions:
        return solve_bounce_return(
            positions,
            target,
            self.table,
            self.ball,
            np.asarray(self.config.post_bounce_times),
            bounce_side=self.table.opponent_side,
            acceleration=acceleration,
            bounce_margin=self.config.bounce_margin,
            net_clearance=self.config.net_clearance,
            max_speed=self.config.max_outgoing_speed,
            min_time_to_bounce=self.config.min_time_to_bounce,
        )

    # ------------------------------------------------------------- reachability
    def movement_time(self, start: TaskState, positions: np.ndarray, velocities: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Lower bound on the time the arm needs for the requested motion.

        Uses the closed-form peak velocity/acceleration of a minimum-jerk
        profile (``v_peak = 1.875 D/T``, ``a_peak = 5.7735 D/T^2``) instead of
        simulating the motion, which keeps the planner cheap enough to run
        inside the control loop.  Broadcasts over any leading dimensions.

        Args:
            start: Current *reference* state of the paddle (not the measured one -- planning from the reference keeps the motion continuous).
            positions: Requested paddle positions at the impact.
            velocities: Requested paddle velocities at the impact.
            normals: Requested paddle normals at the impact.
        
        Returns:
            The minimum time the arm needs to reach each requested state, in the same shape as the leading dimensions of ``positions``.
        """
        distance = np.linalg.norm(np.asarray(positions) - start.position, axis=-1)

        # Compute the time required to reach the requested position at the requested speed, and the time required to accelerate to that speed from rest
        by_speed = 1.875 * distance / self.arm.max_task_speed
        by_acceleration = np.sqrt(5.7735 * distance / self.arm.max_task_acceleration)
        by_impact_speed = 1.8 * np.linalg.norm(np.asarray(velocities) - start.velocity, axis=-1) / self.arm.max_task_acceleration

        # Compute the time required to turn the paddle from its current orientation to the requested normal
        cosine = np.clip(np.sum(np.asarray(normals) * normalise(start.normal), axis=-1), -1.0, 1.0)
        angle = np.arccos(cosine)
        by_turn = np.maximum(
            1.875 * angle / self.arm.max_task_angular_speed,
            np.sqrt(5.7735 * angle / self.arm.max_task_angular_acceleration),
        )
        return np.maximum(np.maximum(by_speed, by_acceleration), np.maximum(by_impact_speed, by_turn))

    # -------------------------------------------------------------------- plan
    def plan(self, now: float, trajectory: Trajectory, paddle: TaskState, target: np.ndarray, acceleration: np.ndarray | None = None, previous: StrikePlan | None = None) -> StrikePlan | None:
        """Return the cheapest feasible stroke, or ``None`` if there is none.

        Args:
            now: Current absolute simulation time.
            trajectory: Predicted ball trajectory.
            paddle: Current *reference* state of the paddle (not the measured one -- planning from the reference keeps the motion continuous).
            target: Desired landing point.
            acceleration: Acceleration used for the ballistic model.
            previous: Previous plan, if any, to encourage continuity.
        
        Returns:
            The cheapest feasible stroke, or ``None`` if there is none.
        """
        if len(trajectory) == 0:
            return None
        acceleration = GRAVITY if acceleration is None else np.asarray(acceleration, dtype=float)

        # Pick a subset of the predicted trajectory that is inside the working volume, within the time limits and approaching the paddle
        mask = self._candidate_mask(trajectory)
        indices = np.flatnonzero(mask)[:: max(1, self.config.candidate_stride)]
        if indices.size == 0:
            return None

        times = trajectory.times[indices]
        positions = trajectory.positions[indices]
        incoming = trajectory.velocities[indices]

        # Solve the return problem for every candidate point, ignoring the arm's reachability
        solutions = self._solve_returns(positions, target, acceleration)
        if not np.any(solutions.feasible):
            return None

        # Compute the paddle normal and speed that would produce the requested outgoing velocity
        outgoing = solutions.outgoing_velocity
        normals, speeds, paddle_velocity, feasible = self._impact_geometry(incoming, solutions)

        # Compute the time the arm would need to reach each candidate normal, and discard those that are unreachable
        required = self.movement_time(paddle, positions[:, None, :], paddle_velocity, normals)
        feasible &= required <= times[:, None] * self.config.time_margin
        if not np.any(feasible):
            return None

        # Score every candidate with a cost function that trades off speed, distance, paddle speed and continuity with the previous plan
        cost = self._cost(positions, times, required, speeds, solutions.time_to_bounce, previous)
        cost = np.where(feasible, cost, np.inf)
        i, k = np.unravel_index(int(np.argmin(cost)), cost.shape)

        # Build a plan object for the cheapest candidate and return it
        return self._build_plan(now, times, positions, incoming, outgoing, normals, paddle_velocity, speeds, solutions, required, cost, i, k, target, trajectory, acceleration)

    # ------------------------------------------------------------------ refine
    def refine(self, now: float, trajectory: Trajectory, paddle: TaskState, target: np.ndarray, impact_time: float, acceleration: np.ndarray | None = None) -> StrikePlan | None:
        """Update a committed plan without moving its impact *time*.

        Freezing the instant of the impact and only correcting *where* the ball
        will be at that instant is what keeps the swing smooth: the reference
        trajectory keeps its duration and merely bends a little as the
        prediction improves.

        Args:
            now: Current absolute simulation time.
            trajectory: Predicted ball trajectory.
            paddle: Current *reference* state of the paddle (not the measured one -- planning from the reference keeps the motion continuous).
            target: Desired landing point.
            impact_time: Absolute simulation time of the committed impact.
            acceleration: Acceleration used for the ballistic model.
        
        Returns:
            The cheapest feasible stroke that hits the ball at the requested time, or ``None`` if there is none.
        """
        if len(trajectory) == 0:
            return None
        acceleration = GRAVITY if acceleration is None else np.asarray(acceleration, dtype=float)

        horizon = impact_time - now
        index = int(np.argmin(np.abs(trajectory.times - horizon)))
        times = trajectory.times[index : index + 1]
        positions = trajectory.positions[index : index + 1]
        incoming = trajectory.velocities[index : index + 1]

        # Solve the return problem for every candidate point, ignoring the arm's reachability
        solutions = self._solve_returns(positions, target, acceleration)
        outgoing = solutions.outgoing_velocity
        normals, speeds, paddle_velocity, feasible = self._impact_geometry(incoming, solutions)
        if not np.any(feasible):
            return None

        # Compute the time the arm would need to reach each candidate normal, and discard those that are unreachable
        required = self.movement_time(paddle, positions[:, None, :], paddle_velocity, normals)
        feasible &= required <= np.maximum(times[:, None], 1e-3) * self.config.refine_time_margin
        if not np.any(feasible):
            return None

        # Score every candidate with a cost function that trades off speed, distance, paddle speed and continuity with the previous plan
        cost = self._cost(positions, times, required, speeds, solutions.time_to_bounce)
        cost = np.where(feasible, cost, np.inf)
        i, k = np.unravel_index(int(np.argmin(cost)), cost.shape)

        # Build a plan object for the cheapest candidate and return it
        return self._build_plan(now, times, positions, incoming, outgoing, normals, paddle_velocity, speeds, solutions, required, cost, i, k, target, trajectory, acceleration)

    # ------------------------------------------------------------------ pieces
    def _impact_geometry(self, incoming: np.ndarray, solutions: ReturnSolutions):
        """Paddle normal, normal speed and feasibility for every candidate.

        Inverts the friction-aware impact model, see
        :func:`table_tennis_control.physics.paddle_impact_inverse`.
        """
        # The ball's velocity just before the impact is the outgoing velocity minus the effect of gravity during the flight to the paddle
        outgoing = solutions.outgoing_velocity
        outgoing = outgoing - GRAVITY * self.ball.paddle_impact_duration

        # Compute the paddle normal and speed that would produce the requested outgoing velocity, and check whether the arm can reach it in time
        normals, speeds = paddle_impact_inverse(incoming[:, None, :], outgoing, self.ball.paddle_restitution, self.ball.paddle_tangential)
        paddle_velocity = speeds[..., None] * normals

        # Check whether the candidate normals are physically feasible and within the arm's speed limits
        feasible = solutions.feasible & np.isfinite(speeds) & np.all(np.isfinite(normals), axis=-1)
        feasible &= np.abs(speeds) <= self.config.max_paddle_speed

        # Check whether the candidate normals are pointing towards the opponent's side of the table and whether the ball is approaching the paddle
        feasible &= normals[..., 1] * self.table.opponent_side > 0.0
        feasible &= np.sum(incoming[:, None, :] * normals, axis=-1) < 0.0
        return normals, speeds, paddle_velocity, feasible

    def _cost(self, positions: np.ndarray, times: np.ndarray, required: np.ndarray, speeds: np.ndarray, bounce_time: np.ndarray, previous: StrikePlan | None = None) -> np.ndarray:
        """Score every candidate with a cost function that trades off speed, distance, paddle speed and continuity with the previous plan."""
        ready = np.asarray(self.arm.ready_position, dtype=float)

        # Compute the cost of every candidate with a weighted sum of the time the arm needs to reach it, the time until the impact, the paddle speed and the distance from the ready position
        cost = (
            self.config.weight_move_time * required
            + self.config.weight_impact_time * times[:, None]
            + self.config.weight_paddle_speed * np.abs(speeds)
            + self.config.weight_ready_distance * np.linalg.norm(positions - ready, axis=-1)[:, None]
        )

        # Encourage candidates that produce a bounce time close to the preferred one, and discourage candidates that produce a bounce time outside the feasible range
        cost = cost + np.where(np.isfinite(bounce_time), self.config.weight_bounce_time * np.abs(bounce_time - self.config.preferred_bounce_time), 0.0)

        # Encourage continuity with the previous plan by penalising candidates that are far from the previous paddle position
        if previous is not None:
            drift = np.linalg.norm(positions - previous.paddle_position, axis=-1)[:, None]
            cost = cost + self.config.weight_continuity * drift
        return cost

    def _build_plan(self, now, times, positions, incoming, outgoing, normals, paddle_velocity, speeds, solutions, required, cost, i, k, target, trajectory, acceleration) -> StrikePlan:
        """Build a plan object for the cheapest candidate."""
        bounce_time = solutions.time_to_bounce
        impact_time = float(times[i])
        normal = normals[i, k].copy()

        # Count how many times the ball has bounced on the table before the impact time, so that the controller can decide whether to commit to the stroke or wait for a better one
        remaining_bounces = sum(1 for event in trajectory.events if event.surface == "table" and event.time < impact_time)

        return StrikePlan(
            impact_time=now + impact_time,
            position=positions[i].copy(),
            paddle_position=positions[i] - normal * self.arm.contact_offset,
            normal=normal,
            paddle_velocity=paddle_velocity[i, k].copy(),
            incoming_velocity=incoming[i].copy(),
            outgoing_velocity=outgoing[i, k].copy(),
            bounce_point=solutions.bounce_point[i, k].copy() if np.isfinite(bounce_time[i, k]) else None,
            landing_point=np.asarray(target, dtype=float).copy(),
            move_time=float(required[i, k]),
            cost=float(cost[i, k]),
            bounces_before=remaining_bounces,
            time_to_bounce=float(bounce_time[i, k]),
            time_after_bounce=float(solutions.time_after_bounce[i, k]),
            acceleration=np.asarray(acceleration, dtype=float).copy(),
        )
