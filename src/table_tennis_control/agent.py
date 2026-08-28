"""Defines the rally agent that wires the sensor, observer, predictor, planner and controller into the full table-tennis control loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import SimulationConfig
from .control import ArmDiagnostics, RobotArm
from .estimation import BallObserver, BallPredictor, Trajectory
from .planning import GoToTrajectory, StrikePlan, StrikePlanner, SwingTrajectory, TaskState
from .world import BallLauncher, BallSensor, Scene

__all__ = ["Phase", "RallyAgent", "AgentDiagnostics", "RallyStatistics"]


class Phase(Enum):
    """State of the stroke state machine."""

    IDLE = "idle"       # no playable ball, hold the ready pose
    TRACK = "track"     # a stroke is planned and still being refined
    SWING = "swing"     # plan committed, executing the stroke
    RECOVER = "recover" # returning to the ready pose


@dataclass
class RallyStatistics:
    """Bookkeeping over a session."""

    serves: int = 0
    strikes: int = 0
    landings: int = 0
    on_target_half: int = 0
    landing_errors: list[float] = field(default_factory=list)

    @property
    def mean_error(self) -> float:
        return float(np.mean(self.landing_errors)) if self.landing_errors else float("nan")

    @property
    def last_error(self) -> float:
        return self.landing_errors[-1] if self.landing_errors else float("nan")

    @property
    def success_rate(self) -> float:
        return self.on_target_half / self.strikes if self.strikes else float("nan")


@dataclass
class AgentDiagnostics:
    """Everything the overlay and the logger need for one step."""

    time: float
    phase: Phase
    arm: ArmDiagnostics
    reference: TaskState
    measured: TaskState
    prediction: Trajectory | None
    plan: StrikePlan | None
    ball_position: np.ndarray
    ball_velocity: np.ndarray
    disturbance: np.ndarray
    sensor_position: np.ndarray
    time_to_impact: float


class RallyAgent:
    """Plays one ball after the other against the launcher."""

    # Minimum spacing between recorded points of the actual-path trace [s].
    _PATH_SAMPLE_DT = 0.01

    def __init__(self, scene: Scene, arm: RobotArm, config: SimulationConfig | None = None):
        self.scene = scene
        self.arm = arm
        self.config = config or scene.config

        dt = scene.timestep
        gravity = scene.gravity

        self.sensor = BallSensor(self.config.sensor)
        self.observer = BallObserver(dt, self.config.observer, gravity)
        self.predictor = BallPredictor(
            self.config.table,
            self.config.ball,
            gravity,
            sample_time=0.008,
            horizon=self.config.planner.max_time_to_impact + 0.6,
        )
        self.planner = StrikePlanner(
            self.config.table, 
            self.config.ball, 
            self.config.arm, 
            self.config.planner
        )
        self.launcher = BallLauncher(
            scene.ball,
            self.config.table,
            self.config.ball,
            self.config.launcher,
            scene.generator,
            strike_zone=self.config.arm.workspace,
        )

        self.phase = Phase.IDLE
        self.plan: StrikePlan | None = None
        self.prediction: Trajectory | None = None
        self.statistics = RallyStatistics()

        self._replan_period = 1.0 / max(self.config.planner.replan_rate, 1e-3)
        self._next_replan = -np.inf
        self._trajectory = self._ready_trajectory(arm.measure(), scene.time, duration=1.0)
        self._next_serve_time = scene.time + 0.5
        self._awaiting_landing = False
        self._ball_geom = scene.ball.geom_id
        self._ground_geom = int(scene.model.geom("ground").id)
        self._locked_impact_time: float | None = None

        # The actual path of the ball from the last launch to its first floor bounce
        self._actual_path: list[np.ndarray] = []
        self._actual_path_active = False
        self._actual_path_next_sample = -np.inf

        #: Cached conversion of `_actual_path`, rebuilt only when it has grown
        self._actual_path_array = np.zeros((0, 3))
        self._actual_path_array_len = 0

        # The paddle's actual (measured) position over the same span as the ball's path
        # above, and frozen at the same moment (the ball's first floor bounce).
        self._paddle_path: list[np.ndarray] = []
        self._paddle_path_phases: list[Phase] = []
        self._paddle_path_active = False
        self._paddle_path_next_sample = -np.inf

        #: Cached conversion of `_paddle_path`, rebuilt only when it has grown
        self._paddle_path_array = np.zeros((0, 3))
        self._paddle_path_array_len = 0

    # --------------------------------------------------------------- helpers
    def _ready_trajectory(self, start: TaskState, now: float, duration: float = 0.6) -> GoToTrajectory:
        """Return a trajectory that moves the paddle to the ready pose."""
        return GoToTrajectory(
            start,
            np.asarray(self.config.arm.ready_position, dtype=float),
            np.asarray(self.config.arm.ready_normal, dtype=float),
            duration=duration,
            start_time=now,
        )

    def _estimated_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Current ball state, with the sensor's delay compensated for."""
        position = self.observer.position
        velocity = self.observer.velocity

        # Compensate for the sensor's delay by predicting the ball state forward in time.
        lag = self.sensor.lag_ticks * self.scene.timestep
        if lag > 0.0:
            position, velocity = self.predictor.state_after(position, velocity, self.observer.acceleration, lag)

        return position, velocity

    # ------------------------------------------------------------- serve / landing
    def serve(self) -> None:
        """Launch a new ball from the opponent's side and aim a target across from it."""
        serve = self.launcher.serve()
        if self.config.target.resample_per_serve:
            self.scene.target.set_position(self.scene.target_sampler.sample(launch_x=float(serve.position[0])))

        # Reset the sensor and observer to the new ball state, so neither tries to track the old one
        self.sensor.reset()
        self.observer.reset(serve.position, serve.velocity)

        # Update statistics and reset the state machine for the new rally
        self.statistics.serves += 1
        self._awaiting_landing = False
        self.plan = None
        self.phase = Phase.IDLE
        self._next_serve_time = self.scene.time + self.config.launcher.serve_interval

        # Old trace disappears the instant the new ball is launched; the new one starts right at the launch point.
        self._actual_path = [self.scene.ball.position.copy()]
        self._actual_path_active = True
        self._actual_path_next_sample = self.scene.time + self._PATH_SAMPLE_DT
        self._actual_path_array_len = 0  # invalidate the cache: a new list means a stale length could alias

        # Same reset for the paddle's trace, seeded with its current (pre-serve) position.
        self._paddle_path = [self.arm.measure().position.copy()]
        self._paddle_path_phases = [self.phase]
        self._paddle_path_active = True
        self._paddle_path_next_sample = self.scene.time + self._PATH_SAMPLE_DT
        self._paddle_path_array_len = 0

    def _find_ground_contact(self) -> np.ndarray | None:
        """Position of the ball's first contact with the floor this step, if any."""
        data = self.scene.data
        for index in range(data.ncon):
            contact = data.contact[index]
            geoms = (int(contact.geom1), int(contact.geom2))

            if self._ball_geom not in geoms:
                continue

            # If the ball is in contact with the ground, return the contact position
            other = geoms[0] if geoms[1] == self._ball_geom else geoms[1]
            if other == self._ground_geom:
                return np.asarray(contact.pos, dtype=float)

        # If the ball is not in contact with the ground, return None
        return None

    def _check_landing(self, ground_contact: np.ndarray | None) -> None:
        """Watch for the landing of a returned ball.

        The target always sits on the floor, so the *landing* is always the
        first ``ground`` contact.  A "standard" return is *supposed* to touch
        the opponent's table on the way there -- that intermediate bounce
        does not count (only a ``ground`` contact does, see
        :meth:`_find_ground_contact`).
        """
        if not self._awaiting_landing or ground_contact is None:
            return

        target = self.scene.target.position
        self.statistics.landings += 1
        self.statistics.landing_errors.append(float(np.linalg.norm(ground_contact[:2] - target[:2])))

        # If the ball landed on the opponent's half of the table, count it as a successful return.
        if ground_contact[1] * self.config.table.opponent_side > 0.0:
            self.statistics.on_target_half += 1

        self._awaiting_landing = False

    # ------------------------------------------------------------------- plan
    def _update_prediction(self) -> None:
        position, velocity = self._estimated_state()
        self.prediction = self.predictor.predict(position, velocity, self.observer.acceleration)

    def _swing_from(self, plan: StrikePlan, reference: TaskState, now: float) -> SwingTrajectory:
        return SwingTrajectory(
            start=reference,
            start_time=now,
            impact_time=plan.impact_time,
            impact_position=plan.paddle_position,
            impact_velocity=plan.paddle_velocity,
            impact_normal=plan.normal,
            follow_through_time=self.config.planner.follow_through_time,
            follow_through_distance=self.config.planner.follow_through_distance,
        )

    def _anticipate(self, reference: TaskState, now: float) -> None:
        """Anticipate the ball's trajectory and prepare for interception."""
        assert self.prediction is not None
        anticipation = self.planner.anticipation_point(self.prediction)

        # If the ball is not yet in a playable state, go to the ready pose and wait for it to become playable
        if anticipation is None:
            if self.phase in (Phase.TRACK, Phase.SWING):
                self.phase = Phase.RECOVER
                self._trajectory = self._ready_trajectory(reference, now)
            return

        # The ball is in a playable state, so plan a stroke to intercept it
        position, velocity, horizon = anticipation
        target = self.scene.target.position

        # If the planner cannot find a stroke that will hit the target, it will use the ready normal instead of the anticipated normal
        normal = self.planner.anticipated_normal(reference, position, velocity, horizon, target, self.observer.acceleration)
        if normal is None:
            normal = np.asarray(self.config.arm.ready_normal, dtype=float)

        # Offset the paddle position along normal to avoid the ball being inside the paddle at impact
        offset = normal * self.config.arm.contact_offset
        self.phase = Phase.TRACK
        self._trajectory = GoToTrajectory(
            reference,
            position - offset,
            normal,
            duration=float(np.clip(0.7 * horizon, 0.12, 1.2)),
            start_time=now,
        )

    def _replan(self, now: float) -> None:
        """Search for a stroke, or refine the one we already committed to."""
        if self.prediction is None:
            self._update_prediction()
        assert self.prediction is not None

        reference = self._trajectory.evaluate(now)
        target = self.scene.target.position
        acceleration = self.observer.acceleration

        # If we already committed to a stroke, try to refine it
        if self._locked_impact_time is not None:
            plan = self.planner.refine(now, self.prediction, reference, target, self._locked_impact_time, acceleration)

            # If the refinement succeeded, update the plan and trajectory
            if plan is not None:
                self.plan = plan
                self._trajectory = self._swing_from(plan, reference, now)
                return
            
            # If the refinement failed, we can no longer trust the locked impact time, so we clear it and try to find a new plan
            self._locked_impact_time = None

        plan = self.planner.plan(now, self.prediction, reference, target, acceleration=acceleration, previous=self.plan)

        if plan is None:
            # If we already have a plan, and the impact time is still far enough in the future, we can keep it and not replan yet
            if self.plan is not None and self.plan.impact_time > now + self.config.planner.min_time_to_impact:
                return

            # If we have no plan, or the impact time is too soon, we clear the plan and anticipate the ball's trajectory to prepare for interception
            self.plan = None
            self._anticipate(reference, now)
            return

        self.plan = plan

        # If the new plan is committable, we lock its impact time to prevent further refinements from changing it
        if plan.is_committable:
            self._locked_impact_time = plan.impact_time
        self.phase = Phase.TRACK
        self._trajectory = self._swing_from(plan, reference, now)

    # ------------------------------------------------------------------- step
    def step(self) -> AgentDiagnostics:
        """Advance observer, planner and controller by one simulation step."""
        now = self.scene.time
        config = self.config

        # 1 - Sensor + observer: update the ball state estimate from the latest (delayed) measurement
        measurement = self.sensor.measure(self.scene.ball)
        self.observer.update(measurement, self.sensor.settled)

        time_to_impact = float("inf")
        if self.plan is not None:
            time_to_impact = self.plan.impact_time - now

        # If we are committed to a stroke, we cannot change the impact time anymore, so we lock it and switch to the SWING phase
        committed = self.phase is Phase.SWING or (self.plan is not None and time_to_impact <= config.planner.commit_horizon)
        if committed and self.phase is Phase.TRACK:
            self.phase = Phase.SWING
            self.statistics.strikes += 1
            self._awaiting_landing = True

        # 2 - Planner: update the ball prediction and search for a new stroke if needed
        if now >= self._next_replan and not self.observer.in_contact:
            self._next_replan = now + self._replan_period
            self._update_prediction()
            if not committed:
                self._replan(now)
                if self.plan is not None:
                    time_to_impact = self.plan.impact_time - now

        # Update phase transitions based on the trajectory's end time
        if self.phase is Phase.SWING and now > self._trajectory.end_time:
            self.phase = Phase.RECOVER
            self.plan = None
            self._locked_impact_time = None
            self._trajectory = self._ready_trajectory(self._trajectory.evaluate(now), now, duration=0.8)
        elif self.phase is Phase.RECOVER and now > self._trajectory.end_time:
            self.phase = Phase.IDLE

        # 3 - Controller: compute the reference state and apply feedback control to the robot arm
        reference = self._trajectory.evaluate(now)
        measured = self.arm.measure()
        arm_diagnostics = self.arm.update(reference)

        # Step the simulation forward and limit the arm's velocity to avoid numerical instability
        self.scene.step()
        self.arm.limit_velocity()

        # Record the actual path of the ball from the last launch to its first floor bounce
        ground_contact = self._find_ground_contact()
        if self._actual_path_active:
            if ground_contact is not None or now >= self._actual_path_next_sample:
                self._actual_path.append(self.scene.ball.position.copy())
                self._actual_path_next_sample = now + self._PATH_SAMPLE_DT
            if ground_contact is not None:
                self._actual_path_active = False  # trace freezes here until the next serve()
        self._check_landing(ground_contact)

        # Record the paddle's actual position over the same span, frozen at the same moment
        if self._paddle_path_active:
            if ground_contact is not None or now >= self._paddle_path_next_sample:
                self._paddle_path.append(measured.position.copy())
                self._paddle_path_phases.append(self.phase)
                self._paddle_path_next_sample = now + self._PATH_SAMPLE_DT
            if ground_contact is not None:
                self._paddle_path_active = False  # trace freezes here until the next serve()

        # Return diagnostics for logging and overlay display
        return AgentDiagnostics(
            time=now,
            phase=self.phase,
            arm=arm_diagnostics,
            reference=reference,
            measured=measured,
            prediction=self.prediction,
            plan=self.plan,
            ball_position=self.observer.position.copy(),
            ball_velocity=self.observer.velocity.copy(),
            disturbance=self.observer.disturbance.copy(),
            sensor_position=measurement,
            time_to_impact=time_to_impact,
        )

    @property
    def actual_path(self) -> np.ndarray:
        """True ball positions from the last launch to its first floor bounce.

        Converted from the recording buffer on demand and cached until it
        grows again, since a rebuild replaces (never mutates) the cached array.
        """
        if len(self._actual_path) < 2:
            return np.zeros((0, 3))

        if self._actual_path_array_len != len(self._actual_path):
            self._actual_path_array = np.asarray(self._actual_path, dtype=float)
            self._actual_path_array_len = len(self._actual_path)
        return self._actual_path_array

    @property
    def paddle_path(self) -> np.ndarray:
        """True paddle positions from the last launch to the ball's first floor bounce.

        Converted from the recording buffer on demand and cached until it
        grows again, since a rebuild replaces (never mutates) the cached array.
        """
        if len(self._paddle_path) < 2:
            return np.zeros((0, 3))

        if self._paddle_path_array_len != len(self._paddle_path):
            self._paddle_path_array = np.asarray(self._paddle_path, dtype=float)
            self._paddle_path_array_len = len(self._paddle_path)
        return self._paddle_path_array

    @property
    def paddle_path_phases(self) -> list[Phase]:
        """:class:`Phase` recorded alongside every point of :attr:`paddle_path`."""
        if len(self._paddle_path) < 2:
            return []
        return list(self._paddle_path_phases)

    def maybe_serve(self) -> bool:
        """Serve again once the rally is over or the interval has elapsed.
        
        Return True if a new ball was served, False otherwise.
        """
        if self.scene.time < self._next_serve_time:
            return False

        ball_z = float(self.scene.ball.position[2])
        settled = ball_z < self.config.table.height and abs(float(self.scene.ball.velocity[2])) < 0.6

        if settled or self.scene.time > self._next_serve_time + 1.5:
            self.serve()
            return True
        return False
