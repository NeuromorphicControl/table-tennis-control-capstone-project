"""Central configuration for the ping-pong robot.

Every tunable number of the project lives here so that experiments only ever
touch a single file.  The dataclasses are grouped by the block of the control
architecture they belong to (plant, observer, planner, controller).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The directory containing the MuJoCo XML world and asset files.
ASSET_DIR = Path(__file__).parent / "assets"
WORLD_XML = ASSET_DIR / "world.xml"

# Gravitational acceleration used by every analytic model in the project.
GRAVITY = np.array([0.0, 0.0, -9.81])


def _critically_damped(kp: tuple[float, float, float]) -> tuple[float, float, float]:
    """``kd = 2 sqrt(kp)``, the unit-mass critically damped PD gain (see :class:`ControlConfig`)."""
    kx, ky, kz = kp
    return 2.0 * kx**0.5, 2.0 * ky**0.5, 2.0 * kz**0.5


def _centered_grid(center: float, lower_spread: float, upper_spread: float, step: float) -> tuple[float, ...]:
    """Evenly spaced grid from ``center - lower_spread`` to ``center + upper_spread``."""
    lower = center - lower_spread
    n = int(round((lower_spread + upper_spread) / step))
    return tuple(round(lower + i * step, 10) for i in range(n + 1))


@dataclass(frozen=True)
class TableSpec:
    """Geometry of the table, the net and the two half-courts.

    The robot stands at negative ``y``; :attr:`robot_side` encodes that sign so
    that "own half" and "opponent half" can be expressed without magic numbers.
    """

    half_width: float = 0.7625     # extent along x [m]
    half_length: float = 1.37      # extent along y (one half court) [m]
    height: float = 0.76           # height of the play surface [m]
    net_height: float = 0.9125     # absolute height of the top of the net [m]
    net_half_width: float = 0.9225 # extent of the net along x [m]
    robot_side: int = -1           # sign of y of the robot's own half

    @property
    def opponent_side(self) -> int:
        return -self.robot_side

    def half_bounds(self, side: int, margin: float = 0.0) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((x_min, x_max), (y_min, y_max))`` of one half court.

        Args:
            side: ``-1`` for the robot's half, ``+1`` for the opponent's half.
            margin: Safety margin that shrinks the rectangle on every edge [m].
        
        Returns:
            ``((x_min, x_max), (y_min, y_max))`` bounds of the half court.
        """
        x_bound = self.half_width - margin
        near = side * margin
        far = side * (self.half_length - margin)
        return (-x_bound, x_bound), (min(near, far), max(near, far))

    def is_on_half(self, points: np.ndarray, side: int, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of the ``(..., 3)`` points that lie on the given half.
        
        Args:
            points: Array of shape ``(..., 3)`` containing 3D points.
            side: ``-1`` for the robot's half, ``+1`` for the opponent's half.
            margin: Safety margin that shrinks the rectangle on every edge [m].
        
        Returns:
            Boolean mask of shape ``(...)`` indicating which points lie on the given half.
        """
        (x_min, x_max), (y_min, y_max) = self.half_bounds(side, margin)
        x, y = points[..., 0], points[..., 1]
        return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)


@dataclass(frozen=True)
class BallSpec:
    """Physical properties of the ball and of the surfaces it bounces off."""

    radius: float = 0.02
    mass: float = 0.00167
    #: Coefficient of restitution when bouncing off the table (measured).
    table_restitution: float = 0.896
    #: Fraction of the tangential velocity that survives a table bounce (measured).
    table_tangential: float = 0.882
    #: Coefficient of restitution of the ball/paddle impact (measured).
    paddle_restitution: float = 0.896
    #: Fraction of the tangential relative velocity that survives a paddle hit (measured).
    paddle_tangential: float = 0.913
    #: Quadratic drag coefficient ``a_drag = -k |v| v`` [1/m], 0 disables drag.
    drag_coefficient: float = 0.0
    #: How long a paddle impact actually takes to resolve in the simulator [s]
    #: (measured). Compensated for before inverting the impact model, see
    #: :meth:`StrikePlanner._impact_geometry`.
    paddle_impact_duration: float = 0.0212
    #: Same compliant-contact effect as :attr:`paddle_impact_duration`, but
    #: for a ball/table impact -- measured to be statistically zero.
    table_impact_duration: float = 0.0

    def bounce_plane(self, table: TableSpec) -> float:
        """Height of the ball centre at the moment it touches the table."""
        return table.height + self.radius


@dataclass(frozen=True)
class ArmSpec:
    """Names and kinematic limits of the actuated arm."""

    joint_names: tuple[str, ...] = (
        "base_x",
        "base_y",
        "rotator1",
        "rotator2",
        "arm1",
        "arm2",
        "paddle_rotator",
        "paddle",
    )
    site_name: str = "paddle_site"
    #: Local axis of the paddle site that coincides with the paddle face normal.
    normal_axis: int = 1
    #: Force/torque limit of each actuator, in the order of :attr:`joint_names`.
    torque_limits: tuple[float, ...] = (800.0, 800.0, 200.0, 200.0, 200.0, 150.0, 60.0, 60.0)
    #: Conservative end-effector speed/acceleration used for reachability tests.
    max_task_speed: float = 4.5  # [m/s]
    max_task_acceleration: float = 45.0  # [m/s^2]
    max_task_angular_speed: float = 14.0  # [rad/s]
    max_task_angular_acceleration: float = 140.0  # [rad/s^2]
    #: Hard cap on joint speed (norm over all DOFs) [rad or m /s], enforced
    #: directly on ``qvel`` after every physics step -- the last line of
    #: defense against joint runaway (measured; see config_provenance.md).
    max_joint_speed: float = 15.0
    #: Box the paddle is allowed to work in, ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``.
    #: The robot plays from behind its own baseline, like a human would.
    #: ``z_max`` is set by an arm self-collision clearance measurement, see
    #: config_provenance.md.
    workspace: tuple[tuple[float, float], ...] = (
        (-1.0, 1.0),
        (-2.45, -1.45),
        (0.60, 1.50),
    )
    #: Distance between the paddle site and the ball centre at contact [m].
    contact_offset: float = 0.030
    #: Pose the arm returns to while it waits for the next ball.
    ready_position: tuple[float, float, float] = (0.0, -1.95, 1.02)
    #: Paddle normal while waiting (pointing at the opponent).
    ready_normal: tuple[float, float, float] = (0.0, 1.0, 0.0)
    #: Joint configuration for :attr:`ready_position`/:attr:`ready_normal`, in
    #: the order of :attr:`joint_names` -- the null-space posture target (see
    #: :class:`ControlConfig`); derived by settling the arm from rest, see
    #: config_provenance.md.
    neutral_qpos: tuple[float, ...] = (0.1527, -0.0388, 0.0557, 0.0006, 0.4639, -0.0330, 2.0016, 0.0554)


@dataclass
class ControlConfig:
    """Gains of the operational-space controller (see :mod:`table_tennis_control.control`)."""

    #: Critically damped task-space PD: ``kd = 2 sqrt(kp)`` for a unit mass.
    #: Leave the ``kd`` fields as ``None`` (the default) to derive them from
    #: ``kp`` automatically -- set one explicitly only to deliberately run a
    #: different damping ratio. See config_provenance.md for the gain sweep
    #: behind these defaults.
    position_kp: tuple[float, float, float] = (5000.0, 5000.0, 5000.0)
    position_kd: tuple[float, float, float] | None = None
    orientation_kp: tuple[float, float, float] = (7200.0, 7200.0, 7200.0)
    orientation_kd: tuple[float, float, float] | None = None
    #: Null-space posture gains (joint space).
    posture_kp: float = 12.0
    posture_kd: float = 6.0
    #: Extra weight of the two base slides in the posture task.
    base_posture_weight: float = 4.0
    #: Regularisation of the operational-space inertia inverse.
    inertia_regularisation: float = 1e-4
    #: Saturation of the operational-space force [N].
    max_task_force: float = 900.0
    #: Error clamps that keep a reference step from turning into an impulse.
    max_position_error: float = 0.25  # [m]
    max_orientation_error: float = 0.8  # [rad]
    #: Include the ``-Lambda * Jdot * qdot`` term of the operational-space law.
    use_jacobian_dot: bool = True

    def __post_init__(self) -> None:
        if self.position_kd is None:
            self.position_kd = _critically_damped(self.position_kp)
        if self.orientation_kd is None:
            self.orientation_kd = _critically_damped(self.orientation_kp)


@dataclass
class CollisionConfig:
    """Artificial-potential-field collision avoidance."""

    enabled: bool = True
    #: Distance below which the repulsive field switches on [m].
    influence_distance: float = 0.12
    #: Distance at which the repulsive force saturates [m].
    saturation_distance: float = 0.02
    #: Gain of the repulsive potential.
    gain: float = 40.0
    #: Damping acting against motion towards the obstacle.
    damping: float = 12.0
    #: Maximum repulsive force per witness point [N].
    max_force: float = 250.0
    #: Cap on the combined joint torque summed over every simultaneously
    #: active (link, obstacle) and (link, link) pair -- a *global* safety net
    #: on top of :attr:`max_force`. Deliberately loose; see
    #: config_provenance.md.
    max_total_torque: float = 150.0
    #: Only recompute the (expensive) distance queries every n control steps.
    decimation: int = 5
    link_geoms: tuple[str, ...] = (
        "base_collider",
        "rotator1_collider",
        "rotator2_collider",
        "arm1_collider",
        "arm2_collider",
        "paddle_rotator_collider",
        "paddle",
    )
    obstacle_geoms: tuple[str, ...] = (
        "table_collider_h",
        "table_collider_v",
        "ground",
    )
    #: Non-adjacent link pairs to exempt from the self-collision potential
    #: field even though they're not directly joined (real hard contact stays
    #: enabled via ``world.xml``). See config_provenance.md for why these
    #: three.
    self_collision_excludes: tuple[tuple[str, str], ...] = (
        ("rotator1_collider", "arm1_collider"),
        ("base_collider", "rotator2_collider"),
        ("arm2_collider", "paddle"),
    )


@dataclass
class ObserverConfig:
    """Extended-state observer for the ball (Lecture 4).

    There is deliberately no ``measurement_noise`` field: this is a
    *fixed-gain* Luenberger observer, not a Kalman filter, and does not
    support artificial measurement noise. See config_provenance.md for why.
    """

    #: Observer bandwidth [rad/s]; the three poles are placed at ``-bandwidth``.
    bandwidth: float = 90.0
    #: Innovation above which the observer re-initialises on a position
    #: teleport (a genuine contact is instead caught by
    #: :attr:`contact_detect_threshold`, see :meth:`BallObserver.update`).
    reset_innovation: float = 0.05
    #: Deviation [m/s] between a raw two-sample finite-difference velocity
    #: and the tracked estimate that flags a contact in progress, see
    #: :meth:`~table_tennis_control.estimation.ball_observer.BallObserver.update`.
    contact_detect_threshold: float = 0.3
    #: Maximum tick-to-tick change [m/s] between consecutive raw
    #: finite-difference velocities required to confirm a contact has
    #: ended and free flight has resumed.
    contact_clear_threshold: float = 0.05


@dataclass
class SensorConfig:
    """Latency of the ball's position sensor -- e.g. a real camera system.

    The delay sits between the plant (the true ball) and everything else:
    the observer, and everything downstream of it, only ever sees where the
    ball *was* ``delay_steps`` control steps ago. The predictor compensates
    for it (:meth:`~table_tennis_control.agent.RallyAgent._estimated_state`),
    the same way a real system would have to compensate for camera latency.
    See :class:`~table_tennis_control.world.ball_sensor.BallSensor`.
    """

    #: Sensing delay in control steps that the predictor compensates for.
    delay_steps: int = 0


@dataclass
class PlannerConfig:
    """High-level strike planner."""

    #: Every return bounces once on the opponent's half before the target,
    #: like a normal table-tennis rally shot.
    #: Nicest-looking post-bounce flight time [s] -- the centre of
    #: :attr:`post_bounce_times` below (swept; see config_provenance.md).
    preferred_bounce_time: float = 0.70
    #: How far below/above :attr:`preferred_bounce_time` the candidate
    #: post-bounce flight-time grid reaches, and its step [s].  Asymmetric
    #: because the target sits on the floor, well below the bounce height,
    #: so the grid needs much more reach towards a long, lofted arc than
    #: towards a short, flat one.
    post_bounce_time_lower_spread: float = 0.15
    post_bounce_time_upper_spread: float = 0.75
    post_bounce_time_step: float = 0.1
    #: Candidate post-bounce flight times swept by the return solver [s],
    #: generated from :attr:`preferred_bounce_time` and the spreads above
    #: (see :meth:`__post_init__`) so the two can never drift out of sync.
    #: Do not set this directly -- tune the three fields above instead.
    post_bounce_times: tuple[float, ...] = field(init=False)
    #: Planner update rate [Hz]; the low-level controller runs at the sim rate.
    replan_rate: float = 50.0
    #: Once the impact is closer than this the plan is frozen (no more jerks)
    #: [s] (measured; see config_provenance.md).
    commit_horizon: float = 0.22
    #: Never aim for an interception that is closer in time than this [s].
    min_time_to_impact: float = 0.23
    #: Interception candidates are only taken from the ball's flight up to here [s].
    max_time_to_impact: float = 1.5
    #: Sub-sampling of the predicted trajectory when searching for a strike point.
    candidate_stride: int = 4
    #: Safety margins used when validating a bounce point [m].
    bounce_margin: float = 0.12
    #: Extra height the return has to clear the net by [m].
    net_clearance: float = 0.08
    #: Shortest time between the impact and the bounce [s].
    min_time_to_bounce: float = 0.30
    #: Only accept an interception if the arm needs less than this fraction of
    #: the available time -- leaves room for the prediction to move around.
    time_margin: float = 0.85
    #: The same margin while *refining* an already committed stroke; the arm is
    #: already moving, so it may be more optimistic here.
    refine_time_margin: float = 0.95
    #: Cost weights.  ``weight_move_time`` prefers the point the arm reaches
    #: fastest, ``weight_impact_time`` breaks ties towards hitting early --
    #: negative here instead rewards a *later* impact. Kept small so it only
    #: nudges close ties rather than overriding them.
    weight_move_time: float = 1.0
    weight_impact_time: float = -0.15
    weight_paddle_speed: float = 0.06
    weight_ready_distance: float = 0.15
    weight_bounce_time: float = 0.4
    #: Penalises switching the target away from the previously planned impact
    #: position [1/m], while still *not planned* (fixes replan-tick target
    #: jitter; see config_provenance.md).
    weight_continuity: float = 0.5
    #: Reject solutions whose required paddle speed exceeds this [m/s].
    max_paddle_speed: float = 20.0
    #: Reject returns faster than this [m/s].
    max_outgoing_speed: float = 18.0
    #: Duration of the follow-through after the impact [s].
    follow_through_time: float = 0.36
    #: Distance the paddle keeps travelling after the impact [m].
    follow_through_distance: float = 0.20

    def __post_init__(self) -> None:
        self.post_bounce_times = _centered_grid(
            self.preferred_bounce_time,
            self.post_bounce_time_lower_spread,
            self.post_bounce_time_upper_spread,
            self.post_bounce_time_step,
        )


@dataclass
class LauncherConfig:
    """Serves the incoming balls from the opponent's side."""

    #: Launch position range ``(x, y, z)`` on the opponent side.
    x_range: tuple[float, float] = (-0.55, 0.55)
    y_range: tuple[float, float] = (1.9, 2.5)
    z_range: tuple[float, float] = (0.95, 1.35)
    #: Where the serve is supposed to bounce on the robot's half.
    bounce_x_range: tuple[float, float] = (-0.5, 0.5)
    bounce_y_range: tuple[float, float] = (-1.15, -0.35)
    #: Time of flight from the launcher to that bounce point [s] -- the main
    #: speed knob (shorter time == faster serve). See config_provenance.md
    #: for the reliability tradeoff behind this range.
    flight_time_range: tuple[float, float] = (0.65, 0.95)
    #: Alternate, shorter flight-time range used for a fraction of serves
    #: (see :attr:`shallow_serve_probability`) to get flatter launch angles,
    #: passing closer over the net. See config_provenance.md.
    shallow_flight_time_range: tuple[float, float] = (0.50, 0.75)
    #: Fraction of serves that use :attr:`shallow_flight_time_range` instead
    #: of :attr:`flight_time_range`, for a wider mix of launch angles.
    shallow_serve_probability: float = 0.35
    #: Seconds between two serves.
    serve_interval: float = 3.5
    #: Fail the serve rather than launching it into the net.
    net_clearance: float = 0.05


@dataclass
class TargetConfig:
    """Where the robot is supposed to place the ball.

    The target always sits on the floor behind the opponent's half.  Its
    distance behind the table's back edge is expressed as a *margin* rather
    than an absolute coordinate, so it is always well clear of the table
    regardless of :attr:`TableSpec.half_length`.
    """

    #: Extra distance behind the table's back edge, sampled uniformly [m].
    floor_margin_range: tuple[float, float] = (1.0, 2.2)
    floor_x_range: tuple[float, float] = (-0.8, 0.8)
    #: Draw a new target for every rally.
    resample_per_serve: bool = True
    #: Minimum horizontal gap [m] enforced between the target and the centre
    #: line -- the target is always sampled from the *opposite* half of
    #: :attr:`floor_x_range` from the serve's launch site (see
    #: :meth:`TargetSampler.sample`; see config_provenance.md).
    cross_court_gap: float = 0.3


@dataclass
class VisualisationConfig:
    """In-window overlay (no extra matplotlib window, no line charts)."""

    enabled: bool = True
    #: Overlay refresh rate [Hz].
    update_rate: float = 24.0
    show_predicted_trajectory: bool = True
    show_plan: bool = True
    show_actual_path: bool = True


@dataclass
class SimulationConfig:
    """Everything the applications need to build a simulation."""

    table: TableSpec = field(default_factory=TableSpec)
    ball: BallSpec = field(default_factory=BallSpec)
    arm: ArmSpec = field(default_factory=ArmSpec)
    control: ControlConfig = field(default_factory=ControlConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    visualisation: VisualisationConfig = field(default_factory=VisualisationConfig)
    seed: int | None = None
    real_time: bool = True
