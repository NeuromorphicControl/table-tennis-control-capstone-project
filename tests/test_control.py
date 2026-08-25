"""Controller, trajectory generation and the closed loop in simulation."""

from __future__ import annotations

import numpy as np
import pytest

from table_tennis_control.agent import Phase, RallyAgent
from table_tennis_control.config import SimulationConfig
from table_tennis_control.control import RobotArm
from table_tennis_control.kinematics import axis_alignment_error, orientation_from_normal, rotation_aligning
from table_tennis_control.planning import GoToTrajectory, QuinticSegment, SwingTrajectory, TaskState
from table_tennis_control.world import load_scene


@pytest.fixture(scope="module")
def config() -> SimulationConfig:
    return SimulationConfig(seed=7)


def make_arm(config: SimulationConfig):
    scene = load_scene(config)
    arm = RobotArm(scene.model, scene.data, config.arm, config.control, config.collision)
    return scene, arm


# --------------------------------------------------------------------- maths
def test_quintic_matches_its_boundary_conditions():
    start_position = np.array([0.0, 0.0, 0.0])
    start_velocity = np.array([0.1, -0.2, 0.0])
    end_position = np.array([0.4, 0.5, -0.2])
    end_velocity = np.array([1.0, 0.0, 0.5])
    segment = QuinticSegment(
        start_position, start_velocity, np.zeros(3), end_position, end_velocity, np.zeros(3), 0.4
    )

    position, velocity, acceleration = segment.evaluate(0.0)
    assert np.allclose(position, start_position)
    assert np.allclose(velocity, start_velocity)
    assert np.allclose(acceleration, 0.0)

    position, velocity, acceleration = segment.evaluate(0.4)
    assert np.allclose(position, end_position, atol=1e-9)
    assert np.allclose(velocity, end_velocity, atol=1e-9)
    assert np.allclose(acceleration, 0.0, atol=1e-8)


def test_quintic_derivatives_are_consistent():
    segment = QuinticSegment(
        np.zeros(3), np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), np.zeros(3), np.zeros(3), 1.0
    )
    step = 1e-5
    for time in (0.2, 0.5, 0.8):
        _, velocity, acceleration = segment.evaluate(time)
        ahead, velocity_ahead, _ = segment.evaluate(time + step)
        behind, velocity_behind, _ = segment.evaluate(time - step)
        assert np.allclose(velocity, (ahead - behind) / (2 * step), atol=1e-4)
        assert np.allclose(acceleration, (velocity_ahead - velocity_behind) / (2 * step), atol=1e-4)


def test_rotation_aligning_maps_source_onto_target():
    generator = np.random.default_rng(3)
    for _ in range(50):
        source = generator.normal(size=3)
        target = generator.normal(size=3)
        source /= np.linalg.norm(source)
        target /= np.linalg.norm(target)
        rotation = rotation_aligning(source, target)
        assert np.allclose(rotation @ source, target, atol=1e-9)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)


def test_rotation_aligning_handles_antiparallel_vectors():
    rotation = rotation_aligning(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]))
    assert np.allclose(rotation @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0], atol=1e-9)


def test_axis_alignment_error_vanishes_when_aligned():
    normal = np.array([0.0, 1.0, 0.0])
    rotation = orientation_from_normal(normal, np.eye(3), normal_axis=1)
    assert np.allclose(axis_alignment_error(rotation, normal, normal_axis=1), 0.0, atol=1e-9)


def test_axis_alignment_error_has_no_spin_component():
    """The error must never command a rotation about the paddle normal."""
    generator = np.random.default_rng(5)
    for _ in range(50):
        desired = generator.normal(size=3)
        desired /= np.linalg.norm(desired)
        rotation = orientation_from_normal(generator.normal(size=3), np.eye(3), 1)
        error = axis_alignment_error(rotation, desired, 1)
        current_normal = rotation[:, 1]
        assert abs(float(np.dot(error, current_normal))) < 1e-9


# ---------------------------------------------------------------- controller
def test_controller_holds_its_pose(config):
    scene, arm = make_arm(config)
    reference = arm.hold()
    for _ in range(1500):
        diagnostics = arm.update(reference)
        scene.step()
    assert np.all(np.isfinite(scene.data.qvel))
    assert diagnostics.position_error < 5e-3


def test_controller_tracks_a_fast_reach(config):
    """A 0.5 m minimum-jerk reach in 0.35 s stays within a few millimetres."""
    scene, arm = make_arm(config)
    start = arm.measure()
    goal = start.position + np.array([0.3, 0.2, -0.3])
    trajectory = GoToTrajectory(start, goal, np.array([0.0, 1.0, 0.0]), duration=0.35, start_time=scene.time)

    worst = 0.0
    while scene.time < 0.9:
        diagnostics = arm.update(trajectory.evaluate(scene.time))
        scene.step()
        worst = max(worst, diagnostics.position_error)

    assert worst < 0.02
    assert np.linalg.norm(arm.position - goal) < 5e-3
    assert np.allclose(arm.normal, [0.0, 1.0, 0.0], atol=0.05)


def test_controller_survives_a_reference_step(config):
    """A jump in the reference must not blow the simulation up."""
    scene, arm = make_arm(config)
    reference = TaskState(
        np.array([0.2, -1.6, 1.05]), np.zeros(3), np.zeros(3), np.array([0.0, 1.0, 0.0]), np.zeros(3)
    )
    for _ in range(3000):
        arm.update(reference)
        scene.step()
    assert np.all(np.isfinite(scene.data.qvel))
    assert np.linalg.norm(arm.position - reference.position) < 0.02


def test_swing_arrives_with_the_requested_velocity(config):
    scene, arm = make_arm(config)
    start = arm.measure()
    impact_position = start.position + np.array([0.15, 0.2, 0.1])
    impact_velocity = np.array([0.0, 1.2, 0.4])
    swing = SwingTrajectory(
        start,
        start_time=scene.time,
        impact_time=scene.time + 0.4,
        impact_position=impact_position,
        impact_velocity=impact_velocity,
        impact_normal=np.array([0.0, 1.0, 0.2]),
    )

    impact_time = swing.impact_time
    while scene.time < impact_time:
        arm.update(swing.evaluate(scene.time))
        scene.step()

    measured = arm.measure()
    assert np.linalg.norm(measured.position - impact_position) < 0.01
    assert np.linalg.norm(measured.velocity - impact_velocity) < 0.15


def test_null_space_posture_does_not_disturb_the_task(config):
    """The posture task must be invisible to the end effector."""
    scene, arm = make_arm(config)
    start = arm.measure()
    trajectory = GoToTrajectory(
        start, start.position + np.array([0.2, -0.2, 0.1]), np.array([0.0, 1.0, 0.0]), 0.4, scene.time
    )
    with_posture = 0.0
    while scene.time < 0.8:
        with_posture = max(with_posture, arm.update(trajectory.evaluate(scene.time)).orientation_error)
        scene.step()

    config.control.posture_kp, config.control.posture_kd = 0.0, 0.0
    scene, arm = make_arm(config)
    start = arm.measure()
    trajectory = GoToTrajectory(
        start, start.position + np.array([0.2, -0.2, 0.1]), np.array([0.0, 1.0, 0.0]), 0.4, scene.time
    )
    without_posture = 0.0
    while scene.time < 0.8:
        without_posture = max(without_posture, arm.update(trajectory.evaluate(scene.time)).orientation_error)
        scene.step()
    config.control.posture_kp, config.control.posture_kd = 12.0, 6.0

    assert with_posture < without_posture + 0.02


# --------------------------------------------------------------------- agent
def test_agent_returns_serves_onto_the_opponent_half():
    config = SimulationConfig(seed=2)
    scene = load_scene(config)
    arm = RobotArm(scene.model, scene.data, config.arm, config.control, config.collision)
    agent = RallyAgent(scene, arm, config)

    phases = set()
    while scene.time < 16.0:
        agent.maybe_serve()
        phases.add(agent.step().phase)

    statistics = agent.statistics
    assert statistics.serves >= 3
    assert Phase.SWING in phases
    assert statistics.strikes >= 2
    assert statistics.on_target_half >= 1
    assert np.all(np.isfinite(scene.data.qvel))


def test_agent_still_returns_serves_under_a_delayed_sensor():
    """The predictor's delay compensation should keep strikes working under sensor latency."""
    config = SimulationConfig(seed=2)
    config.sensor.delay_steps = 8
    scene = load_scene(config)
    arm = RobotArm(scene.model, scene.data, config.arm, config.control, config.collision)
    agent = RallyAgent(scene, arm, config)

    while scene.time < 16.0:
        agent.maybe_serve()
        agent.step()

    statistics = agent.statistics
    assert statistics.strikes >= 2
    assert statistics.on_target_half >= 1
    assert np.all(np.isfinite(scene.data.qvel))


def test_standard_returns_bounce_on_the_opponents_half_first():
    """A standard return must touch the opponent's half before the floor target."""
    config = SimulationConfig(seed=4)
    scene = load_scene(config)
    arm = RobotArm(scene.model, scene.data, config.arm, config.control, config.collision)
    agent = RallyAgent(scene, arm, config)

    while scene.time < 10.0:
        agent.maybe_serve()
        diagnostics = agent.step()
        plan = diagnostics.plan
        if plan is not None and plan.bounce_point is not None:
            assert plan.bounce_point[1] * config.table.opponent_side > 0.0
            assert abs(plan.bounce_point[0]) <= config.table.half_width
            assert plan.landing_point[1] * config.table.opponent_side > 0.0
            assert plan.landing_point[2] == pytest.approx(config.ball.radius)
            return
    pytest.fail("no stroke was planned")
