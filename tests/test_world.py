"""Scene, launcher, target sampling and collision configuration."""

from __future__ import annotations

import numpy as np
import pytest

from table_tennis_control.config import SimulationConfig
from table_tennis_control.estimation import BallPredictor
from table_tennis_control.kinematics import point_in_box
from table_tennis_control.world import BallLauncher, load_scene


@pytest.fixture(scope="module")
def scene():
    return load_scene(SimulationConfig(seed=11))


def test_model_loads_with_the_expected_actuators(scene):
    config = scene.config
    assert scene.model.nu == len(config.arm.joint_names)
    for name in config.arm.joint_names:
        assert scene.model.actuator(name) is not None


def test_table_geometry_matches_the_specification(scene):
    """The analytic table model has to agree with the MuJoCo one."""
    table = scene.config.table
    collider = scene.model.geom("table_collider_h")
    half_sizes = np.asarray(collider.size, dtype=float)
    assert half_sizes[0] == pytest.approx(table.half_width)
    assert half_sizes[1] == pytest.approx(table.half_length)
    top = float(scene.model.geom_pos[collider.id][2]) + half_sizes[2]
    assert top == pytest.approx(table.height)

    net = scene.model.geom("table_collider_v")
    net_top = float(scene.model.geom_pos[net.id][2]) + float(net.size[2])
    assert net_top == pytest.approx(table.net_height)


def test_arm_links_collide_with_the_table_and_the_ball(scene):
    """Collision masks: links vs environment yes, links vs ball no."""
    model = scene.model

    def collides(first: str, second: str) -> bool:
        one, other = model.geom(first), model.geom(second)
        return bool(one.contype & other.conaffinity) or bool(other.contype & one.conaffinity)

    assert collides("arm1_collider", "table_collider_h")
    assert collides("arm2_collider", "ground")
    assert not collides("arm1_collider", "ball_geom")
    assert collides("paddle", "ball_geom")


def test_links_are_bitmasked_for_self_collision(scene):
    """The link/paddle bitmasks enable self-collision in general.

    Whether a *specific* pair actually generates a contact also depends on
    the ``<exclude>`` list for directly-joined neighbours (see
    :func:`test_ready_pose_has_no_spurious_self_contact`), which contype and
    conaffinity alone can't express -- every link proxy shares the same
    class, so this only checks the bits, not any one pair.
    """
    model = scene.model
    link = model.geom("arm1_collider")
    paddle = model.geom("paddle")
    assert link.contype & link.conaffinity, "links must be able to collide with other links"
    assert paddle.contype & link.conaffinity and link.contype & paddle.conaffinity


def test_ready_pose_has_no_spurious_self_contact(scene):
    """Directly-joined links must not be in constant contact at the ready pose."""
    import mujoco

    data = scene.data
    mujoco.mj_resetData(scene.model, data) # type: ignore
    mujoco.mj_forward(scene.model, data) # type: ignore
    self_contacts = [
        (int(c.geom1), int(c.geom2))
        for c in data.contact[: data.ncon]
        if scene.model.geom(int(c.geom1)).name != "ground" and scene.model.geom(int(c.geom2)).name != "ground"
    ]
    assert not self_contacts


def test_collision_proxies_are_massless(scene):
    """The added colliders must not change the dynamics of the arm."""
    total = float(scene.model.body_subtreemass[scene.model.body("base").id])
    assert total == pytest.approx(14.25, abs=1e-6)


def test_target_sampler_stays_reachably_clear_of_the_table(scene):
    """Targets sit on the floor, always the configured margin past the table's
    back edge -- otherwise the return would have to clip through the tabletop
    to reach them.
    """
    table = scene.config.table
    config = scene.config.target
    min_margin = config.floor_margin_range[0]
    for _ in range(200):
        target = scene.target_sampler.sample()
        assert target[2] == pytest.approx(scene.config.ball.radius)
        assert target[1] * table.opponent_side >= table.half_length + min_margin - 1e-9
        assert config.floor_x_range[0] <= target[0] <= config.floor_x_range[1]


class TestLauncher:
    def test_serves_start_on_the_opponent_side_and_fly_towards_the_robot(self, scene):
        config = scene.config
        launcher = BallLauncher(
            scene.ball, config.table, config.ball, config.launcher, scene.generator, config.arm.workspace
        )
        for _ in range(30):
            serve = launcher.plan_serve()
            assert serve.position[1] * config.table.opponent_side > 0.0
            assert serve.velocity[1] * config.table.robot_side > 0.0
            assert bool(config.table.is_on_half(serve.bounce_point, config.table.robot_side))

    def test_serves_are_playable(self, scene):
        """Every serve has to clear the net and cross the robot's strike zone."""
        config = scene.config
        launcher = BallLauncher(
            scene.ball, config.table, config.ball, config.launcher, scene.generator, config.arm.workspace
        )
        predictor = BallPredictor(config.table, config.ball, sample_time=0.005, horizon=3.0)

        playable = 0
        attempts = 25
        for _ in range(attempts):
            serve = launcher.plan_serve()
            trajectory = predictor.predict(serve.position, serve.velocity)
            assert trajectory.events, "the serve has to hit something"
            assert trajectory.events[0].surface == "table"

            inside = point_in_box(trajectory.positions, config.arm.workspace)
            playable += int(np.any(inside))
        assert playable >= attempts - 2
