"""Loading of the MuJoCo model and the handles that go with it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ..config import WORLD_XML, SimulationConfig
from .ball import Ball
from .target import Target, TargetSampler


@dataclass
class Scene:
    """Bundles the MuJoCo model/data with the objects living inside it."""

    model: mujoco.MjModel # type: ignore
    data: mujoco.MjData # type: ignore
    ball: Ball
    target: Target
    target_sampler: TargetSampler
    config: SimulationConfig
    generator: np.random.Generator

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def gravity(self) -> np.ndarray:
        return np.asarray(self.model.opt.gravity, dtype=float)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data) # type: ignore


def load_scene(config: SimulationConfig | None = None, xml_path: str | Path | None = None, viewport: tuple[int, int] | None = None) -> Scene:
    """Build a :class:`Scene` from the packaged world description.

    The model's offscreen framebuffer (``<visual><global offwidth=.../>``)
    is a compile-time size that a renderer cannot exceed, so it is grown 
    here to fit rather than leaving oversized requests to fail deep inside 
    :class:`mujoco.Renderer`.

    Args:
        viewport: ``(width, height)`` the caller intends to render at. 
    """
    config = config or SimulationConfig()
    model = mujoco.MjModel.from_xml_path(str(xml_path or WORLD_XML)) # type: ignore
    data = mujoco.MjData(model) # type: ignore

    # Adjust the offscreen framebuffer size if a viewport is specified
    if viewport is not None:
        width, height = viewport
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))

    generator = np.random.default_rng(config.seed)
    ball = Ball(model, data)
    target = Target(model, data)
    sampler = TargetSampler(config.table, config.ball, config.target, generator)

    target.set_position(sampler.sample())
    mujoco.mj_forward(model, data) # type: ignore

    return Scene(
        model=model,
        data=data,
        ball=ball,
        target=target,
        target_sampler=sampler,
        config=config,
        generator=generator,
    )
