"""A table tennis playing robot arm, controlled in operational space.

The package is organised along the control architecture of the lecture:

``table_tennis_control.world``
    the plant -- MuJoCo model, ball, target and the ball launcher
``table_tennis_control.estimation``
    observer and forward model (the internal model of the ball)
``table_tennis_control.planning``
    the outer loop -- return solver, strike planner, reference trajectories
``table_tennis_control.control``
    the inner loop -- operational-space control, feedback linearisation and
    collision avoidance
``table_tennis_control.visualization``
    the in-window overlay and head-up display
``table_tennis_control.agent``
    glues all of the above into one rally-playing agent
"""

from .agent import AgentDiagnostics, Phase, RallyAgent, RallyStatistics
from .config import SimulationConfig
from .control import RobotArm
from .world import load_scene

__version__ = "1.0.0"

__all__ = [
    "AgentDiagnostics",
    "Phase",
    "RallyAgent",
    "RallyStatistics",
    "RobotArm",
    "SimulationConfig",
    "load_scene",
    "__version__",
]
