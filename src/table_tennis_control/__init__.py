"""Table-tennis-playing robot arm package, organised into world, estimation, planning, control and visualization subpackages tied together by the top-level rally agent."""

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
