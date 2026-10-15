"""
Configuration constants and global mappings.

Attributes:
    data: Data configuration constants
    hpo: Hyperparameter optimization configuration constants
    models: Model configuration constants
    paths: Path configuration constants
    routing: Routing configuration constants
    simulation: Simulation configuration constants
    stats: Statistical function mappings
    system: System configuration constants
    testing: Testing configuration constants
    user_interface: User interface configuration constants
    waste: Waste configuration constants

Example:
    >>> from logic.src.constants import *
    >>> ROOT_DIR
    PosixPath('~/Repositories/WSmart-Route')
"""

from __future__ import annotations

# Re-exporting from split modules
from .data import *  # noqa: F403
from .hpo import *  # noqa: F403
from .models import *  # noqa: F403
from .paths import *  # noqa: F403
from .plotting import *  # noqa: F403
from .routing import *  # noqa: F403
from .simulation import *  # noqa: F403
from .stats import *  # noqa: F403
from .system import *  # noqa: F403
from .temporal import *  # noqa: F403
from .testing import *  # noqa: F403
from .user_interface import *  # noqa: F403
from .waste import *  # noqa: F403
