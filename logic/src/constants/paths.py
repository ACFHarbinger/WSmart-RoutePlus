"""
Path constants for the project.

This module provides platform-independent path resolution for the WSmart+ Route
project root and asset locations. Used by:
- Application icon loading for exported tooling
- All modules needing project-relative paths (configs, data, outputs)
- CLI entry points (main.py) for workspace detection

Root Directory Resolution
--------------------------
The ROOT_DIR is dynamically computed by searching upward from cwd until finding
the project root marker directory name. This supports:
- Running from any subdirectory (notebooks/, logic/test/, gui/)
- Multiple project clones (WSmart-Route, WSmart-Simulator)
- Virtual environment isolation (paths work regardless of venv location)

Path Resolution Order:
1. Get current working directory
2. Search upward for "WSmart-Route" or "WSmart-Simulator" in path parts
3. Set ROOT_DIR to that location
4. Derive asset paths relative to ROOT_DIR

Critical Files
--------------
- ICON_FILE: GUI application icon (used in window title bar, taskbar)

Attributes:
    path: Current working directory
    parts: Split path into components
    root_dir: Project root directory (absolute path)
    ROOT_DIR: Project root directory (absolute path)
    ICON_FILE: GUI application icon (used in window title bar, taskbar)

Example:
    >>> from logic.src.constants import ROOT_DIR, ICON_FILE
    >>> ROOT_DIR
    PosixPath('/home/user/Repositories/WSmart-Route')
    >>> ICON_FILE
    '/home/user/Repositories/WSmart-Route/assets/images/logo-wsmartroute-white.png'
"""

import os
import sys
from pathlib import Path

def _frozen_root() -> Path:
    """Project root for a PyInstaller build.

    The bundle unpacks code to sys._MEIPASS, but data/simulator and the outputs
    (assets/, logs) belong to the user's working folder: $WSMART_ROUTE_ROOT if set,
    else the current directory when it holds data/, else the executable's folder.
    """
    env_root = os.environ.get("WSMART_ROUTE_ROOT")
    if env_root:
        return Path(env_root).expanduser().absolute()
    if (Path.cwd() / "data").is_dir():
        return Path.cwd().absolute()
    return Path(sys.executable).parent.absolute()


# Dynamic root directory resolution: the directory that contains logic/.
if getattr(sys, "frozen", False):
    root_dir = _frozen_root()
else:
    parts: tuple[str, ...] = Path(__file__).parent.absolute().parts
    try:
        root_dir = Path(*parts[: parts.index("logic")]).absolute()
    except ValueError:
        root_dir = Path(*parts[:-3]).absolute()

# Project root directory (absolute path)
# Example: /home/user/Repositories/WSmart-Route
ROOT_DIR: Path = root_dir

# Hydra configurations directory — absolute so @hydra.main resolves correctly
# regardless of which subdirectory the entry-point file lives in.
# paths.py is at logic/src/constants/paths.py, so .parent×3 == logic/
CONFIGS_DIR: str = str(Path(__file__).parent.parent.parent / "configs")

# Application icon (PNG format, white logo on transparent background)
# Used by exported tooling and report generators
# Dimensions: 512x512 px (scales down for UI)
ICON_FILE: str = os.path.join(ROOT_DIR, "assets", "images", "logo-wsmartroute-white.png")
