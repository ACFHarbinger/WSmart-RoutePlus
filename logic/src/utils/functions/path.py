"""
Path manipulation utilities.

Attributes:
    get_path_until_string: Truncates a path up to a specific directory component.

Example:
    >>> from logic.src.utils.functions import get_path_until_string
    >>> path = "/home/user/project/src/module.py"
    >>> get_path_until_string(path, "src")
    '/home/user/project/src'
"""

from __future__ import annotations

import os
import re
from typing import Optional

# Characters that separate path components here: "/" only on POSIX, both "\\" and "/" on Windows
_SEPARATORS = re.escape(os.sep + (os.altsep or ""))


def get_path_until_string(path: str, end_str: str) -> Optional[str]:
    """
    Truncates a path up to the first occurrence of a specific directory component.

    Args:
        path: The full path.
        end_str: The directory name to truncate at.

    Returns:
        The truncated path or None if end_str is not found. The path keeps its own separators.
    """
    for component in re.finditer(f"[^{_SEPARATORS}]+", path):
        if component.group() == end_str:
            return path[: component.end()]
    print(f"Path '{path}' does not contain '{end_str}'")
    return None
