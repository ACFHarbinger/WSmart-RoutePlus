"""Locate policy YAML files in ``logic/configs/policies``.

The folder is organised like ``logic/src/policies``: routing algorithms under
``routing/<category>/policy_<key>.yaml``, joint policies under ``packing_and_routing/``, and the
stage files under ``other/packing/``, ``other/route_improvement/`` and
``other/acceptance_criteria/``. File names are unique, so a bare name or a pre-reorganisation
flat path (``policy_alns.yaml``, ``other/pk_last_minute.yaml``) still resolves.

Attributes:
    policies_dir: The canonical ``logic/configs/policies`` folder.
    find_policy_file: The file a key names, or ``None``.
    policy_file: The file a key names, or the direct join when there is none.
    routing_policy_files: Every ``policy_<key>.yaml`` by routing key.

Example:
    >>> policy_file("other/pk_last_minute.yaml").relative_to(policies_dir()).as_posix()
    'other/packing/pk_last_minute.yaml'
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union

PathLike = Union[str, "os.PathLike[str]"]


def policies_dir() -> Path:
    """The canonical ``logic/configs/policies`` folder."""
    from logic.src.constants.paths import CONFIGS_DIR

    return Path(CONFIGS_DIR) / "policies"


@lru_cache(maxsize=16)
def _index(root: str) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(Path(root).rglob("*.yaml")):
        index.setdefault(path.name, path)
    return index


def find_policy_file(file_key: PathLike, root: Optional[PathLike] = None) -> Optional[Path]:
    """The policy YAML ``file_key`` names under ``root``: its relative path, else its file name.

    Args:
        file_key: A path relative to the policies folder, or a file name.
        root: The folder to search (default: :func:`policies_dir`).

    Returns:
        The file, or ``None`` when nothing matches.
    """
    base = Path(root) if root is not None else policies_dir()
    direct = base / file_key
    if direct.is_file():
        return direct
    return _index(str(base)).get(Path(file_key).name)


def policy_file(file_key: PathLike, root: Optional[PathLike] = None) -> Path:
    """Like :func:`find_policy_file`, but the direct join when nothing matches (callers check existence)."""
    base = Path(root) if root is not None else policies_dir()
    return find_policy_file(file_key, base) or base / file_key


def routing_policy_files(root: Optional[PathLike] = None) -> Dict[str, Path]:
    """Every ``policy_<key>.yaml`` under ``root`` by routing key."""
    base = Path(root) if root is not None else policies_dir()
    return {p.stem[len("policy_") :]: p for p in sorted(base.rglob("policy_*.yaml"))}
