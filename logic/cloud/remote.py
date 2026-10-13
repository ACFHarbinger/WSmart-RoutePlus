"""Shared plumbing for the Drive sync: sync targets, remote paths and the rclone call.

Attributes:
    TARGETS: The synchronised directories (``data``, ``output``, ``tracking``), keyed by their CLI name.
    DEFAULT_REMOTE: Drive root used when no remote is given (``rclone`` remote name and folder).

Example:
    >>> join_remote("gdrive:WSmart-Route", "data")
    'gdrive:WSmart-Route/data'
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

RCLONE = "rclone"
DEFAULT_REMOTE = "gdrive:WSmart-Route"


@dataclass(frozen=True)
class SyncTarget:
    """A local directory (relative to the repository root) and its folder on the remote.

    Attributes:
        local: Directory relative to the repository root.
        remote_dir: Sub-folder of the remote root that mirrors ``local``.
    """

    local: Path
    remote_dir: str


TARGETS = {
    "data": SyncTarget(local=Path("data"), remote_dir="data"),
    "output": SyncTarget(local=Path("outputs/simulator"), remote_dir="output"),
    "tracking": SyncTarget(local=Path("assets/tracking"), remote_dir="tracking"),
}


def repository_root() -> Path:
    """Return the repository root (the directory that contains ``logic/``)."""
    return Path(__file__).resolve().parents[2]


def join_remote(remote: str, remote_dir: str) -> str:
    """Append a folder to an rclone remote path.

    Args:
        remote: rclone path such as ``gdrive:WSmart-Route``.
        remote_dir: Folder inside the remote root.

    Returns:
        The combined rclone path.
    """
    return f"{remote.rstrip('/')}/{remote_dir}"


def local_directory(target: str, *, root: Optional[Path] = None) -> Path:
    """Return the local directory of a sync target, refusing symlinks.

    A symlinked directory is refused on both sides of the sync, so a copy never
    reads from or writes through a link to some other location.

    Args:
        target: Key of ``TARGETS``.
        root: Repository root (defaults to :func:`repository_root`).

    Returns:
        The absolute, non-symlinked local directory.

    Raises:
        KeyError: If ``target`` is unknown.
        ValueError: If the directory, or one of its parents below the root, is a symlink.
    """
    spec = TARGETS[target]
    base = (root or repository_root()).resolve()
    path = base / spec.local
    for part in [path, *path.parents]:
        if part.is_symlink():
            raise ValueError(f"{part} is a symlink; sync the real directory instead")
        if part == base:
            break
    return path


def run_rclone(
    source: str,
    destination: str,
    *,
    dry_run: bool = False,
    excludes: Sequence[str] = (),
    transfers: int = 4,
    config: Optional[str] = None,
) -> List[str]:
    """Copy ``source`` to ``destination`` with rclone, comparing files by checksum.

    ``rclone copy`` never deletes files on either side. Symlinks are skipped, not followed.

    Args:
        source: rclone path or local directory to copy from.
        destination: rclone path or local directory to copy to.
        dry_run: Report what would be copied without changing anything.
        excludes: rclone exclude patterns.
        transfers: Parallel transfers.
        config: Optional rclone configuration file.

    Returns:
        The rclone command line that was run.

    Raises:
        RuntimeError: If rclone is not installed.
        subprocess.CalledProcessError: If rclone fails.
    """
    if shutil.which(RCLONE) is None:
        raise RuntimeError("rclone is not installed; see https://rclone.org/install/")
    argv = [
        RCLONE,
        "copy",
        source,
        destination,
        "--checksum",
        "--skip-links",
        "--create-empty-src-dirs",
        "--transfers",
        str(transfers),
        "--log-level",
        "INFO",
    ]
    for pattern in excludes:
        argv += ["--exclude", pattern]
    if dry_run:
        argv.append("--dry-run")
    if config:
        argv += ["--config", config]
    subprocess.run(argv, check=True)
    return argv
