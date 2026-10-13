"""Runner behind the ``sync_drive`` command of ``main.py``.

Example:
    python main.py sync_drive --direction upload --target data --dry-run
    python main.py sync_drive --direction download --target output --remote gdrive:WSmart-Route
"""

from __future__ import annotations

from typing import Any, Dict, List

from logic.cloud.downloader.sync import download
from logic.cloud.remote import DEFAULT_REMOTE, TARGETS
from logic.cloud.uploader.sync import upload


def run_sync(opts: Dict[str, Any]) -> List[List[str]]:
    """Run an upload or download for the requested targets.

    Args:
        opts: Parsed ``sync_drive`` options (``direction``, ``target``, ``remote``,
            ``dry_run``, ``exclude``, ``config``).

    Returns:
        The rclone command line of each sync that was run.
    """
    action = upload if opts["direction"] == "upload" else download
    targets = sorted(TARGETS) if opts["target"] == "all" else [opts["target"]]
    commands = []
    for target in targets:
        commands.append(
            action(
                target,
                opts.get("remote") or DEFAULT_REMOTE,
                dry_run=bool(opts.get("dry_run")),
                excludes=opts.get("exclude") or (),
                config=opts.get("config"),
            )
        )
    return commands
