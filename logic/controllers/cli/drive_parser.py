"""CLI parser for the Google Drive sync (``sync_drive`` sub-command).

Example::

    python main.py sync_drive --direction upload --target all --dry-run
    python main.py sync_drive --direction download --target data

The sync is copy-only through rclone; see :mod:`logic.cloud` for the remote setup.

Attributes:
    add_drive_sync_args: Register ``sync_drive`` arguments on an existing sub-parser.
"""

from __future__ import annotations

import argparse

from logic.cloud.remote import DEFAULT_REMOTE, TARGETS


def add_drive_sync_args(parser: argparse.ArgumentParser) -> None:
    """Register ``sync_drive`` arguments on *parser*.

    Args:
        parser: An :class:`argparse.ArgumentParser` (or sub-parser) to populate.
    """
    parser.add_argument(
        "--direction",
        required=True,
        choices=["upload", "download"],
        help="upload: local directories to Drive; download: Drive folders to local.",
    )
    parser.add_argument(
        "--target",
        default="all",
        choices=[*sorted(TARGETS), "all"],
        help="Which directory to sync (default: all).",
    )
    parser.add_argument(
        "--remote",
        default=DEFAULT_REMOTE,
        help=f"rclone path of the Drive root folder (default: {DEFAULT_REMOTE}).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without copying.")
    parser.add_argument("--exclude", action="append", default=[], help="rclone exclude pattern (repeatable).")
    parser.add_argument("--config", default=None, help="rclone configuration file.")
