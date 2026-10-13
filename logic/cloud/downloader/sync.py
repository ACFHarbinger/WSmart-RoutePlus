"""Download a Google Drive folder into its local directory (copy only, nothing is deleted).

Example:
    python -m logic.cloud.downloader.sync --target output --dry-run
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from logic.cloud.remote import DEFAULT_REMOTE, TARGETS, join_remote, local_directory, run_rclone


def download(
    target: str,
    remote: str = DEFAULT_REMOTE,
    *,
    dry_run: bool = False,
    excludes: Sequence[str] = (),
    config: Optional[str] = None,
) -> List[str]:
    """Copy a remote folder into the local directory of a target.

    Args:
        target: Key of ``TARGETS`` (``data``, ``output`` or ``tracking``).
        remote: rclone path of the Drive root folder.
        dry_run: Report only.
        excludes: rclone exclude patterns.
        config: Optional rclone configuration file.

    Returns:
        The rclone command line that was run.
    """
    destination = local_directory(target)
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
    source = join_remote(remote, TARGETS[target].remote_dir)
    return run_rclone(source, str(destination), dry_run=dry_run, excludes=excludes, config=config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True, help="Local directory to fill.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="rclone path of the Drive root folder.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded.")
    parser.add_argument("--exclude", action="append", default=[], help="rclone exclude pattern (repeatable).")
    parser.add_argument("--config", default=None, help="rclone configuration file.")
    args = parser.parse_args(argv)
    download(args.target, args.remote, dry_run=args.dry_run, excludes=args.exclude, config=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
