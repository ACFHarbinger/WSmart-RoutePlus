"""Synchronise the local data, output and tracking directories with Google Drive folders.

The sync is a thin, copy-only wrapper around ``rclone``. It needs an rclone remote
for Google Drive, created once with ``rclone config`` (the OAuth token stays in the
rclone configuration, never in this repository). Nothing is deleted on either side.

Subpackages:
    uploader: local directories to Drive (``python -m logic.cloud.uploader.sync``).
    downloader: Drive folders to local directories (``python -m logic.cloud.downloader.sync``).
"""
