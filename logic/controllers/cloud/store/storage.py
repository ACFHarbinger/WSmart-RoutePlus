"""Hashing and tree transfer on top of any :class:`~logic.controllers.cloud.interfaces.ObjectStore`.

Attributes:
    sha256_file: Hex SHA-256 of a file.
    upload_tree: Upload every file under a directory and return their records.
    download_results: Download a results index into a local root, checking hashes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

from logic.controllers.cloud.model import ResultFile

if TYPE_CHECKING:
    from logic.controllers.cloud.interfaces import ObjectStore


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def upload_tree(store: "ObjectStore", root: Path, prefix: str) -> List[ResultFile]:
    """Upload every file under ``root`` to ``<prefix>/<relative path>``.

    Returns:
        One record per file, with ``relpath`` relative to ``root``.
    """
    records = []
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        relpath = path.relative_to(root).as_posix()
        uploaded = store.put_file(path, f"{prefix}/{relpath}")
        records.append(ResultFile(relpath=relpath, key=uploaded.key, sha256=uploaded.sha256, size=uploaded.size))
    return records


def download_results(store: "ObjectStore", files: Sequence[ResultFile], dest: Path) -> List[Path]:
    """Download result files under ``dest`` at their ``relpath``, checking each hash.

    Raises:
        ValueError: A downloaded file's hash differs from the index.
    """
    written = []
    for record in files:
        target = Path(dest) / record.relpath
        store.get_file(record.key, target)
        if sha256_file(target) != record.sha256:
            raise ValueError(f"hash mismatch for {record.relpath} ({record.key})")
        written.append(target)
    return written
