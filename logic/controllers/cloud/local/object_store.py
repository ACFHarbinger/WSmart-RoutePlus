"""Filesystem object store and local secret stores.

Attributes:
    FilesystemObjectStore: :class:`~logic.controllers.cloud.interfaces.ObjectStore` on a local directory.
    EnvSecretStore: Secrets read from environment variables (read-only).
    MemorySecretStore: Secrets held in memory, for tests and short-lived BYO licenses.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Union

from logic.controllers.cloud.interfaces import ObjectStore, SecretStore
from logic.controllers.cloud.model import ResultFile
from logic.controllers.cloud.storage import sha256_file


class FilesystemObjectStore(ObjectStore):
    """Objects as files under a root directory; keys are relative ``/``-paths."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        parts = PurePosixPath(key).parts
        if not parts or key.startswith("/") or ".." in parts:
            raise ValueError(f"invalid object key {key!r}")
        return self.root.joinpath(*parts)

    def put_file(self, local: Path, key: str) -> ResultFile:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        shutil.copyfile(local, tmp)
        os.replace(tmp, target)
        return ResultFile(relpath=key, key=key, sha256=sha256_file(target), size=target.stat().st_size)

    def get_file(self, key: str, local: Path) -> None:
        source = self._path(key)
        if not source.is_file():
            raise KeyError(key)
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str) -> List[str]:
        keys = []
        for path in self.root.rglob("*"):
            if path.is_file() and not path.name.endswith(".part"):
                key = path.relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class EnvSecretStore(SecretStore):
    """Secrets from environment variables; ``prefix`` is prepended to the upper-cased name."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def _var(self, name: str) -> str:
        return self.prefix + name.upper().replace("-", "_").replace("/", "_")

    def get(self, name: str) -> Optional[str]:
        return os.environ.get(self._var(name))

    def put(self, name: str, value: str) -> None:
        raise PermissionError("the environment secret store is read-only")

    def delete(self, name: str) -> None:
        raise PermissionError("the environment secret store is read-only")


class MemorySecretStore(SecretStore):
    """Secrets in memory."""

    def __init__(self, values: Optional[Dict[str, str]] = None) -> None:
        self._values = dict(values or {})

    def get(self, name: str) -> Optional[str]:
        return self._values.get(name)

    def put(self, name: str, value: str) -> None:
        self._values[name] = value

    def delete(self, name: str) -> None:
        self._values.pop(name, None)
