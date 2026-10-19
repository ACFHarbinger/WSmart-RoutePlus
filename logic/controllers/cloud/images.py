"""Push the release image to Artifact Registry, ECR or ACR, and pin jobs to its digest.

:class:`DockerRegistry` tags and pushes a local image with the Docker CLI and
returns the pushed digest, so jobs can run ``<repository>@sha256:...`` instead
of a mutable tag. The provider factories differ only in how they log Docker in:

- :func:`artifact_registry`: ``gcloud auth configure-docker <host>``;
- :func:`ecr`: ``aws ecr get-login-password`` piped to ``docker login``;
- :func:`acr`: ``az acr login --name <registry>``.

Commands go through an injected runner (``subprocess.run`` by default), so the
tests check the exact commands without Docker or a registry.

Attributes:
    DockerRegistry: :class:`~logic.controllers.cloud.interfaces.ImageRegistry` over the Docker CLI.
    artifact_registry: Google Artifact Registry.
    ecr: Amazon ECR.
    acr: Azure Container Registry.
    pinned: ``repository@digest`` for a pushed image.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Callable, List, Optional

from logic.controllers.cloud.interfaces import ImageRegistry

Runner = Callable[..., Any]


class RegistryError(RuntimeError):
    """A Docker or registry command failed."""


class DockerRegistry(ImageRegistry):
    """An image repository reached through the Docker CLI.

    Args:
        repository: Full repository, for example ``europe-west1-docker.pkg.dev/proj/wsr/wsmart-route``.
        login: Commands that log Docker in to the registry, run before every push.
        run: Command runner with ``subprocess.run``'s interface.
    """

    def __init__(self, repository: str, login: Optional[List[List[str]]] = None, run: Runner = subprocess.run) -> None:
        self.repository = repository.rstrip("/")
        self.login = login or []
        self.run = run

    def _call(self, argv: List[str], input_from: Optional[List[str]] = None) -> str:
        stdin = None
        if input_from is not None:
            stdin = self._call(input_from)
        result = self.run(argv, input=stdin, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RegistryError(f"{' '.join(argv[:3])} failed: {(result.stderr or '').strip()[:500]}")
        return result.stdout

    def push(self, local_ref: str, tag: str) -> str:
        """Tag ``local_ref`` as ``<repository>:<tag>``, push it and return its ``sha256:`` digest."""
        for step in self.login:
            if step and step[0] == "|":  # ["|", producer..., "--", consumer...]
                split = step.index("--")
                self._call(step[split + 1 :], input_from=step[1:split])
            else:
                self._call(step)
        remote = f"{self.repository}:{tag}"
        self._call(["docker", "tag", local_ref, remote])
        self._call(["docker", "push", remote])
        digests = json.loads(
            self._call(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", remote]) or "[]"
        )
        for entry in digests:
            name, _, digest = entry.partition("@")
            if name == self.repository and digest.startswith("sha256:"):
                return digest
        raise RegistryError(f"no digest for {remote} after the push")

    def resolve(self, tag: str) -> Optional[str]:
        """The digest behind ``<repository>:<tag>`` in the registry, or None."""
        try:
            out = self._call(["docker", "manifest", "inspect", "--verbose", f"{self.repository}:{tag}"])
        except RegistryError:
            return None
        data = json.loads(out)
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            digest = (entry.get("Descriptor") or {}).get("digest")
            if digest:
                return digest
        return None


def artifact_registry(
    region: str, project: str, repository: str, image: str = "wsmart-route", run: Runner = subprocess.run
) -> DockerRegistry:
    """``<region>-docker.pkg.dev/<project>/<repository>/<image>``."""
    host = f"{region}-docker.pkg.dev"
    return DockerRegistry(
        f"{host}/{project}/{repository}/{image}",
        login=[["gcloud", "auth", "configure-docker", host, "--quiet"]],
        run=run,
    )


def ecr(account: str, region: str, image: str = "wsmart-route", run: Runner = subprocess.run) -> DockerRegistry:
    """``<account>.dkr.ecr.<region>.amazonaws.com/<image>``."""
    host = f"{account}.dkr.ecr.{region}.amazonaws.com"
    login = ["|", "aws", "ecr", "get-login-password", "--region", region, "--",
             "docker", "login", "--username", "AWS", "--password-stdin", host]  # fmt: skip
    return DockerRegistry(f"{host}/{image}", login=[login], run=run)


def acr(registry: str, image: str = "wsmart-route", run: Runner = subprocess.run) -> DockerRegistry:
    """``<registry>.azurecr.io/<image>``."""
    return DockerRegistry(f"{registry}.azurecr.io/{image}", login=[["az", "acr", "login", "--name", registry]], run=run)


def pinned(registry: DockerRegistry, digest: str) -> str:
    """The immutable reference jobs should use: ``<repository>@<digest>``."""
    if not digest.startswith("sha256:"):
        raise ValueError(f"not a digest: {digest!r}")
    return f"{registry.repository}@{digest}"
