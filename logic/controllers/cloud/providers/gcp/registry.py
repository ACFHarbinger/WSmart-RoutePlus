"""Google Artifact Registry image registry (CR-05).

Pushing goes through Docker (``gcloud auth configure-docker <region>-docker.pkg.dev`` once,
see :func:`logic.controllers.cloud.images.artifact_registry`). Resolving a tag uses the
Artifact Registry API when a client is given, otherwise Docker.

Attributes:
    ArtifactRegistry: :class:`~logic.controllers.cloud.interfaces.ImageRegistry` for Artifact Registry.
"""

from __future__ import annotations

import subprocess
from typing import Any, Optional

from logic.controllers.cloud.images import artifact_registry
from logic.controllers.cloud.interfaces import ImageRegistry
from logic.controllers.cloud.providers.gcp._sdk import field


class ArtifactRegistry(ImageRegistry):
    """``<region>-docker.pkg.dev/<project>/<repository>/<image>``.

    Args:
        project: Project id.
        region: Repository location (for example ``europe-west1``).
        repository: Artifact Registry repository.
        image: Image name.
        client: An ``ArtifactRegistryClient`` for :meth:`resolve` (optional; tests inject a fake).
        run: Subprocess runner for Docker.
    """

    def __init__(
        self,
        project: str,
        region: str,
        repository: str = "wsr",
        image: str = "wsmart-route",
        client: Any = None,
        run: Any = subprocess.run,
    ) -> None:
        self.project, self.region, self.repository_id, self.image = project, region, repository, image
        self.client = client
        self._docker = artifact_registry(project=project, region=region, repository=repository, image=image, run=run)

    @property
    def repository(self) -> str:
        return self._docker.repository

    def push(self, local_ref: str, tag: str) -> str:
        return self._docker.push(local_ref, tag)

    def resolve(self, tag: str) -> Optional[str]:
        if self.client is None:
            return self._docker.resolve(tag)
        name = (
            f"projects/{self.project}/locations/{self.region}/repositories/{self.repository_id}"
            f"/packages/{self.image}/tags/{tag}"
        )
        try:
            version = field(self.client.get_tag(name=name), "version", "")
        except Exception:
            return None
        digest = str(version).rsplit("/", 1)[-1]
        return digest if digest.startswith("sha256:") else None


__all__ = ["ArtifactRegistry"]
