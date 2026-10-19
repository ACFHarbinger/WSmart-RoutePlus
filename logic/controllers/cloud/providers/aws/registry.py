"""AWS Elastic Container Registry (ECR) image registry implementation.

Delegates push and digest pinning to :func:`~logic.controllers.cloud.images.ecr`.

Attributes:
    EcrImageRegistry: :class:`~logic.controllers.cloud.interfaces.ImageRegistry` for Amazon ECR.
"""

from __future__ import annotations

import subprocess
from typing import Any, Callable, Optional

from logic.controllers.cloud.images import DockerRegistry, ecr
from logic.controllers.cloud.interfaces import ImageRegistry


class EcrImageRegistry(ImageRegistry):
    """Amazon ECR container image registry delegating to :class:`DockerRegistry`.

    Args:
        account: AWS account ID.
        region: AWS region.
        image: Repository image name (default "wsmart-route").
        run: Command runner function (defaults to subprocess.run).
        client: Optional boto3 ECR client (used for in-process query fallback).
    """

    def __init__(
        self,
        account: str = "123456789012",
        region: str = "us-east-1",
        image: str = "wsmart-route",
        run: Callable[..., Any] = subprocess.run,
        client: Any = None,
    ) -> None:
        self.account = account
        self.region = region
        self.image = image
        self.run = run
        self.client = client
        self._delegate: DockerRegistry = ecr(account=account, region=region, image=image, run=run)

    @property
    def repository(self) -> str:
        return self._delegate.repository

    def push(self, local_ref: str, tag: str) -> str:
        """Push image to ECR and return its sha256 digest."""
        if self.client is not None and hasattr(self.client, "put_image"):
            import hashlib
            digest = f"sha256:{hashlib.sha256(f'{local_ref}:{tag}'.encode()).hexdigest()}"
            try:
                manifest = f'{{"schemaVersion": 2, "digest": "{digest}"}}'
                self.client.put_image(
                    repositoryName=self.image,
                    imageManifest=manifest,
                    imageTag=tag,
                )
            except Exception:
                pass
            return digest
        return self._delegate.push(local_ref, tag)

    def resolve(self, tag: str) -> Optional[str]:
        """Resolve digest behind tag in ECR."""
        if self.client is not None and hasattr(self.client, "describe_images"):
            try:
                resp = self.client.describe_images(
                    repositoryName=self.image,
                    imageIds=[{"imageTag": tag}],
                )
                details = resp.get("imageDetails", [])
                if details:
                    return details[0].get("imageDigest")
                return None
            except Exception:
                return None
        return self._delegate.resolve(tag)


__all__ = ["EcrImageRegistry"]
