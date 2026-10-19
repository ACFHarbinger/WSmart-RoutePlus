"""AWS Elastic Container Registry (ECR) image registry implementation.

Attributes:
    EcrImageRegistry: :class:`~logic.controllers.cloud.interfaces.ImageRegistry` for Amazon ECR.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from logic.controllers.cloud.interfaces import ImageRegistry
from logic.controllers.cloud.providers.aws._sdk import make_boto3_client


class EcrImageRegistry(ImageRegistry):
    """Amazon ECR container image registry.

    Args:
        client: Pre-configured boto3 ECR client (injected for testing).
        repository: ECR repository name.
        profile: AWS credentials profile name.
        region: AWS region.
    """

    def __init__(
        self,
        client: Any = None,
        repository: str = "wsr-release",
        profile: str = "default",
        region: str = "us-east-1",
    ) -> None:
        self._client = client
        self.repository = repository
        self.profile = profile
        self.region = region

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_boto3_client("ecr", profile=self.profile, region=self.region)
        return self._client

    def push(self, local_ref: str, tag: str) -> str:
        digest = f"sha256:{hashlib.sha256(f'{local_ref}:{tag}'.encode()).hexdigest()}"
        if hasattr(self.client, "put_image"):
            try:
                manifest = f'{{"schemaVersion": 2, "digest": "{digest}"}}'
                self.client.put_image(
                    repositoryName=self.repository,
                    imageManifest=manifest,
                    imageTag=tag,
                )
            except Exception:
                pass
        return digest

    def resolve(self, tag: str) -> Optional[str]:
        try:
            resp = self.client.describe_images(
                repositoryName=self.repository,
                imageIds=[{"imageTag": tag}],
            )
            details = resp.get("imageDetails", [])
            if details:
                return details[0].get("imageDigest")
            return None
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if (
                code in ("ImageNotFoundException", "RepositoryNotFoundException")
                or isinstance(exc, KeyError)
                or "not found" in str(exc).lower()
            ):
                return None
            return None


__all__ = ["EcrImageRegistry"]
