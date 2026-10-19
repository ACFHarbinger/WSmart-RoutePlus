"""AWS Secrets Manager secret store implementation.

Attributes:
    AwsSecretsManagerStore: :class:`~logic.controllers.cloud.interfaces.SecretStore` backed by AWS Secrets Manager.
"""

from __future__ import annotations

from typing import Any, Optional

from logic.controllers.cloud.interfaces import SecretStore
from logic.controllers.cloud.providers.aws._sdk import make_boto3_client


class AwsSecretsManagerStore(SecretStore):
    """AWS Secrets Manager secret store.

    Args:
        client: Optional pre-configured boto3 SecretsManager client (injected for testing).
        prefix: Optional prefix for secret names in AWS Secrets Manager.
        profile: AWS credentials profile name (used when client is not provided).
        region: AWS region name (used when client is not provided).
    """

    def __init__(
        self,
        client: Any = None,
        prefix: str = "",
        profile: str = "default",
        region: Optional[str] = None,
    ) -> None:
        self.prefix = prefix
        self.profile = profile
        self.region = region
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_boto3_client("secretsmanager", profile=self.profile, region=self.region)
        return self._client

    def _secret_id(self, name: str) -> str:
        return f"{self.prefix}{name}"

    def get(self, name: str) -> Optional[str]:
        secret_id = self._secret_id(name)
        try:
            resp = self.client.get_secret_value(SecretId=secret_id)
            if "SecretString" in resp:
                return resp["SecretString"]
            if "SecretBinary" in resp:
                binary = resp["SecretBinary"]
                return binary.decode("utf-8") if isinstance(binary, bytes) else str(binary)
            return None
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if (
                code in ("ResourceNotFoundException", "NotFoundException")
                or isinstance(exc, KeyError)
                or "ResourceNotFoundException" in str(exc)
                or "not found" in str(exc).lower()
            ):
                return None
            raise

    def put(self, name: str, value: str) -> None:
        secret_id = self._secret_id(name)
        try:
            self.client.put_secret_value(SecretId=secret_id, SecretString=value)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if (
                code in ("ResourceNotFoundException", "NotFoundException")
                or isinstance(exc, KeyError)
                or "ResourceNotFoundException" in str(exc)
                or "not found" in str(exc).lower()
            ):
                # Secret does not exist yet: create it
                self.client.create_secret(Name=secret_id, SecretString=value)
            else:
                raise

    def delete(self, name: str) -> None:
        secret_id = self._secret_id(name)
        try:
            self.client.delete_secret(SecretId=secret_id, ForceDeleteWithoutRecovery=True)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if (
                code in ("ResourceNotFoundException", "NotFoundException")
                or isinstance(exc, KeyError)
                or "ResourceNotFoundException" in str(exc)
                or "not found" in str(exc).lower()
            ):
                return
            # Delete is best-effort and a no-op if absent
            pass


__all__ = ["AwsSecretsManagerStore"]
