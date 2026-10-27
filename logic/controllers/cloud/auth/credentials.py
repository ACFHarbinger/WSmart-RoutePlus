"""Provider credentials: local CLI logins and keychain service accounts.

Two sources are supported (owner ruling, 2026-10-19):

- **CLI logins:** gcloud Application Default Credentials, AWS shared config and
  credentials profiles (including SSO), and the Azure CLI's ``az login``.
  WSmart-Route stores nothing; :func:`detect_logins` reports which providers are usable.
- **Keychain:** a service-account key stored in the OS keychain by the Studio
  (CR-13) or ``python -m logic.controllers.cloud secrets set``. Service
  ``wsmart-route.cloud``, account ``{provider}/{profile}``. Other secrets, such as a
  Gurobi license, use account ``secret/{name}``.

Key formats in the keychain, by provider:

- ``gcp``: the service-account JSON key file's contents;
- ``aws``: ``{"aws_access_key_id": ..., "aws_secret_access_key": ...[, "aws_session_token": ...]}``;
- ``azure``: ``{"tenant_id": ..., "client_id": ..., "client_secret": ...}``.

A brokered, multi-user control plane is future work (roadmap §J.3).

Attributes:
    KEYCHAIN_SERVICE: The keychain service name shared with the Studio.
    LoginStatus: Whether a provider's CLI login is usable.
    detect_logins: CLI login status per provider.
    KeychainSecretStore: :class:`~logic.controllers.cloud.interfaces.SecretStore` on the OS keychain.
    ProviderCredentials: SDK credential objects for one profile.
"""

from __future__ import annotations

import configparser
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from logic.controllers.cloud.interfaces import SecretStore

KEYCHAIN_SERVICE = "wsmart-route.cloud"
PROVIDERS = ("gcp", "aws", "azure")


def credential_account(provider: str, profile: str) -> str:
    """Keychain account of a provider credential (the Studio's naming, CR-13)."""
    return f"{provider.strip().lower()}/{profile.strip()}"


def secret_account(name: str) -> str:
    """Keychain account of a named secret, such as a Gurobi license."""
    return f"secret/{name.strip()}"


@dataclass
class LoginStatus:
    """Whether a provider's CLI login is usable.

    Attributes:
        provider: ``gcp``, ``aws`` or ``azure``.
        usable: A login or credential source was found.
        source: Where it was found (a file or an environment variable), or None.
        profiles: Named profiles, where the provider has them (AWS).
    """

    provider: str
    usable: bool
    source: Optional[str] = None
    profiles: List[str] = field(default_factory=list)


def _aws_profiles(home: Path, env: Mapping[str, str]) -> List[str]:
    names = set()
    for path, prefix in (
        (Path(env.get("AWS_SHARED_CREDENTIALS_FILE") or home / ".aws" / "credentials"), ""),
        (Path(env.get("AWS_CONFIG_FILE") or home / ".aws" / "config"), "profile "),
    ):
        if path.is_file():
            parser = configparser.ConfigParser()
            parser.read(path)
            for section in parser.sections():
                names.add(section[len(prefix) :] if prefix and section.startswith(prefix) else section)
    return sorted(names)


def detect_logins(env: Optional[Mapping[str, str]] = None, home: Optional[Path] = None) -> List[LoginStatus]:
    """CLI login status of every provider; reads files and environment variables only."""
    env = os.environ if env is None else env
    home = Path.home() if home is None else Path(home)
    statuses = []

    gcp_file = env.get("GOOGLE_APPLICATION_CREDENTIALS")
    adc = Path(env.get("CLOUDSDK_CONFIG") or home / ".config" / "gcloud") / "application_default_credentials.json"
    if gcp_file and Path(gcp_file).is_file():
        statuses.append(LoginStatus("gcp", True, "GOOGLE_APPLICATION_CREDENTIALS"))
    elif adc.is_file():
        statuses.append(LoginStatus("gcp", True, str(adc)))
    else:
        statuses.append(LoginStatus("gcp", False))

    profiles = _aws_profiles(home, env)
    if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        statuses.append(LoginStatus("aws", True, "AWS_ACCESS_KEY_ID", profiles))
    else:
        statuses.append(LoginStatus("aws", bool(profiles), "~/.aws" if profiles else None, profiles))

    azure_dir = Path(env.get("AZURE_CONFIG_DIR") or home / ".azure")
    if env.get("AZURE_CLIENT_ID") and env.get("AZURE_TENANT_ID") and env.get("AZURE_CLIENT_SECRET"):
        statuses.append(LoginStatus("azure", True, "AZURE_CLIENT_ID"))
    elif (azure_dir / "azureProfile.json").is_file() and (azure_dir / "msal_token_cache.json").is_file():
        statuses.append(LoginStatus("azure", True, str(azure_dir)))
    else:
        statuses.append(LoginStatus("azure", False))
    return statuses


class KeychainSecretStore(SecretStore):
    """Secrets in the OS keychain (``keyring``), under service ``wsmart-route.cloud``.

    Names are keychain accounts: use :func:`credential_account` or :func:`secret_account`.

    Args:
        backend: An object with ``get_password``, ``set_password`` and ``delete_password``
            (default: the ``keyring`` module).
    """

    def __init__(self, backend: Any = None) -> None:
        if backend is None:
            import keyring  # pyrefly: ignore[missing-import]  # optional cloud SDK

            backend = keyring
        self.backend = backend

    def get(self, name: str) -> Optional[str]:
        return self.backend.get_password(KEYCHAIN_SERVICE, name)

    def put(self, name: str, value: str) -> None:
        self.backend.set_password(KEYCHAIN_SERVICE, name, value)

    def delete(self, name: str) -> None:
        try:
            self.backend.delete_password(KEYCHAIN_SERVICE, name)
        except Exception as exc:  # keyring.errors.PasswordDeleteError when absent
            if self.get(name) is not None:
                raise exc


class MissingCredentialError(LookupError):
    """A keychain profile has no stored credential."""


@dataclass
class ProviderCredentials:
    """SDK credential objects for one provider profile.

    Attributes:
        provider: ``gcp``, ``aws`` or ``azure``.
        profile: Profile name.
        source: ``cli`` (the provider's own login chain) or ``keychain``.
        keychain: Store holding keychain credentials.
    """

    provider: str
    profile: str = "default"
    source: str = "cli"
    keychain: Optional[SecretStore] = None

    def _key(self) -> Dict[str, Any]:
        store = self.keychain or KeychainSecretStore()
        raw = store.get(credential_account(self.provider, self.profile))
        if raw is None:
            raise MissingCredentialError(f"no {self.provider} credential stored for profile {self.profile!r}")
        return json.loads(raw)

    def aws_session_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``boto3.Session``."""
        if self.source == "cli":
            return {} if self.profile == "default" else {"profile_name": self.profile}
        key = self._key()
        kwargs = {"aws_access_key_id": key["aws_access_key_id"], "aws_secret_access_key": key["aws_secret_access_key"]}
        if key.get("aws_session_token"):
            kwargs["aws_session_token"] = key["aws_session_token"]
        return kwargs

    def gcp_credentials(self) -> Any:
        """A ``google.auth`` credentials object, or None for Application Default Credentials."""
        if self.source == "cli":
            return None
        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_info(self._key())

    def azure_credential(self) -> Any:
        """An ``azure.identity`` credential."""
        if self.source == "cli":
            from azure.identity import DefaultAzureCredential  # pyrefly: ignore[missing-import]  # optional cloud SDK

            return DefaultAzureCredential()
        from azure.identity import ClientSecretCredential  # pyrefly: ignore[missing-import]  # optional cloud SDK

        key = self._key()
        return ClientSecretCredential(key["tenant_id"], key["client_id"], key["client_secret"])
