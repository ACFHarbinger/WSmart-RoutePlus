"""Cloud profiles: where jobs run and how to authenticate, without secrets in the file.

The file is ``$WSR_CLOUD_CONFIG``, or ``~/.config/wsmart-route/cloud.yaml``:

.. code-block:: yaml

    profiles:
      gcp-batch:
        provider: gcp             # gcp | aws | azure
        compute: batch            # default compute kind: batch | serverless | vm
        region: europe-west1
        credentials: cli          # cli (gcloud/aws/az login) | keychain
        keychain_profile: default # keychain account {provider}/{keychain_profile}
        control_plane: firestore://my-project
        store: gs://my-bucket/wsmart-route
        image: europe-west1-docker.pkg.dev/my-project/wsr/wsmart-route:1.1.0
        gurobi_secret: gurobi-wls # optional: secret name of the Gurobi license

Attributes:
    Profile: One profile.
    load_profiles: Read and validate the profiles file.
    default_config_path: Where the file is looked for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

from logic.controllers.cloud.credentials import PROVIDERS, ProviderCredentials
from logic.controllers.cloud.interfaces import SecretStore
from logic.controllers.cloud.model import ComputeKind
from logic.controllers.cloud.service import RemoteConfig

_CONTROL_PLANE_SCHEMES = {"sqlite", "firestore", "supabase"}
_STORE_SCHEMES = {"file", "gs", "s3", "az"}


@dataclass
class Profile:
    """One cloud profile (see the module docstring for the fields)."""

    name: str
    provider: str
    control_plane: str
    store: str
    compute: str = "batch"
    region: Optional[str] = None
    credentials: str = "cli"
    keychain_profile: str = "default"
    image: Optional[str] = None
    gurobi_secret: Optional[str] = None

    def validate(self) -> None:
        """Raise ``ValueError`` for an unusable profile."""
        if self.provider not in PROVIDERS:
            raise ValueError(f"profile {self.name!r}: provider must be one of {', '.join(PROVIDERS)}")
        ComputeKind(self.compute)
        if self.compute == ComputeKind.PROCESS.value:
            raise ValueError(f"profile {self.name!r}: compute 'process' is the local backend, not a cloud profile")
        if self.credentials not in ("cli", "keychain"):
            raise ValueError(f"profile {self.name!r}: credentials must be 'cli' or 'keychain'")
        if urlparse(self.control_plane).scheme not in _CONTROL_PLANE_SCHEMES:
            raise ValueError(f"profile {self.name!r}: unsupported control_plane {self.control_plane!r}")
        if urlparse(self.store).scheme not in _STORE_SCHEMES:
            raise ValueError(f"profile {self.name!r}: unsupported store {self.store!r}")

    def provider_credentials(self, keychain: Optional[SecretStore] = None) -> ProviderCredentials:
        """The profile's SDK credentials."""
        return ProviderCredentials(self.provider, self.keychain_profile, self.credentials, keychain)

    def remote_config(self, project_root: Path) -> RemoteConfig:
        """The service's remote settings for this profile."""
        return RemoteConfig(control_plane_uri=self.control_plane, store_uri=self.store, project_root=project_root)


def default_config_path() -> Path:
    """``$WSR_CLOUD_CONFIG``, or ``~/.config/wsmart-route/cloud.yaml``."""
    return Path(os.environ.get("WSR_CLOUD_CONFIG") or Path.home() / ".config" / "wsmart-route" / "cloud.yaml")


def load_profiles(path: Optional[Path] = None) -> Dict[str, Profile]:
    """Read and validate every profile; an absent file has none.

    Raises:
        ValueError: Unknown keys (secrets have no field here), missing keys, or an invalid profile.
    """
    path = default_config_path() if path is None else Path(path)
    if not path.is_file():
        return {}
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    known = {f.name for f in fields(Profile)} - {"name"}
    profiles = {}
    for name, body in (data.get("profiles") or {}).items():
        unknown = set(body) - known
        if unknown:
            raise ValueError(f"profile {name!r}: unknown key(s) {', '.join(sorted(unknown))}")
        try:
            profile = Profile(name=name, **body)
        except TypeError as exc:
            raise ValueError(f"profile {name!r}: {exc}") from None
        profile.validate()
        profiles[name] = profile
    return profiles
