"""Secret references the task runtime resolves through the machine's own identity.

Where a compute service cannot inject a secret natively (Azure Batch, the VM
backends), the backend sets ``WSR_SECRET_REFS`` to a JSON object mapping an
environment variable to a reference, and the runtime resolves each one before
starting the simulator:

=================================  ==============================================
``akv://<vault>/<name>``            Azure Key Vault, managed identity
``awssm://<region>/<name>``         AWS Secrets Manager, instance or task role
``gcpsm://<project>/<name>``        Google Secret Manager (latest version), service account
``env://<VAR>``                     an environment variable of the runtime (tests, local use)
=================================  ==============================================

Values are never written to a job record, a task environment setting or a
launch template; :func:`redact` removes them from text the runtime stores.

Attributes:
    RESOLVERS: Scheme -> resolver; providers' resolvers import their SDK lazily.
    parse_refs: Read ``WSR_SECRET_REFS``.
    resolve_refs: Resolve every reference.
    redact: Replace secret values in text.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, Iterable, Optional
from urllib.parse import ParseResult, urlparse

Resolver = Callable[[ParseResult], str]


def _akv(ref: ParseResult) -> str:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    client = SecretClient(vault_url=f"https://{ref.netloc}.vault.azure.net", credential=DefaultAzureCredential())
    return client.get_secret(ref.path.strip("/")).value


def _awssm(ref: ParseResult) -> str:
    import boto3

    client = boto3.client("secretsmanager", region_name=ref.netloc)
    return client.get_secret_value(SecretId=ref.path.strip("/"))["SecretString"]


def _gcpsm(ref: ParseResult) -> str:
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{ref.netloc}/secrets/{ref.path.strip('/')}/versions/latest"
    return client.access_secret_version(name=name).payload.data.decode()


def _env(ref: ParseResult) -> str:
    value = os.environ.get(ref.netloc)
    if value is None:
        raise KeyError(f"environment variable {ref.netloc} is not set")
    return value


RESOLVERS: Dict[str, Resolver] = {"akv": _akv, "awssm": _awssm, "gcpsm": _gcpsm, "env": _env}


def parse_refs(raw: Optional[str]) -> Dict[str, str]:
    """``WSR_SECRET_REFS`` as a dict; empty when unset.

    Raises:
        ValueError: Not a JSON object of strings, or an unknown scheme.
    """
    if not raw:
        return {}
    refs = json.loads(raw)
    if not isinstance(refs, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in refs.items()):
        raise ValueError("WSR_SECRET_REFS must be a JSON object of strings")
    for var, ref in refs.items():
        if urlparse(ref).scheme not in RESOLVERS:
            raise ValueError(f"{var}: unsupported secret reference {ref!r}")
    return refs


def resolve_refs(refs: Dict[str, str], resolvers: Optional[Dict[str, Resolver]] = None) -> Dict[str, str]:
    """Resolve every reference; returns ``{env var: value}``.

    Raises:
        LookupError: A reference could not be resolved (the message names the variable, never a value).
    """
    resolvers = RESOLVERS if resolvers is None else resolvers
    values = {}
    for var, ref in refs.items():
        parsed = urlparse(ref)
        try:
            values[var] = resolvers[parsed.scheme](parsed)
        except Exception as exc:
            raise LookupError(
                f"could not resolve the secret for {var} ({parsed.scheme}): {type(exc).__name__}"
            ) from None
    return values


def redact(text: str, values: Iterable[str]) -> str:
    """``text`` with every secret value (and each of its lines) replaced by ``***``."""
    for value in sorted((v for v in values if v), key=len, reverse=True):
        text = text.replace(value, "***")
        for line in value.splitlines():
            if len(line) >= 6:
                text = text.replace(line, "***")
    return text
