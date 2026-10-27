"""Build the Linux VM bootstrap and its ephemeral SSH key.

The generated script installs Docker, exchanges managed-identity credentials
for an ACR refresh token, runs the task, records its exit status, and shuts down.
The provisioning lifecycle remains in vm.py and provisioning.py.
"""

import base64
import json
import shlex
from typing import Optional
from urllib.parse import quote

from logic.controllers.cloud.interfaces import LaunchSpec


class AzureVmBootstrapMixin:
    """Bootstrap methods composed into AzureVmBackend."""

    ssh_public_key: Optional[str]
    vault_name: str
    identity_id: str

    def _get_or_generate_ssh_key(self) -> str:
        """Return configured SSH public key or generate a throwaway key."""
        if self.ssh_public_key:
            return self.ssh_public_key.strip()

        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        return (
            private_key.public_key()
            .public_bytes(
                encoding=serialization.Encoding.OpenSSH,
                format=serialization.PublicFormat.OpenSSH,
            )
            .decode("utf-8")
        )

    def _build_custom_data(self, launch: LaunchSpec) -> str:
        """Construct UserData bootstrap script executing the task in Docker and shutting down."""
        env = dict(launch.env)
        env.setdefault("WSR_JOB_ID", launch.task.job_id)
        env.setdefault("WSR_TASK_ID", launch.task.task_id)

        # Inject secrets via WSR_SECRET_REFS (common secret path)
        if launch.secret_env:
            secret_refs = {
                var: f"akv://{self.vault_name}/{sec_name}" for var, sec_name in sorted(launch.secret_env.items())
            }
            env["WSR_SECRET_REFS"] = json.dumps(secret_refs)

        # Format docker run -e arguments
        env_flags = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in sorted(env.items()))
        image = launch.image or "wsmart-route:latest"
        cmd = " ".join(shlex.quote(arg) for arg in launch.argv)

        registry = image.split("/", 1)[0] if "/" in image else ""
        # The stock Ubuntu image has no Azure CLI. Exchange the managed identity
        # token for an ACR refresh token; ARM tokens are not Docker passwords.
        acr_login = ""
        if registry.endswith(".azurecr.io"):
            acr_login = f"""ACR_TOKEN=$(curl -fsS -m 30 -X POST \\
    --data-urlencode grant_type=access_token \\
    --data-urlencode service={shlex.quote(registry)} \\
    --data-urlencode "access_token=$ACCESS_TOKEN" \\
    https://{registry}/oauth2/exchange | python3 -c 'import json,sys; print(json.load(sys.stdin)["refresh_token"])')
printf '%s' "$ACR_TOKEN" | docker login {shlex.quote(registry)} \\
    -u 00000000-0000-0000-0000-000000000000 --password-stdin
"""

        identity_query = quote(self.identity_id, safe="")
        script = f"""#!/bin/bash
set -euo pipefail
trap 'shutdown -h now' EXIT

# Dependencies for the selected Ubuntu image. Fail before launching if setup fails.
if ! command -v docker >/dev/null 2>&1; then
    apt-get update
    apt-get install -y docker.io curl python3
fi
systemctl start docker

ACCESS_TOKEN=$(curl -fsS -m 30 -H "Metadata:true" \\
    'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/&mi_res_id={identity_query}' \\
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
{acr_login}
EXIT_CODE=0
docker run --rm {env_flags} {shlex.quote(image)} {cmd} || EXIT_CODE=$?

# Refresh after execution: the initial token may have expired during the run.
ACCESS_TOKEN=$(curl -fsS -m 30 -H "Metadata:true" \\
    'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/&mi_res_id={identity_query}' \\
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
VM_ID=$(curl -fsS -m 30 -H "Metadata:true" \\
    'http://169.254.169.254/metadata/instance/compute/resourceId?api-version=2021-02-01&format=text')
# Merge the actual exit code without replacing scheduler ownership tags.
TAG_BODY=$(printf '{{"operation":"Merge","properties":{{"tags":{{"wsr-exit-code":"%s"}}}}}}' "$EXIT_CODE")
curl -fsS -m 30 -X PATCH \\
    -H "Authorization: Bearer $ACCESS_TOKEN" \\
    -H "Content-Type: application/json" \\
    -d "$TAG_BODY" \\
    "https://management.azure.com$VM_ID/providers/Microsoft.Resources/tags/default?api-version=2021-04-01"
"""
        return base64.b64encode(script.encode("utf-8")).decode("utf-8")
