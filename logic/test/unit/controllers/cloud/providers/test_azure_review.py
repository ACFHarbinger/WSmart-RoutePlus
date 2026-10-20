"""Exercise generated bootstrap with local commands, without cloud access."""

import base64
import json
import os
import subprocess
from pathlib import Path

import pytest
from logic.controllers.cloud.interfaces import LaunchSpec
from logic.controllers.cloud.model import CellSpec, Resources, TaskRecord
from logic.controllers.cloud.providers.azure import AzureVmBackend


@pytest.mark.parametrize("exit_code", [0, 7])
def test_bootstrap_exchanges_acr_token_and_records_actual_exit(tmp_path: Path, exit_code: int) -> None:
    launch = LaunchSpec(
        task=TaskRecord(task_id="task", job_id="job", cell=CellSpec(cell_id="cell", argv=["run"])),
        argv=["run", "argument with spaces"],
        env={},
        secret_env={},
        image="example.azurecr.io/runner:latest",
        resources=Resources(),
    )
    script = base64.b64decode(AzureVmBackend()._build_custom_data(launch)).decode()
    commands = {
        "systemctl": "exit 0",
        "shutdown": 'touch "$TEST_ROOT/shutdown"',
        "docker": """if [ "$1" = login ]; then
    read -r TOKEN || true
    test "$TOKEN" = acr-refresh || exit 90
    touch "$TEST_ROOT/login"
else
    test -f "$TEST_ROOT/login" || exit 91
    exit "$TEST_EXIT"
fi""",
        "curl": """case "$*" in
  *oauth2/token*) echo '{"access_token":"arm-token"}' ;;
  *oauth2/exchange*) echo '{"refresh_token":"acr-refresh"}' ;;
  *resourceId*) echo /subscriptions/test/resourceGroups/test/providers/Microsoft.Compute/virtualMachines/test ;;
  *tags/default*)
    while [ "$#" -gt 0 ]; do
      if [ "$1" = -d ]; then shift; printf '%s' "$1" > "$TEST_ROOT/tag.json"; fi
      shift
    done ;;
  *) exit 92 ;;
esac""",
    }
    for name, body in commands.items():
        command = tmp_path / name
        command.write_text("#!/bin/bash\n" + body + "\n")
        command.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", TEST_ROOT=str(tmp_path), TEST_EXIT=str(exit_code))
    subprocess.run(["bash"], input=script, text=True, env=env, check=True, capture_output=True)
    assert json.loads((tmp_path / "tag.json").read_text()) == {
        "operation": "Merge",
        "properties": {"tags": {"wsr-exit-code": str(exit_code)}},
    }
    assert (tmp_path / "shutdown").exists()


def test_ssh_key_is_parseable() -> None:
    from cryptography.hazmat.primitives.serialization import load_ssh_public_key

    load_ssh_public_key(AzureVmBackend()._get_or_generate_ssh_key().encode())
