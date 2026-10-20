"""Offline regressions for cleanup queue persistence and account boundaries."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from logic.controllers.cloud.providers.azure import AzureVmBackend, ContainerAppsJobsBackend


def test_queue_writers_do_not_lose_records(tmp_path: Path) -> None:
    backend = AzureVmBackend(orphan_file=tmp_path / "orphans.json")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda index: backend._record_orphan("vm", "rg", f"vm-{index}"), range(20)))
    assert len(backend._load_orphans()) == 20


def test_shared_queue_never_deletes_another_subscription(tmp_path: Path) -> None:
    path = tmp_path / "orphans.json"
    AzureVmBackend(subscription_id="subscription-a", orphan_file=path)._record_orphan("vm", "rg", "same-name")
    deleted = []
    other = AzureVmBackend(
        subscription_id="subscription-b",
        orphan_file=path,
        client=SimpleNamespace(begin_delete=lambda **kwargs: deleted.append(kwargs)),
    )
    assert other.drain_orphans() == []
    assert deleted == []
    assert len(other._load_orphans()) == 1


def test_corrupt_queue_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "orphans.json"
    path.write_text("[broken")
    backend = AzureVmBackend(orphan_file=path)
    with pytest.raises(ValueError):
        backend._record_orphan("vm", "rg", "new-vm")
    assert path.read_text() == "[broken"


def test_write_failure_is_visible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = AzureVmBackend(orphan_file=tmp_path / "orphans.json")

    def fail_replace(*args: object) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr("logic.controllers.cloud.providers.azure.orphan_queue.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk unavailable"):
        backend._record_orphan("vm", "rg", "new-vm")


def test_error_message_is_not_proof_of_resource_absence() -> None:
    assert not AzureVmBackend._is_not_found(RuntimeError("permission not found for request 40403"))
    assert AzureVmBackend._is_not_found(type("Missing", (Exception,), {"status_code": 404})())


def test_log_query_uses_required_timespan_and_surfaces_errors() -> None:
    def query_workspace(workspace_id: str, query: str, *, timespan: object) -> None:
        assert timespan is None
        raise RuntimeError("authorization denied")

    backend = ContainerAppsJobsBackend(
        workspace_id="workspace", logs_client=SimpleNamespace(query_workspace=query_workspace)
    )
    with pytest.raises(RuntimeError, match="authorization denied"):
        backend.logs("azure-aca:eastus:rg:job:exec")


def test_monitor_sdk_signature_is_compatible_offline() -> None:
    import inspect

    sdk = pytest.importorskip("azure.monitor.query")
    inspect.signature(sdk.LogsQueryClient.query_workspace).bind(object(), "workspace", "query", timespan=None)
