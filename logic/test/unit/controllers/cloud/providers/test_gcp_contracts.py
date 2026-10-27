"""GCP provider (CR-05): the backend, object-store and secret-store contracts against fakes,
plus request shapes, preemption, limits, self-deleting VMs, pricing and the registry."""

from __future__ import annotations

from urllib.parse import urlparse

import pytest
from logic.controllers.cloud.interfaces import AttemptState, CapacityError
from logic.controllers.cloud.model import ComputeKind, Provider, Resources
from logic.controllers.cloud.providers import load_backend
from logic.controllers.cloud.providers.gcp import (
    ArtifactRegistry,
    CloudRunJobsBackend,
    ComputeEngineBackend,
    GcpBatchBackend,
    GcpSecretManagerStore,
    GcsObjectStore,
    price_per_hour,
    store_from_uri,
)
from logic.controllers.cloud.testing import (
    backend_contract,
    contract_launch,
    object_store_contract,
    secret_store_contract,
)
from logic.test.unit.controllers.cloud.providers.gcp_fakes import (
    FakeBatchClient,
    FakeGcsClient,
    FakeInstancesClient,
    FakeRunClients,
    FakeSecretManagerClient,
)


def _secrets():
    return GcpSecretManagerStore("proj", client=FakeSecretManagerClient())


def test_gcs_store_passes_its_project_to_the_client(monkeypatch):
    seen = {}

    def fake(service, credentials=None, project=None):
        seen["service"] = service
        seen["credentials"] = credentials
        seen["project"] = project
        return object()

    monkeypatch.setattr("logic.controllers.cloud.providers.gcp.storage.make_client", fake)
    store = GcsObjectStore("bucket", project="my-project")
    assert store.client is not None
    assert seen == {"service": "storage", "credentials": None, "project": "my-project"}


def test_storage_client_pins_project_and_quota_project(monkeypatch):
    pytest.importorskip("google.cloud.storage")
    captured = {}

    class _Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import google.cloud.storage as gcs

    monkeypatch.setattr(gcs, "Client", _Client)
    from logic.controllers.cloud.providers.gcp._sdk import client

    client("storage", project="my-project")
    assert captured["project"] == "my-project"
    assert captured["client_options"] == {"quota_project_id": "my-project"}
    assert "credentials" not in captured


def _batch(**kw):
    return GcpBatchBackend(
        project="proj", region="europe-west1", client=kw.pop("client", FakeBatchClient()), secrets=_secrets(), **kw
    )


def _run(clients=None, **kw):
    clients = clients or FakeRunClients()
    return CloudRunJobsBackend(
        project="proj",
        region="europe-west1",
        jobs_client=clients.jobs,
        executions_client=clients.executions,
        secrets=_secrets(),
        **kw,
    )


def _vm(gcs=None, instances=None):
    gcs = gcs or FakeGcsClient()
    store = GcsObjectStore("wsr-bucket", prefix="runs", client=gcs)
    return ComputeEngineBackend(
        project="proj", region="europe-west1", status_store=store, client=instances or FakeInstancesClient(gcs)
    )


def _wait(backend, handle, polls=10):
    for _ in range(polls):
        status = backend.poll(handle)
        if status.state not in (AttemptState.PENDING, AttemptState.RUNNING):
            return status
    raise AssertionError("attempt did not finish")


@pytest.mark.unit
def test_gcs_object_store_contract_and_uri():
    gcs = FakeGcsClient()
    object_store_contract(lambda: GcsObjectStore("wsr-bucket", prefix="team/runs", client=gcs))
    assert all(name.startswith("team/runs/") for name in gcs.buckets["wsr-bucket"])  # keys live under the prefix
    store = store_from_uri(urlparse("gs://b/some/prefix/"), client=gcs)
    assert (store.bucket_name, store.prefix) == ("b", "some/prefix/")


@pytest.mark.unit
def test_secret_manager_contract_and_names():
    secret_store_contract(_secrets)
    store = GcpSecretManagerStore("proj", client=FakeSecretManagerClient())
    assert store.version("gurobi-wls") == "projects/proj/secrets/gurobi-wls/versions/latest"
    assert store.secret_id("gcp/default") == "gcp--default"
    assert store.resource("projects/x/secrets/y") == "projects/x/secrets/y"


@pytest.mark.unit
@pytest.mark.parametrize("make", [_batch, _run, _vm], ids=["batch", "cloud-run", "compute-engine"])
def test_backend_contract(make):
    backend_contract(make, launch=contract_launch, timeout=5.0, poll_seconds=0.0)


@pytest.mark.unit
def test_batch_job_request_and_preemption():
    client = FakeBatchClient()
    backend = _batch(client=client)
    launch = contract_launch(
        "preempt",
        secret_env={"WSR_GUROBI_LICENSE": "gurobi-wls"},
        resources=Resources(cpus=2, memory_gb=6, spot=True, machine_type="n2-standard-4", max_wall_seconds=7200),
    )
    handle = backend.submit(launch)
    job = next(iter(client.jobs.values()))["spec"]
    spec = job["task_groups"][0]["task_spec"]
    assert spec["environment"]["secret_variables"] == {
        "WSR_GUROBI_LICENSE": "projects/proj/secrets/gurobi-wls/versions/latest"
    }
    assert "gurobi" not in str(spec["environment"]["variables"]).lower()
    assert spec["compute_resource"] == {"cpu_milli": 2000, "memory_mib": 6144} and spec["max_run_duration"] == "7200s"
    assert spec["max_retry_count"] == 0 and job["task_groups"][0]["task_count"] == 1
    assert job["allocation_policy"]["instances"][0]["policy"] == {
        "machine_type": "n2-standard-4",
        "provisioning_model": "SPOT",
    }
    assert job["labels"]["wsr_job_id"] == "job-contract"
    status = _wait(backend, handle)
    assert status.state == AttemptState.PREEMPTED and status.exit_code == 50001
    failed = _wait(backend, backend.submit(contract_launch("fail")))
    assert failed.state == AttemptState.FAILED and failed.exit_code == 3


@pytest.mark.unit
def test_quota_errors_become_capacity_errors():
    backend = _batch(client=FakeBatchClient(capacity=1))
    backend.submit(contract_launch("hang"))
    with pytest.raises(CapacityError, match="Quota"):
        backend.submit(contract_launch("succeed"))


@pytest.mark.unit
def test_cloud_run_limits_secrets_and_cleanup():
    clients = FakeRunClients()
    backend = _run(clients)
    with pytest.raises(CapacityError, match="24|86400"):
        backend.submit(contract_launch("succeed", resources=Resources(max_wall_seconds=2 * 86400)))
    with pytest.raises(CapacityError, match="vCPUs"):
        backend.submit(contract_launch("succeed", resources=Resources(cpus=16)))
    handle = backend.submit(contract_launch("succeed", secret_env={"WSR_GUROBI_LICENSE": "gurobi-wls"}))
    job = next(iter(clients.jobs_by_name.values()))
    env = job["template"]["template"]["containers"][0]["env"]
    secret = [e for e in env if e["name"] == "WSR_GUROBI_LICENSE"][0]
    assert secret == {
        "name": "WSR_GUROBI_LICENSE",
        "value_source": {"secret_key_ref": {"secret": "projects/proj/secrets/gurobi-wls", "version": "latest"}},
    }
    assert job["template"]["template"]["max_retries"] == 0
    assert _wait(backend, handle).state == AttemptState.SUCCEEDED
    assert clients.deleted and clients.deleted[0].endswith("/jobs/" + handle.split("/jobs/")[1].split("/")[0])
    assert backend.capabilities().supports_spot is False


@pytest.mark.unit
def test_vm_startup_script_carries_refs_not_secrets_and_survives_restarts():
    gcs = FakeGcsClient()
    instances = FakeInstancesClient(gcs)
    backend = _vm(gcs, instances)
    launch = contract_launch("fail", secret_env={"WSR_GUROBI_LICENSE": "gurobi-wls"}, resources=Resources(spot=True))
    handle = backend.submit(launch)
    resource = instances.inserted[-1]
    script = resource["metadata"]["items"][0]["value"]
    assert "gcpsm://proj/gurobi-wls" in script and "WSR_SECRET_REFS" in script
    assert (
        resource["scheduling"]["provisioning_model"] == "SPOT"
        and resource["scheduling"]["instance_termination_action"] == "DELETE"
    )
    assert "instances/wsr-task-fail" in script and "uploadType=media" in script
    fresh = _vm(gcs, instances)  # a new coordinator process polls the old handle after the VM deleted itself
    status = _wait(fresh, handle)
    assert status.state == AttemptState.FAILED and status.exit_code == 3
    preempted = _wait(backend, backend.submit(contract_launch("preempt", resources=Resources(spot=True))))
    assert preempted.state == AttemptState.PREEMPTED
    on_demand = _wait(backend, backend.submit(contract_launch("preempt")))
    assert on_demand.state == AttemptState.FAILED  # an on-demand VM gone without an exit code is a failure
    assert "serial output" in backend.logs(handle)


@pytest.mark.unit
def test_logs_come_from_the_runtime_upload():
    gcs = FakeGcsClient()
    log_store = GcsObjectStore("logs", client=gcs)
    backend = _batch(log_store=log_store)
    launch = contract_launch("succeed")
    handle = backend.submit(launch)
    gcs.bucket("logs").blob(f"work/job-contract/{launch.task.task_id}/logs/attempt-1.log").store[
        f"work/job-contract/{launch.task.task_id}/logs/attempt-1.log"
    ] = b"day 1\nday 2\nday 3"
    assert backend.logs(handle, tail=2) == "day 2\nday 3"


@pytest.mark.unit
def test_pricing_capabilities_registry_and_backend_table():
    assert price_per_hour("e2-standard-4", "us-central1", False) == 0.134
    assert price_per_hour("e2-standard-4", "europe-west1", True) == round(0.0402 * 1.10, 5)
    assert price_per_hour("nope") is None and price_per_hour() == 0.134
    batch = _batch()
    assert batch.price_per_hour("n2-standard-8", None, False) == round(0.3885 * 1.10, 5)
    assert batch.capabilities().machine_types["e2-standard-4"] == {"cpus": 4.0, "memory_gb": 16.0}
    assert _run().price_per_hour(None, "us-central1", True) == 0.0792  # no spot on Cloud Run

    class Tags:
        def get_tag(self, name):
            assert name.endswith("/repositories/wsr/packages/wsmart-route/tags/v1.1.0")
            return {"version": f"{name.split('/tags/')[0]}/versions/sha256:abc"}

    registry = ArtifactRegistry("proj", "europe-west1", client=Tags())
    assert registry.repository == "europe-west1-docker.pkg.dev/proj/wsr/wsmart-route"
    assert registry.resolve("v1.1.0") == "sha256:abc"
    gcs = FakeGcsClient()
    store = GcsObjectStore("b", client=gcs)
    loaded = load_backend(
        Provider.GCP, ComputeKind.VM, project="p", status_store=store, client=FakeInstancesClient(gcs)
    )
    assert isinstance(loaded, ComputeEngineBackend)
    assert isinstance(
        load_backend(Provider.GCP, ComputeKind.BATCH, client=FakeBatchClient(), secrets=_secrets()), GcpBatchBackend
    )
