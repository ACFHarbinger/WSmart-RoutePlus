"""Release image registries (CR-03) with a recording command runner, and the Dockerfile's invariants."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from logic.controllers.cloud.images import RegistryError, acr, artifact_registry, ecr, pinned

ROOT = Path(__file__).resolve().parents[5]
DIGEST = "sha256:" + "ab" * 32


class Runner:
    """Records commands; answers ``docker image inspect`` and ``manifest inspect`` like Docker does."""

    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def __call__(self, argv, input=None, capture_output=True, text=True, check=False):  # noqa: A002 - subprocess API
        self.calls.append((list(argv), input))
        if argv[:2] == ["docker", "push"] and "push" in self.fail:
            return SimpleNamespace(returncode=1, stdout="", stderr="denied: requested access")
        if argv[:3] == ["docker", "image", "inspect"]:
            repo = argv[-1].rsplit(":", 1)[0]
            return SimpleNamespace(
                returncode=0, stdout=json.dumps([f"other/repo@sha256:{'00' * 32}", f"{repo}@{DIGEST}"]), stderr=""
            )
        if argv[:3] == ["docker", "manifest", "inspect"]:
            if "manifest" in self.fail:
                return SimpleNamespace(returncode=1, stdout="", stderr="no such manifest")
            return SimpleNamespace(returncode=0, stdout=json.dumps({"Descriptor": {"digest": DIGEST}}), stderr="")
        if argv[:4] == ["aws", "ecr", "get-login-password", "--region"]:
            return SimpleNamespace(returncode=0, stdout="PASSWORD", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


@pytest.mark.unit
def test_artifact_registry_push_returns_the_pushed_digest():
    run = Runner()
    registry = artifact_registry("europe-west1", "proj", "wsr", run=run)
    digest = registry.push("wsmart-route:abc123", "abc123")
    assert digest == DIGEST
    remote = "europe-west1-docker.pkg.dev/proj/wsr/wsmart-route:abc123"
    assert [c[0] for c in run.calls[:3]] == [
        ["gcloud", "auth", "configure-docker", "europe-west1-docker.pkg.dev", "--quiet"],
        ["docker", "tag", "wsmart-route:abc123", remote],
        ["docker", "push", remote],
    ]
    assert pinned(registry, digest) == f"europe-west1-docker.pkg.dev/proj/wsr/wsmart-route@{DIGEST}"
    assert registry.resolve("abc123") == DIGEST


@pytest.mark.unit
def test_ecr_login_passes_the_password_on_stdin_only():
    run = Runner()
    ecr("123456789012", "eu-west-1", run=run).push("wsmart-route:abc", "abc")
    login = next(call for call in run.calls if call[0][:2] == ["docker", "login"])
    assert login[1] == "PASSWORD" and "PASSWORD" not in login[0]
    assert login[0][-1] == "123456789012.dkr.ecr.eu-west-1.amazonaws.com"


@pytest.mark.unit
def test_acr_and_failures():
    run = Runner(fail={"push", "manifest"})
    registry = acr("wsrregistry", run=run)
    assert run is registry.run and registry.repository == "wsrregistry.azurecr.io/wsmart-route"
    with pytest.raises(RegistryError, match="docker push"):
        registry.push("wsmart-route:abc", "abc")
    assert registry.resolve("abc") is None
    with pytest.raises(ValueError):
        pinned(registry, "latest")


@pytest.mark.unit
def test_dockerfile_invariants():
    text = (ROOT / "infra/cloud/image/Dockerfile").read_text()
    assert 'CMD ["python", "-m", "logic.controllers.cloud.runtime"]' in text
    assert not any(line.startswith("ENTRYPOINT") for line in text.splitlines())
    assert "USER wsr" in text and "ARG EXTRAS=solvers,cloud" in text
    copies = [line.split()[1:] for line in text.splitlines() if line.startswith("COPY")]
    assert copies == [["pyproject.toml", "/tmp/pyproject.toml"], [".", "/app"]]  # the release tree only
    assert "model_weights" not in text and "data/simulator" not in text.replace("(see", "")
