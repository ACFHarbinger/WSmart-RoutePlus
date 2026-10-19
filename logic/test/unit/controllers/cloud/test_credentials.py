"""Credentials, keychain, secret references, redaction and profiles (CR-08)."""

import json
from pathlib import Path

import pytest
from logic.controllers.cloud import cli
from logic.controllers.cloud.auth.config import load_profiles
from logic.controllers.cloud.auth.credentials import (
    KeychainSecretStore,
    MissingCredentialError,
    ProviderCredentials,
    credential_account,
    detect_logins,
)
from logic.controllers.cloud.auth.secret_refs import parse_refs, redact, resolve_refs
from logic.controllers.cloud.runtime import EXIT_NO_SECRET, run_task
from logic.controllers.cloud.testing import secret_store_contract


class FakeKeyring:
    """The three ``keyring`` functions over a dict."""

    def __init__(self):
        self.store = {}

    def get_password(self, service, account):
        return self.store.get((service, account))

    def set_password(self, service, account, value):
        self.store[(service, account)] = value

    def delete_password(self, service, account):
        if (service, account) not in self.store:
            raise RuntimeError("PasswordDeleteError")
        del self.store[(service, account)]


@pytest.mark.unit
def test_detect_logins_from_files_and_environment(tmp_path):
    statuses = {s.provider: s for s in detect_logins(env={}, home=tmp_path)}
    assert not any(s.usable for s in statuses.values())

    (tmp_path / ".config/gcloud").mkdir(parents=True)
    (tmp_path / ".config/gcloud/application_default_credentials.json").write_text("{}")
    (tmp_path / ".aws").mkdir()
    (tmp_path / ".aws/credentials").write_text("[default]\naws_access_key_id=x\n")
    (tmp_path / ".aws/config").write_text("[profile dev]\nregion=eu-west-1\n")
    (tmp_path / ".azure").mkdir()
    for name in ("azureProfile.json", "msal_token_cache.json"):
        (tmp_path / ".azure" / name).write_text("{}")
    statuses = {s.provider: s for s in detect_logins(env={}, home=tmp_path)}
    assert all(s.usable for s in statuses.values())
    assert statuses["aws"].profiles == ["default", "dev"]

    env = {"AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "b"}
    assert {s.provider: s.source for s in detect_logins(env=env, home=tmp_path / "none")}["aws"] == "AWS_ACCESS_KEY_ID"


@pytest.mark.unit
def test_keychain_store_follows_the_studio_naming():
    secret_store_contract(lambda: KeychainSecretStore(FakeKeyring()))
    backend = FakeKeyring()
    KeychainSecretStore(backend).put(credential_account("AWS", "dev"), "k")
    assert ("wsmart-route.cloud", "aws/dev") in backend.store


@pytest.mark.unit
def test_provider_credentials():
    keychain = KeychainSecretStore(FakeKeyring())
    assert ProviderCredentials("aws", "default", "cli").aws_session_kwargs() == {}
    assert ProviderCredentials("aws", "dev", "cli").aws_session_kwargs() == {"profile_name": "dev"}
    assert ProviderCredentials("gcp", "default", "cli").gcp_credentials() is None
    with pytest.raises(MissingCredentialError):
        ProviderCredentials("aws", "dev", "keychain", keychain).aws_session_kwargs()
    keychain.put("aws/dev", json.dumps({"aws_access_key_id": "AK", "aws_secret_access_key": "SK"}))
    assert ProviderCredentials("aws", "dev", "keychain", keychain).aws_session_kwargs() == {
        "aws_access_key_id": "AK",
        "aws_secret_access_key": "SK",
    }


@pytest.mark.unit
def test_secret_refs_parse_resolve_and_redact():
    assert parse_refs(None) == {}
    refs = parse_refs(json.dumps({"WSR_GUROBI_LICENSE": "akv://vault/gurobi", "OTHER": "env://X"}))
    with pytest.raises(ValueError, match="unsupported"):
        parse_refs(json.dumps({"A": "http://x"}))
    with pytest.raises(ValueError, match="JSON object"):
        parse_refs("[1]")
    values = resolve_refs(refs, {"akv": lambda ref: f"lic-{ref.path.strip('/')}", "env": lambda ref: "x"})
    assert values == {"WSR_GUROBI_LICENSE": "lic-gurobi", "OTHER": "x"}

    def boom(ref):
        raise RuntimeError("leaky message with the value s3cr3t")

    with pytest.raises(LookupError) as info:
        resolve_refs({"V": "akv://v/n"}, {"akv": boom})
    assert "s3cr3t" not in str(info.value) and "V" in str(info.value)
    assert redact("id=WLSACCESSID=abc123 ok", ["WLSACCESSID=abc123\nWLSSECRET=zz"]) == "id=*** ok"


@pytest.mark.unit
def test_runtime_resolves_refs_and_never_stores_the_value(tmp_path, monkeypatch):
    from logic.controllers.cloud.model import GurobiMode
    from logic.test.unit.controllers.cloud.test_runtime import _job, _machine  # noqa: PLC0415 - shared fixtures

    secret = "WLSACCESSID=a1b2c3\nWLSSECRET=d4e5f6g7\nLICENSEID=42"
    monkeypatch.setenv("WSR_TEST_LICENSE", secret)
    cp, store, *_ = _job(tmp_path, days=1, gurobi=GurobiMode.WLS_SECRET, gurobi_secret="gurobi-wls")
    root, work = _machine(tmp_path, "m")
    code = run_task(
        cp,
        store,
        "job-1",
        "task-1",
        work,
        project_root=root,
        sync_seconds=0.05,
        secret_refs={"WSR_GUROBI_LICENSE": "env://WSR_TEST_LICENSE"},
    )
    assert code == 0
    license_file = Path(json.loads(next((work / "outputs").rglob("log_x_1N.json")).read_text())["license"])
    assert license_file.read_text().startswith("WLSACCESSID=a1b2c3")
    for path in [tmp_path / "cp.db", *(tmp_path / "store").rglob("*")]:
        if path.is_file():
            assert b"d4e5f6g7" not in path.read_bytes(), path

    monkeypatch.delenv("WSR_TEST_LICENSE")
    cp, store, *_ = _job(tmp_path / "again", days=1)
    root, work = _machine(tmp_path / "again", "m")
    code = run_task(cp, store, "job-1", "task-1", work, project_root=root, secret_refs={"X": "env://WSR_TEST_LICENSE"})
    assert code == EXIT_NO_SECRET and "X" in cp.get_task("task-1").message


@pytest.mark.unit
def test_profiles_file(tmp_path, monkeypatch):
    path = tmp_path / "cloud.yaml"
    assert load_profiles(path) == {}
    path.write_text(
        "profiles:\n  g:\n    provider: gcp\n    control_plane: firestore://p\n    store: gs://b/x\n"
        "    credentials: keychain\n    keychain_profile: ci\n    gurobi_secret: gurobi-wls\n"
    )
    profile = load_profiles(path)["g"]
    assert profile.compute == "batch" and profile.provider_credentials().source == "keychain"
    assert profile.remote_config(tmp_path).store_uri == "gs://b/x"
    for bad, message in [
        ("provider: ibm", "provider"),
        ("credentials: password", "credentials"),
        ("store: ftp://x", "store"),
        ("api_key: abc", "unknown key"),
        ("compute: process", "local backend"),
    ]:
        key = bad.split(":")[0]
        body = {"provider": "gcp", "control_plane": "firestore://p", "store": "gs://b"}
        lines = [f"    {k}: {v}" for k, v in body.items() if k != key] + [f"    {bad}"]
        path.write_text("profiles:\n  g:\n" + "\n".join(lines) + "\n")
        with pytest.raises(ValueError, match=message):
            load_profiles(path)
    monkeypatch.setenv("WSR_CLOUD_CONFIG", str(tmp_path / "missing.yaml"))
    assert load_profiles() == {}


@pytest.mark.unit
def test_cli_secrets_never_print_values(tmp_path, monkeypatch, capsys):
    backend = FakeKeyring()
    monkeypatch.setattr(
        "logic.controllers.cloud.auth.credentials.KeychainSecretStore.__init__",
        lambda self, backend_=None: setattr(self, "backend", backend),
    )
    value_file = tmp_path / "lic"
    value_file.write_text("WLSSECRET=topsecret")
    base = {"root": str(tmp_path), "json": True, "cloud_command": "secrets"}
    cli.run_cloud({**base, "action": "set", "name": "gurobi-wls", "from_file": str(value_file)})
    assert backend.store[("wsmart-route.cloud", "secret/gurobi-wls")] == "WLSSECRET=topsecret"
    assert cli.run_cloud({**base, "action": "exists", "name": "gurobi-wls"}) is True
    with pytest.raises(ValueError, match="never taken as arguments"):
        cli.run_cloud({**base, "action": "set", "name": "x"})
    cli.run_cloud({**base, "action": "delete", "name": "gurobi-wls"})
    assert "topsecret" not in capsys.readouterr().out
