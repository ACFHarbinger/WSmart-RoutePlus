"""Keep provider cleanup state isolated from the developer's actual state directory."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_provider_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WSR_STATE_DIR", str(tmp_path / "provider-state"))
