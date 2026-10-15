"""Release closure includes runtime imports and never deletes an existing destination."""

from pathlib import Path

import pytest
from logic.package import build_release as builder


def test_build_refuses_existing_user_files_before_computing_closure(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "user.txt"
    sentinel.write_text("keep")
    monkeypatch.setattr(builder, "compute_closure", lambda _: pytest.fail("must reject destination first"))
    with pytest.raises(ValueError, match="absent or empty"):
        builder.run_build({}, output)
    assert sentinel.read_text() == "keep"


def test_build_refuses_repository_destination():
    with pytest.raises(ValueError, match="repository"):
        builder.run_build({}, builder.REPO)


def test_multiline_lazy_map_is_detected():
    source = '_LAZY = {\n    "Thing": "logic.src.thing:Thing",\n}\n'
    assert builder.LAZY_LINE_RE.search(source).group(2) == "Thing"


def test_closure_keeps_parent_initializers_and_deferred_runtime_imports(tmp_path, monkeypatch):
    for name, source in {
        "logic/__init__.py": "",
        "logic/src/__init__.py": "",
        "logic/src/feature/__init__.py": "",
        "logic/src/feature/entry.py": "def run():\n    from logic.src.feature.runtime import execute\n    return execute()\n",
        "logic/src/feature/runtime.py": "def execute(): return 42\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.setattr(builder, "REPO", tmp_path)
    monkeypatch.setattr(builder, "LOGIC", tmp_path / "logic")
    monkeypatch.setattr(builder, "STAGES", {})
    kept, _ = builder.compute_closure(
        {"infrastructure": ["logic.src.feature.entry"], "stages": {}, "excluded_packages": []}
    )
    assert {
        Path("logic/src/__init__.py"),
        Path("logic/src/feature/__init__.py"),
        Path("logic/src/feature/runtime.py"),
    } <= {path.relative_to(tmp_path) for path in kept}
