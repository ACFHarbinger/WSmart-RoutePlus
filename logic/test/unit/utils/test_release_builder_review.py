"""Release closure includes runtime imports and never deletes an existing destination."""

from pathlib import Path

import pytest
from logic.package import build_release as builder
from logic.package import release_closure as closure


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
    assert closure.LAZY_LINE_RE.search(source).group(2) == "Thing"


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
    monkeypatch.setattr(closure, "REPO", tmp_path)
    monkeypatch.setattr(closure, "LOGIC", tmp_path / "logic")
    monkeypatch.setattr(closure, "STAGES", {})
    kept, _ = builder.compute_closure(
        {"infrastructure": ["logic.src.feature.entry"], "stages": {}, "excluded_packages": []}
    )
    assert {
        Path("logic/src/__init__.py"),
        Path("logic/src/feature/__init__.py"),
        Path("logic/src/feature/runtime.py"),
    } <= {path.relative_to(tmp_path) for path in kept}


@pytest.mark.parametrize("quote", ["'", '"'])
def test_registry_manifest_accepts_formatted_quotes(tmp_path, quote):
    path = tmp_path / "_manifest.py"
    path.write_text(
        "ENTRIES = {\n    "
        + quote
        + "keep"
        + quote
        + ": "
        + quote
        + "logic.src.keep:Keep"
        + quote
        + ",\n    "
        + quote
        + "drop"
        + quote
        + ": "
        + quote
        + "logic.src.drop:Drop"
        + quote
        + ",\n}\n"
    )
    assert set(builder.parse_registry_manifest(path)) == {"keep", "drop"}
    builder.filter_manifest_file(path, {"keep"})
    assert builder.parse_registry_manifest(path) == {"keep": "logic.src.keep:Keep"}


def test_release_keeps_selected_solver_extra_without_unreleased_backends(tmp_path):
    import tomli

    builder.write_release_pyproject(
        {"version": "1.0.0", "dependency_extras": {"solvers": ["gurobipy", "ortools", "pyvrp", "alns", "fast-tsp"]}},
        tmp_path,
    )
    project = tomli.loads((tmp_path / "pyproject.toml").read_text())["project"]
    extra = project["optional-dependencies"]["solvers"]
    assert any(value.startswith("gurobipy") for value in extra)
    assert not any(value.startswith(("hexaly", "vrpy", "pyomo", "torch")) for value in extra)
    assert not any(value.startswith("torch") for value in project["dependencies"])


def test_lazy_closure_does_not_depend_on_queue_order(tmp_path, monkeypatch):
    sources = {
        "logic/__init__.py": "",
        "logic/src/__init__.py": "",
        "logic/src/entry.py": "from logic.src.facade import Thing\n",
        "logic/src/facade/__init__.py": "_LAZY = {\n    'Thing': (\n        'logic.src.facade.impl:Thing'\n    ),\n}\n",
        "logic/src/facade/impl.py": "class Thing: pass\n",
    }
    for name, text in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(closure, "REPO", tmp_path)
    monkeypatch.setattr(closure, "LOGIC", tmp_path / "logic")
    monkeypatch.setattr(closure, "STAGES", {})
    kept, _ = builder.compute_closure({"stages": {}, "infrastructure": ["logic.src.entry"], "excluded_packages": []})
    assert tmp_path / "logic/src/facade/impl.py" in kept


def test_multiline_lazy_map_pruning_is_format_independent(tmp_path, monkeypatch):
    package = tmp_path / "logic/src/facade"
    package.mkdir(parents=True)
    (package / "keep.py").write_text("")
    (package / "drop.py").write_text("")
    path = package / "__init__.py"
    path.write_text(
        '_LAZY = {\n"Keep": ("logic.src.facade.keep:Keep"),\n"Drop": (\n"logic.src.facade.drop:Drop"\n),\n}\n'
    )
    monkeypatch.setattr(closure, "REPO", tmp_path)
    dropped = builder.filter_lazy_map(path, {"logic.src.facade.keep"}, tmp_path)
    assert dropped
    assert closure.lazy_targets(path) == {"Keep": "logic.src.facade.keep"}


def test_staged_bin_analysis_modules_use_the_importable_name(tmp_path):
    pkg = tmp_path / "logic" / "submodules" / "wsmart_bin_analysis" / "export"
    pkg.mkdir(parents=True)
    (pkg / "grid.py").write_text("")
    assert "logic.submodules.wsmart_bin_analysis.export.grid" in builder.staged_module_names(tmp_path)


def test_only_released_task_configs_ship(tmp_path):
    for name in ("test_sim", "train", "hpo", "meta_train"):
        (tmp_path / f"{name}.yaml").write_text("")
    assert builder.prune_task_configs(tmp_path, ["test_sim", "train"]) == ["hpo", "meta_train"]
    assert sorted(p.stem for p in tmp_path.glob("*.yaml")) == ["test_sim", "train"]
    assert builder.prune_task_configs(tmp_path, None) == []


def test_gen_data_does_not_overwrite_by_default():
    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parents[3] / "configs" / "tasks" / "gen_data.yaml").read_text())
    assert cfg["data"]["overwrite"] is False

