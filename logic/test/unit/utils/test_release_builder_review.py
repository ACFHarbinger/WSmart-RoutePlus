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


def test_release_pins_python_and_drops_listed_requirements(tmp_path):
    import tomli

    builder.write_release_pyproject(
        {
            "version": "1.0.2",
            "python": {"requires": ">=3.10,<3.13", "version": "3.12"},
            "dropped_requirements": ["torch-scatter", "torch_sparse"],
        },
        tmp_path,
    )
    text = (tmp_path / "pyproject.toml").read_text()
    project = tomli.loads(text)["project"]
    assert project["requires-python"] == ">=3.10,<3.13"
    assert (tmp_path / ".python-version").read_text() == "3.12\n"
    neural = project["optional-dependencies"]["neural"]
    assert any(value.startswith("torch==") for value in neural)
    assert not any(
        value.startswith(("torch-scatter", "torch-sparse", "torch_scatter", "torch_sparse")) for value in neural
    )
    assert "data.pyg.org" not in text  # no pyg index without the packages that use it


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


def test_release_conftest_registers_only_shipped_fixture_plugins(tmp_path):
    from logic.package.release_staging import write_release_conftest

    source = tmp_path / "conftest_src.py"
    source.write_text(
        'import pytest\npytest_plugins = [\n    "logic.test.fixtures.kept",\n    "logic.test.fixtures.gone",\n]\n'
    )
    fixtures = tmp_path / "out" / "logic" / "test" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "kept.py").write_text("")
    assert write_release_conftest(source, tmp_path / "out") == ["logic.test.fixtures.gone"]
    staged = (tmp_path / "out" / "logic" / "test" / "conftest.py").read_text()
    assert '"logic.test.fixtures.kept"' in staged and "gone" not in staged


def test_release_requirements_pin_to_the_installed_versions():
    from importlib.metadata import version

    from logic.package.release_files import pin_to_tested

    assert pin_to_tested("pytest>=1.0") == f"pytest=={version('pytest')}"
    assert pin_to_tested("surely-not-installed-wsr>=1") == "surely-not-installed-wsr>=1"


def test_release_pyproject_carries_the_pytest_settings(tmp_path):
    from logic.package.release_files import pytest_ini_section

    root = tmp_path / "pyproject.toml"
    root.write_text(
        '[project]\nname = "x"\n\n[tool.pytest.ini_options]\nlog_level = "INFO"\n\n# ===\n[tool.coverage.run]\n'
    )
    assert pytest_ini_section(root) == ["[tool.pytest.ini_options]", 'log_level = "INFO"']


def test_indexed_config_init_ships_only_the_configs_kept_code_imports(tmp_path, monkeypatch):
    configs = "logic/src/configs/policies"
    sources = {
        "logic/__init__.py": "",
        "logic/src/__init__.py": "",
        "logic/src/configs/__init__.py": "",
        f"{configs}/__init__.py": (
            "from logic.src.configs.policies.a import AConfig\n"
            "from logic.src.configs.policies.b import BConfig\n"
            "from logic.src.configs.policies.c import CConfig\n"
        ),
        f"{configs}/a.py": "class AConfig: pass\n",
        f"{configs}/b.py": "class BConfig: pass\n",
        f"{configs}/c.py": "class CConfig: pass\n",
        "logic/src/entry.py": (
            "from logic.src.configs.policies import AConfig\nfrom logic.src.configs.policies.c import CConfig\n"
        ),
    }
    for name, text in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(closure, "REPO", tmp_path)
    monkeypatch.setattr(closure, "LOGIC", tmp_path / "logic")
    monkeypatch.setattr(closure, "STAGES", {})
    kept, _ = builder.compute_closure({"stages": {}, "infrastructure": ["logic.src.entry"], "excluded_packages": []})
    kept = {path.relative_to(tmp_path).as_posix() for path in kept}
    assert {f"{configs}/__init__.py", f"{configs}/a.py", f"{configs}/c.py"} <= kept
    assert f"{configs}/b.py" not in kept


def test_pruned_init_drops_the_exports_and_docstring_lines_of_removed_names(tmp_path, monkeypatch):
    from logic.package import release_staging as staging

    package = tmp_path / "logic/src/facade"
    package.mkdir(parents=True)
    (package / "keep.py").write_text("class Keep: pass\n")
    (package / "drop.py").write_text("class Drop: pass\n")
    path = package / "__init__.py"
    path.write_text(
        '"""Facade.\n\nAttributes:\n    Keep: Kept.\n\n    Drop: Dropped.\n\nExample:\n    >>> 1\n    1\n"""\n\n'
        "from logic.src.facade.keep import Keep\nfrom logic.src.facade.drop import Drop\n\n"
        '__all__ = [\n    "Keep",\n    "Drop",\n]\n'
    )
    monkeypatch.setattr(closure, "REPO", tmp_path)
    assert staging.drop_pruned_init_imports(path, {"logic.src.facade", "logic.src.facade.keep"}, tmp_path)
    text = path.read_text()
    assert "Drop" not in text
    assert '    "Keep",' in text and "    Keep: Kept." in text and ">>> 1" in text


def _tree(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _point_closure_at(monkeypatch, root):
    monkeypatch.setattr(closure, "REPO", root)
    monkeypatch.setattr(closure, "LOGIC", root / "logic")
    monkeypatch.setattr(closure, "STAGES", {})


def test_index_all_inits_ships_only_used_reexports_and_follows_names_the_init_uses(tmp_path, monkeypatch):
    _tree(
        tmp_path,
        {
            "logic/__init__.py": "",
            "logic/src/__init__.py": "",
            "logic/src/pkg/__init__.py": "from .used import Used\nfrom .unused import Unused\nfrom .core import Core\n"
            "DEFAULT = Core()\n",
            "logic/src/pkg/used.py": "class Used: pass\n",
            "logic/src/pkg/unused.py": "class Unused: pass\n",
            "logic/src/pkg/core.py": "class Core: pass\n",
            "logic/src/entry.py": "from logic.src.pkg import Used\n",
        },
    )
    _point_closure_at(monkeypatch, tmp_path)
    manifest = {"stages": {}, "infrastructure": ["logic.src.entry"], "excluded_packages": [], "index_all_inits": True}
    kept = {path.relative_to(tmp_path).as_posix() for path in builder.compute_closure(manifest)[0]}
    assert {"logic/src/pkg/used.py", "logic/src/pkg/core.py"} <= kept
    assert "logic/src/pkg/unused.py" not in kept


def test_import_error_guard_and_type_checking_make_an_excluded_import_optional(tmp_path, monkeypatch):
    _tree(
        tmp_path,
        {
            "logic/__init__.py": "",
            "logic/src/__init__.py": "",
            "logic/src/extra/__init__.py": "",
            "logic/src/entry.py": "from typing import TYPE_CHECKING\n"
            "try:\n    from logic.src.extra import thing\nexcept ImportError:\n    thing = None\n"
            "if TYPE_CHECKING:\n    from logic.src.extra import Other\n",
        },
    )
    _point_closure_at(monkeypatch, tmp_path)
    manifest = {"stages": {}, "infrastructure": ["logic.src.entry"], "excluded_packages": ["logic.src.extra"]}
    kept = {path.relative_to(tmp_path).as_posix() for path in builder.compute_closure(manifest)[0]}
    assert "logic/src/entry.py" in kept and "logic/src/extra/__init__.py" not in kept


def test_a_stage_listed_without_keys_does_not_ship_its_manifest(tmp_path, monkeypatch):
    _tree(tmp_path, {"logic/__init__.py": "", "logic/src/__init__.py": "", "logic/src/entry.py": ""})
    _point_closure_at(monkeypatch, tmp_path)
    stage = tmp_path / "logic/src/stage/_manifest.py"
    _tree(tmp_path, {"logic/src/stage/_manifest.py": "ENTRIES = {}\n"})
    monkeypatch.setattr(closure, "STAGES", {"improvement": stage})
    base = {"infrastructure": ["logic.src.entry"], "excluded_packages": []}
    assert stage in builder.compute_closure({**base, "stages": {}})[0]
    assert stage not in builder.compute_closure({**base, "stages": {"improvement": []}})[0]


def test_unguarded_imports_of_pruned_modules_are_reported(tmp_path):
    from logic.package import release_staging as staging

    _tree(
        tmp_path,
        {
            "logic/src/run.py": "def go():\n    from logic.src.gone import x\n"
            "def safe():\n    try:\n        from logic.src.gone import y\n    except ImportError:\n        pass\n",
        },
    )
    problems = staging.unguarded_missing_imports(tmp_path, {"logic", "logic.src", "logic.src.run"})
    assert len(problems) == 1 and "logic.src.gone" in problems[0] and "run.py:2" in problems[0]


def test_a_profile_manifest_applies_its_patches_to_a_copy_of_the_tracked_source(tmp_path):
    manifest_dir = tmp_path / "resources"
    patches = manifest_dir / "p"
    patches.mkdir(parents=True)
    (patches / "0001.patch").write_text(
        "diff --git a/PATCHED.txt b/PATCHED.txt\nnew file mode 100644\n--- /dev/null\n+++ b/PATCHED.txt\n"
        "@@ -0,0 +1 @@\n+patched\n"
    )
    manifest = {"_path": str(manifest_dir / "m.yaml"), "patches": "p"}
    source = builder.prepare_source(manifest, tmp_path / "source", tracked=["logic/package/build_release.py"])
    assert (source / "PATCHED.txt").read_text() == "patched\n"
    assert (source / "logic" / "package" / "build_release.py").is_file()
    assert not (builder.REPO / "PATCHED.txt").exists()
