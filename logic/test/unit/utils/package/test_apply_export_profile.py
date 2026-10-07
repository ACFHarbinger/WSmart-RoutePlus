"""apply_export_profile: entry-point choice, engine selection and the removed-import check."""

import pytest
from logic.package.apply_export_profile import apply_engines, apply_entrypoints, check_imports, write_file_list

pytestmark = [pytest.mark.unit, pytest.mark.fast]

S = "logic/src/policies/swc"
CATALOGUE = {
    "_comment": "ignored",
    "swc_tcf": {
        "engines": {
            "gurobi": {"files": [], "dependencies": ["gurobipy"]},
            "ortools": {"files": [f"{S}/ortools_wrapper.py"], "dependencies": ["ortools"]},
            "pyomo": {"files": [f"{S}/pyomo_wrapper.py"], "dependencies": ["pyomo"]},
        },
        "default_framework": {
            "value": "ortools",
            "rewrites": [{"files": [f"{S}/params.py"], "pattern": 'framework: str = "{value}"', "replacement": 'framework: str = "{value}"'}],
        },
    },
}


def _tree(tmp_path):
    (tmp_path / S).mkdir(parents=True)
    for name in ("ortools_wrapper.py", "pyomo_wrapper.py", "gurobi.py"):
        (tmp_path / S / name).write_text("")
    (tmp_path / S / "params.py").write_text('framework: str = "ortools"\n')
    (tmp_path / "logic" / "pyproject.toml").write_text(
        'solvers = [\n    "gurobipy>=11.0.3",\n    "ortools>=9.4.1874",\n    "pyomo>=6.9.5",\n]\n'
    )
    (tmp_path / "main.py").write_text("")
    (tmp_path / "__main__.py").write_text("")
    return tmp_path


@pytest.mark.parametrize(("choice", "kept"), [("both", {"main.py", "__main__.py"}), ("main", {"main.py"}),
                                              ("dunder", {"__main__.py"}), ("none", set())])
def test_entrypoints(tmp_path, choice, kept):
    root = _tree(tmp_path)
    apply_entrypoints(root, choice, dry_run=False)
    assert {p for p in ("main.py", "__main__.py") if (root / p).exists()} == kept


def test_engine_selection_drops_files_dependencies_and_default(tmp_path):
    root = _tree(tmp_path)
    apply_engines(root, CATALOGUE, {"swc_tcf": ["gurobi"]}, dry_run=False)
    assert not (root / S / "ortools_wrapper.py").exists() and not (root / S / "pyomo_wrapper.py").exists()
    pyproject = (root / "logic" / "pyproject.toml").read_text()
    assert "gurobipy" in pyproject and "ortools" not in pyproject and "pyomo" not in pyproject
    assert (root / S / "params.py").read_text() == 'framework: str = "gurobi"\n'


def test_default_engine_selection_keeps_everything(tmp_path):
    root = _tree(tmp_path)
    apply_engines(root, CATALOGUE, {}, dry_run=False)
    assert (root / S / "ortools_wrapper.py").exists() and (root / S / "pyomo_wrapper.py").exists()
    assert (root / S / "params.py").read_text() == 'framework: str = "ortools"\n'


def test_unknown_engine_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        apply_engines(_tree(tmp_path), CATALOGUE, {"swc_tcf": ["cplex"]}, dry_run=False)


def test_check_imports_reports_removed_modules_but_not_guarded_ones(tmp_path):
    pkg = tmp_path / "logic" / "src"
    pkg.mkdir(parents=True)
    (tmp_path / "logic" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text(
        "from logic.src.gone import x\ntry:\n    from logic.src.optional import y\nexcept ImportError:\n    y = None\n"
    )
    assert check_imports(tmp_path) == ["logic/src/a.py: logic.src.gone"]


def test_file_list_builds_source_and_saves_the_tree(tmp_path):
    (tmp_path / "logic" / "src" / "__pycache__").mkdir(parents=True)
    (tmp_path / "logic" / "src" / "mod.py").write_text("")
    (tmp_path / "logic" / "src" / "__pycache__" / "mod.cpython-310.pyc").write_text("")
    (tmp_path / "__main__.py").write_text("")
    out = write_file_list(tmp_path, "assets/files/FILE_LIST.txt", dry_run=False)
    assert out == tmp_path / "assets" / "files" / "FILE_LIST.txt"
    assert sorted(p.name for p in (tmp_path / "source").iterdir()) == ["__main__.py", "logic"]
    listing = out.read_text()
    assert listing.startswith(".") and "__main__.py" in listing and "mod.py" in listing and "__pycache__" not in listing
