# 7. Test trace_dependencies
from pathlib import Path
from unittest.mock import patch

import pytest

# 4. Test check_circular_imports
from logic.validation.check_circular_imports import (
    build_graph,
    file_to_module,
    generate_html,
    resolve_to_module,
    tarjan_sccs,
)
from logic.validation.check_circular_imports import (
    main as main_circular,
)

# 9. Test check_interface_compliance
from logic.validation.check_nested_imports import (
    analyze_file as analyze_nested,
)

# 3. Test check_nested_imports
from logic.validation.check_nested_imports import (
    is_constant_expression,
    is_constant_guarded_if,
    is_header_assignment,
    is_header_setup_call,
    is_import_error_try_block,
    is_suppress_import_error_block,
    is_type_checking_block,
)
from logic.validation.check_nested_imports import (
    main as main_nested,
)
from logic.validation.check_nested_imports import (
    print_stats_table as print_stats_nested,
)
from logic.validation.check_relative_imports import (
    analyze_file as analyze_relative,
)

# 1. Test check_relative_imports
from logic.validation.check_relative_imports import (
    format_relative_import,
)
from logic.validation.check_relative_imports import (
    main as main_relative,
)
from logic.validation.check_relative_imports import (
    print_stats_table as print_stats_relative,
)

# 2. Test check_unused_imports
from logic.validation.check_unused_imports import (
    analyze_file as analyze_unused,
)
from logic.validation.check_unused_imports import (
    main as main_unused,
)

# 5. Test count_loc

# 6. Test tree_loc
# 8. Test visualize_module_graph

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_relative_imports_format():
    import ast

    node = ast.parse("from . import a, b as c").body[0]
    # pyrefly: ignore [bad-argument-type]
    assert format_relative_import(node) == "from . import a, b as c"


def test_relative_imports_analyze(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("from .a import b\nfrom ..c import d as e\nimport sys\n")
    results = analyze_relative(f)
    assert len(results) == 2
    assert results[0][0] == 1
    assert results[0][1] == 1
    assert results[1][0] == 2
    assert results[1][1] == 2

    # Syntax error file
    f_err = tmp_path / "err.py"
    f_err.write_text("invalid python code ...")
    assert analyze_relative(f_err) == []


def test_relative_imports_print_stats(capsys):
    results = {"package/file.py": [(1, 1, "from .a import b")]}
    print_stats_relative(results, Path("package"))


def test_relative_imports_main(tmp_path, capsys):
    f = tmp_path / "sample.py"
    f.write_text("from .a import b\n")

    with patch("sys.argv", ["check_relative_imports.py", str(tmp_path)]):
        main_relative()
    captured = capsys.readouterr()
    assert "Found 1 relative import" in captured.out

    # Test non-existent dir
    with patch("sys.argv", ["check_relative_imports.py", "nonexistent_dir"]), pytest.raises(SystemExit) as exc:
        main_relative()
    assert exc.value.code == 1


def test_unused_imports_analyze(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("import sys\nimport os\nprint(sys.argv)\n")
    results = analyze_unused(f)
    assert len(results) == 1
    assert results[0] == (2, "os")

    f_star = tmp_path / "star.py"
    f_star.write_text("from os import *\n")
    assert analyze_unused(f_star) == []

    f_err = tmp_path / "err.py"
    f_err.write_text("invalid python code ...")
    assert analyze_unused(f_err) == []

    f_factory = tmp_path / "factory.py"
    f_factory.write_text("class MyFactory:\n    import math\n")
    assert len(analyze_unused(f_factory, ignore_factories=False)) == 1
    assert len(analyze_unused(f_factory, ignore_factories=True)) == 0


def test_unused_imports_main(tmp_path, capsys):
    f = tmp_path / "sample.py"
    f.write_text("import os\n")

    with patch("sys.argv", ["check_unused_imports.py", str(tmp_path)]):
        main_unused()
    captured = capsys.readouterr()
    assert "Unused Imports Found" in captured.out

    f_clean = tmp_path / "clean.py"
    f_clean.write_text("import sys\nprint(sys.argv)\n")
    with patch("sys.argv", ["check_unused_imports.py", str(f_clean)]):
        main_unused()
    captured = capsys.readouterr()
    assert "No unused imports found" in captured.out

    with patch("sys.argv", ["check_unused_imports.py", "nonexistent.py"]), pytest.raises(SystemExit) as exc:
        main_unused()
    assert exc.value.code == 1


def test_nested_imports_helpers():
    import ast

    node_tc = ast.parse("if TYPE_CHECKING:\n    import sys").body[0]
    assert is_type_checking_block(node_tc)

    node_try = ast.parse("try:\n    import foo\nexcept ImportError:\n    pass").body[0]
    assert is_import_error_try_block(node_try)

    node_suppress = ast.parse("with contextlib.suppress(ImportError):\n    import bar").body[0]
    assert is_suppress_import_error_block(node_suppress)

    # pyrefly: ignore [missing-attribute]
    node_const = ast.parse("X = 42").body[0].value
    assert is_constant_expression(node_const)

    node_header = ast.parse("logger = get_pylogger(__name__)").body[0]
    assert is_header_assignment(node_header)

    node_if = ast.parse("if SOME_FLAG:\n    pass").body[0]
    assert is_constant_guarded_if(node_if)

    node_call = ast.parse("matplotlib.use('Agg')").body[0]
    assert is_header_setup_call(node_call)


def test_nested_imports_analyze(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("import sys\ndef foo():\n    import math\n")
    results = analyze_nested(f)
    assert len(results) == 1
    assert results[0] == (3, "math")

    # Syntax error file
    f_err = tmp_path / "err.py"
    f_err.write_text("invalid python code ...")
    assert analyze_nested(f_err) == []


def test_nested_imports_print_stats():
    results = {"package/file.py": [(1, "math")]}
    print_stats_nested(results, Path("package"))


def test_nested_imports_main(tmp_path, capsys):
    f = tmp_path / "sample.py"
    f.write_text("def foo():\n    import math\n")

    with patch("sys.argv", ["check_nested_imports.py", str(tmp_path)]):
        main_nested()
    captured = capsys.readouterr()
    assert "Found 1 nested imports" in captured.out

    # Test non-existent dir
    with patch("sys.argv", ["check_nested_imports.py", "nonexistent_dir"]):
        main_nested()
    captured = capsys.readouterr()
    assert "does not exist" in captured.out


def test_circular_imports_helpers(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    f = root / "module" / "sub.py"
    f.parent.mkdir()
    f.write_text("")
    assert file_to_module(f, root) == "module.sub"

    known = {"module.sub", "module.other"}
    assert resolve_to_module("sub", 1, "module.other", known) == "module.sub"


def test_circular_imports_build_graph(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    a = root / "a.py"
    b = root / "b.py"
    a.write_text("import b\n")
    b.write_text("import a\n")

    graph = build_graph(root, set())
    assert "a" in graph
    assert "b" in graph
    assert "b" in graph["a"]
    assert "a" in graph["b"]


def test_circular_imports_tarjan():
    graph = {
        "a": {"b"},
        "b": {"c"},
        "c": {"a", "d"},
        "d": set(),
    }
    sccs = tarjan_sccs(graph)
    assert len(sccs) == 1
    assert set(sccs[0]) == {"a", "b", "c"}


def test_circular_imports_html(tmp_path):
    cycles = [["a", "b"]]
    graph = {"a": {"b"}, "b": {"a"}}
    output = tmp_path / "graph.html"

    with patch("logic.validation.check_circular_imports.Network") as mock_net:
        generate_html(cycles, graph, output)
        assert mock_net.called


def test_circular_imports_main(tmp_path, capsys):
    root = tmp_path / "src"
    root.mkdir()
    a = root / "a.py"
    b = root / "b.py"
    a.write_text("import b\n")
    b.write_text("import a\n")

    with patch("sys.argv", ["check_circular_imports.py", str(root)]), pytest.raises(SystemExit) as exc:
        main_circular()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Found 1 circular import group" in captured.out

    # Clean run
    a.write_text("")
    b.write_text("")
    with patch("sys.argv", ["check_circular_imports.py", str(root)]), pytest.raises(SystemExit) as exc:
        main_circular()
    assert exc.value.code == 0

    # Non-existent dir
    with patch("sys.argv", ["check_circular_imports.py", "nonexistent_dir"]), pytest.raises(SystemExit) as exc:
        main_circular()
    assert exc.value.code == 1


# --- Tests for check_interface_compliance ---


# --- Tests for check_embedded_languages.py ---


# --- Tests for check_multi_classes.py ---


# --- Tests for check_type_coverage.py ---
