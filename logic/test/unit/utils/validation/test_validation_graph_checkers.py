"""Tests of the LoC counters and the dependency/module-graph tools in ``logic/validation``."""

import ast
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from logic.validation.count_loc import (
    analyze_file as analyze_count,
)
from logic.validation.count_loc import (
    get_docstring_lines as get_doc_count,
)
from logic.validation.count_loc import (
    group_by_directory as group_count,
)
from logic.validation.count_loc import (
    main as main_count,
)
from logic.validation.trace_dependencies import (
    ASTScopeVisitor,
    DependencyGrapher,
)
from logic.validation.trace_dependencies import (
    main as main_trace,
)
from logic.validation.tree_loc import (
    analyze_file as analyze_tree,
)
from logic.validation.tree_loc import (
    main as main_tree,
)
from logic.validation.tree_loc import (
    print_tree,
)
from logic.validation.visualize_module_graph import (
    build_graph as build_graph_visualize,
)
from logic.validation.visualize_module_graph import (
    collect_module_map,
    condense_to_packages,
    find_violations,
    get_layer,
)
from logic.validation.visualize_module_graph import (
    file_to_module as file_to_module_visualize,
)
from logic.validation.visualize_module_graph import (
    generate_html as generate_html_visualize,
)
from logic.validation.visualize_module_graph import (
    main as main_visualize,
)
from logic.validation.visualize_module_graph import (
    resolve_to_module as resolve_to_module_visualize,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_count_loc_helpers(tmp_path):
    source = '"""module docstring"""\ndef foo():\n    """func docstring"""\n    # a comment\n    return 42\n'
    doc_lines = get_doc_count(source)
    assert len(doc_lines) > 0

    f = tmp_path / "sample.py"
    f.write_text(source)
    metrics = analyze_count(str(f))
    assert metrics["code"] == 2
    assert metrics["comment"] == 1
    assert metrics["docstring"] == 2

    # Test grouping
    data = [
        {"path": "a/b/c.py", "code": 10, "comment": 2, "docstring": 3, "total": 15},
        {"path": "a/b/d.py", "code": 5, "comment": 1, "docstring": 1, "total": 7},
    ]
    grouped = group_count(data, 2)
    assert len(grouped) == 1
    assert grouped[0]["path"] == "a/b"
    assert grouped[0]["code"] == 15


def test_count_loc_main(tmp_path, capsys):
    f = tmp_path / "sample.py"
    f.write_text("import sys\n")

    with patch("sys.argv", ["count_loc.py", str(tmp_path)]):
        main_count()
    captured = capsys.readouterr()
    assert "Codebase Analysis" in captured.out


def test_tree_loc_helpers(tmp_path, capsys):
    f = tmp_path / "sample.py"
    f.write_text("import sys\n# comment\n")
    c, m, d = analyze_tree(str(f))
    assert c == 1
    assert m == 1
    assert d == 0

    # Print tree test
    print_tree(str(tmp_path))
    captured = capsys.readouterr()
    assert "sample.py" in captured.out


def test_tree_loc_main(tmp_path, capsys):
    with patch("sys.argv", ["tree_loc.py", str(tmp_path)]):
        main_tree()
    captured = capsys.readouterr()
    assert "Structure" in captured.out


def test_trace_dependencies_visitor():
    source = """
import sys
from os import path
class A:
    def method(self):
        pass
def func():
    x = 10
"""
    visitor = ASTScopeVisitor()
    tree = ast.parse(source)
    visitor.generic_visit(tree)

    assert "A" in visitor.flat_defs
    assert "func" in visitor.flat_defs
    assert "x" in visitor.flat_defs
    assert "sys" in visitor.imports_graph
    assert "path" in visitor.imports_graph


def test_dependency_grapher_tracing(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    a_py = src_dir / "a.py"
    b_py = src_dir / "b.py"

    a_py.write_text("from src.b import TargetClass\nclass UserClass:\n    pass\n")
    b_py.write_text("class TargetClass:\n    pass\n")

    grapher = DependencyGrapher(str(tmp_path))
    grapher.jinja_env = MagicMock()
    mock_template = MagicMock()
    mock_template.render.return_value = "Mocked HTML Content"
    grapher.jinja_env.get_template.return_value = mock_template

    with patch("logic.validation.dependency_grapher.Network") as mock_network_cls:
        mock_net = MagicMock()
        mock_net.show.side_effect = lambda f, notebook: Path(f).write_text("<html><body></body></html>")
        mock_network_cls.return_value = mock_net

        grapher.scan_project()
        assert len(grapher.all_files) == 2

        try:
            grapher.generate_graph(str(a_py), "TargetClass")
            assert mock_network_cls.called
            assert Path("dependency_graph.html").exists()
        finally:
            if Path("dependency_graph.html").exists():
                os.remove("dependency_graph.html")


def test_trace_dependencies_main(tmp_path):
    a_py = tmp_path / "a.py"
    a_py.write_text("class Target:\n    pass\n")

    with (
        patch("sys.argv", ["trace_dependencies.py", str(tmp_path), str(a_py), "Target"]),
        patch("logic.validation.trace_dependencies.DependencyGrapher.generate_graph") as mock_gen,
    ):
        main_trace()
        mock_gen.assert_called_once_with(os.path.abspath(str(a_py)), "Target")


def test_visualize_module_graph_helpers(tmp_path):
    root = tmp_path
    filepath = root / "logic" / "src" / "utils" / "helper.py"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("class Helper:\n    pass\n")

    assert file_to_module_visualize(filepath, root) == "logic.src.utils.helper"

    m_map = collect_module_map(root, set())
    assert "logic.src.utils.helper" in m_map

    known = {"a.b.c", "d.e"}
    assert resolve_to_module_visualize("c", 1, "a.b.d", known) == "a.b.c"
    assert resolve_to_module_visualize("d.e", 0, "a.b.d", known) == "d.e"

    layer, color = get_layer("logic.src.utils.helper", [("logic", "Logic", "#123")])
    assert layer == "Logic"
    assert color == "#123"

    layer_other, color_other = get_layer("unknown.module", [("logic", "Logic", "#123")])
    assert layer_other == "Other"


def test_visualize_module_graph_build_and_violations(tmp_path):
    root = tmp_path
    a = root / "logic" / "a.py"
    b = root / "gui" / "b.py"
    a.parent.mkdir(parents=True, exist_ok=True)
    b.parent.mkdir(parents=True, exist_ok=True)

    a.write_text("import gui.b\n")
    b.write_text("pass\n")

    graph = build_graph_visualize(root, set())
    assert "logic.a" in graph
    assert "gui.b" in graph

    pkg_graph, node_to_pkg = condense_to_packages(graph, 1)
    assert "logic" in pkg_graph
    assert pkg_graph["logic"] == {"gui"}

    layers = [("logic", "Logic", "#1"), ("gui", "GUI", "#2")]
    forbidden = [("Logic", "GUI")]
    violations = find_violations(graph, layers, forbidden)
    assert len(violations) == 1
    assert violations[0] == ("logic.a", "gui.b")


def test_visualize_module_graph_html(tmp_path):
    graph = {"logic.a": {"gui.b"}}
    layers = [("logic", "Logic", "#1"), ("gui", "GUI", "#2")]
    violation_edges = {("logic.a", "gui.b")}
    output = tmp_path / "module_graph.html"

    with patch("logic.validation.visualize_module_graph.Network") as mock_net_cls:
        mock_net = MagicMock()
        mock_net_cls.return_value = mock_net
        generate_html_visualize(graph, layers, violation_edges, output, depth=0)
        assert mock_net_cls.called


def test_visualize_module_graph_main(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    a = root / "a.py"
    a.write_text("pass\n")

    with patch("sys.argv", ["visualize_module_graph.py", str(root), "--no-html"]), pytest.raises(SystemExit) as exc:
        main_visualize()
    assert exc.value.code == 0
