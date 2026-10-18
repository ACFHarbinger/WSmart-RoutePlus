"""Tests of the interface, embedded-language, multi-class and type-coverage checkers in ``logic/validation``."""

from unittest.mock import patch

import pytest
from logic.validation.check_interface_compliance import (
    base_name,
    build_registry,
    check_compliance,
    collect_python_files,
    get_implemented_methods,
    get_required_abstract_methods,
    has_decorator,
)
from logic.validation.check_interface_compliance import (
    main as main_compliance,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_compliance_base_name():
    import ast

    assert base_name(ast.Name(id="MyBase")) == "MyBase"
    assert base_name(ast.Attribute(attr="MyAttr")) == "MyAttr"  # pyrefly: ignore [missing-argument]
    assert base_name(ast.Constant(value=42)) == ""


def test_compliance_has_decorator():
    import ast

    func1 = ast.FunctionDef(name="foo", decorator_list=[ast.Name(id="abstractmethod")], body=[])  # pyrefly: ignore [missing-argument]
    func2 = ast.FunctionDef(name="bar", decorator_list=[ast.Attribute(attr="abstractmethod")], body=[])  # pyrefly: ignore [missing-argument]
    func3 = ast.FunctionDef(name="baz", decorator_list=[], body=[])  # pyrefly: ignore [missing-argument]
    assert has_decorator(func1, "abstractmethod") is True
    assert has_decorator(func2, "abstractmethod") is True
    assert has_decorator(func3, "abstractmethod") is False


def test_compliance_collect_python_files(tmp_path):
    root = tmp_path
    (root / "dir1").mkdir()
    (root / ".git").mkdir()
    (root / "dir1" / "a.py").write_text("pass")
    (root / "dir1" / "b.txt").write_text("pass")
    (root / ".git" / "c.py").write_text("pass")

    files = collect_python_files(root, set())
    assert len(files) == 1
    assert files[0].name == "a.py"


def test_compliance_flow(tmp_path):
    interface_py = tmp_path / "interface.py"
    interface_py.write_text("""
from abc import ABC, abstractmethod

class MyInterface(ABC):
    @abstractmethod
    def required_one(self):
        pass

    @abstractmethod
    def required_two(self):
        pass
""")

    concrete_py = tmp_path / "concrete.py"
    concrete_py.write_text("""
from interface import MyInterface

class CompliantClass(MyInterface):
    def required_one(self):
        return 1

    def required_two(self):
        return 2

class NonCompliantClass(MyInterface):
    def required_one(self):
        return 1
""")

    files = [interface_py, concrete_py]
    registry = build_registry(files)

    assert "MyInterface" in registry
    assert "CompliantClass" in registry
    assert "NonCompliantClass" in registry

    # Test get_required_abstract_methods
    reqs = get_required_abstract_methods("CompliantClass", registry)
    assert reqs == {"required_one", "required_two"}

    # Test get_implemented_methods
    impl_compliant = get_implemented_methods("CompliantClass", registry)
    assert "required_one" in impl_compliant
    assert "required_two" in impl_compliant

    impl_non_compliant = get_implemented_methods("NonCompliantClass", registry)
    assert "required_one" in impl_non_compliant
    assert "required_two" not in impl_non_compliant

    # Check compliance
    violations = check_compliance(registry)
    assert len(violations) == 1
    assert violations[0]["class"] == "NonCompliantClass"
    assert violations[0]["missing"] == ["required_two"]


@patch("sys.exit")
def test_compliance_main_success(mock_exit, tmp_path):
    mock_exit.side_effect = SystemExit
    interface_py = tmp_path / "interface.py"
    interface_py.write_text("""
from abc import ABC, abstractmethod
class MyInterface(ABC):
    @abstractmethod
    def run(self): pass
class Concrete(MyInterface):
    def run(self): pass
""")

    with patch("sys.argv", ["check_interface_compliance.py", str(tmp_path)]), pytest.raises(SystemExit):
        main_compliance()
    mock_exit.assert_called_once_with(0)


@patch("sys.exit")
def test_compliance_main_failure(mock_exit, tmp_path):
    mock_exit.side_effect = SystemExit
    interface_py = tmp_path / "interface.py"
    interface_py.write_text("""
from abc import ABC, abstractmethod
class MyInterface(ABC):
    @abstractmethod
    def run(self): pass
class Concrete(MyInterface):
    pass
""")

    with patch("sys.argv", ["check_interface_compliance.py", str(tmp_path)]), pytest.raises(SystemExit):
        main_compliance()
    mock_exit.assert_called_once_with(1)


@patch("sys.exit")
def test_compliance_main_non_existent_dir(mock_exit):
    mock_exit.side_effect = SystemExit
    with patch("sys.argv", ["check_interface_compliance.py", "non_existent_directory_abc"]), pytest.raises(SystemExit):
        main_compliance()
    mock_exit.assert_called_once_with(1)


def test_embedded_languages_detect():
    from logic.validation.check_embedded_languages import detect_language

    assert detect_language("") == ""
    assert detect_language("short") == ""
    assert detect_language("<html><body>Hello</body></html>") == "HTML"
    assert detect_language("const x = 10; console.log(x);") == "JavaScript"
    assert detect_language("body { background-color: #fff; }") == "CSS"
    assert detect_language("SELECT * FROM users WHERE id = 1;") == "SQL"


def test_embedded_languages_get_docstring_lines():
    import ast

    from logic.validation.check_embedded_languages import get_docstring_lines

    code = '''
def my_func():
    """This is a docstring
    spanning multiple lines.
    """
    x = 1
'''
    tree = ast.parse(code)
    doc_lines = get_docstring_lines(tree)
    # The docstring in my_func is on lines 3, 4, 5
    assert 3 in doc_lines
    assert 4 in doc_lines
    assert 5 in doc_lines
    assert 6 not in doc_lines


def test_embedded_languages_visitor():
    import ast

    from logic.validation.check_embedded_languages import EmbeddedCodeVisitor

    code = 'html_str = "<div>hello</div>"'
    tree = ast.parse(code)
    visitor = EmbeddedCodeVisitor(doc_lines=set())
    visitor.visit(tree)
    assert len(visitor.findings) == 1
    assert visitor.findings[0][1] == "HTML"

    # Test JoinedStr (f-string)
    code_f = 'html_str = f"const x = {var}; console.log(x);"'
    tree_f = ast.parse(code_f)
    visitor_f = EmbeddedCodeVisitor(doc_lines=set())
    visitor_f.visit(tree_f)
    assert len(visitor_f.findings) > 0
    assert any(f[1] == "JavaScript" for f in visitor_f.findings)


def test_embedded_languages_analyze_file(tmp_path):
    from logic.validation.check_embedded_languages import analyze_file

    py_file = tmp_path / "test_embed.py"
    py_file.write_text("""
def run():
    js = "console.log(123);"
    html = "<html><body></body></html>"
""")
    findings = analyze_file(py_file)
    assert len(findings) == 2
    langs = [f[1] for f in findings]
    assert "JavaScript" in langs
    assert "HTML" in langs

    # Test invalid syntax file
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("class 123Unbound:")
    assert analyze_file(bad_file) == []


@patch("sys.exit")
def test_embedded_languages_main(mock_exit, tmp_path):
    from logic.validation.check_embedded_languages import main as main_embedded

    # Create clean file
    py_file = tmp_path / "clean.py"
    py_file.write_text("def my_func():\n    return 1\n")

    with patch("sys.argv", ["check_embedded_languages.py", str(tmp_path)]):
        main_embedded()

    # Create unclean file
    py_file_unclean = tmp_path / "unclean.py"
    py_file_unclean.write_text('sql = "SELECT * FROM my_table;"')

    with patch("sys.argv", ["check_embedded_languages.py", str(tmp_path)]):
        main_embedded()

    # Non-existent dir
    with patch("sys.argv", ["check_embedded_languages.py", "non_existent_directory_abc"]):
        main_embedded()


def test_multi_classes_get_top_level_classes(tmp_path):
    from logic.validation.check_multi_classes import get_top_level_classes

    py_file = tmp_path / "classes.py"
    py_file.write_text("""
class A:
    pass

class _B:
    pass

def func():
    class Nested:
        pass
""")
    classes = get_top_level_classes(py_file)
    assert classes == ["A", "_B"]

    # Test syntax error
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("class 123Unbound:")
    assert get_top_level_classes(bad_file) == []


@patch("sys.exit")
def test_multi_classes_main(mock_exit, tmp_path):
    from logic.validation.check_multi_classes import main as main_multi_classes

    mock_exit.side_effect = SystemExit

    # 1. Clean run (at most 1 class)
    dir_clean = tmp_path / "clean"
    dir_clean.mkdir()
    py_file = dir_clean / "clean.py"
    py_file.write_text("class A:\n    pass\n")

    with patch("sys.argv", ["check_multi_classes.py", str(dir_clean)]), pytest.raises(SystemExit):
        main_multi_classes()
    mock_exit.assert_called_once_with(0)

    # 2. Violation run (multiple classes)
    dir_viol = tmp_path / "violation"
    dir_viol.mkdir()
    py_file_2 = dir_viol / "unclean.py"
    py_file_2.write_text("class A:\n    pass\nclass B:\n    pass\n")

    # Clear previous mock calls
    mock_exit.reset_mock()
    with patch("sys.argv", ["check_multi_classes.py", str(dir_viol)]), pytest.raises(SystemExit):
        main_multi_classes()
    mock_exit.assert_called_once_with(1)

    # 3. Violation run with --ignore-private
    dir_priv = tmp_path / "private"
    dir_priv.mkdir()
    py_file_3 = dir_priv / "unclean_private.py"
    py_file_3.write_text("class A:\n    pass\nclass _B:\n    pass\n")

    mock_exit.reset_mock()
    with patch("sys.argv", ["check_multi_classes.py", str(dir_priv), "--ignore-private"]), pytest.raises(SystemExit):
        main_multi_classes()
    mock_exit.assert_called_once_with(0)


def test_type_coverage_analyze_function():
    import ast

    from logic.validation.check_type_coverage import analyze_function

    # Full annotation
    node1 = ast.parse("def f(x: int) -> str: pass").body[0]
    assert analyze_function(node1) == (1, 1, True)  # pyrefly: ignore [bad-argument-type]

    # Partial annotation
    node2 = ast.parse("def f(x: int, y) -> str: pass").body[0]
    assert analyze_function(node2) == (1, 2, True)  # pyrefly: ignore [bad-argument-type]

    # None annotation
    node3 = ast.parse("def f(x, y): pass").body[0]
    assert analyze_function(node3) == (0, 2, False)  # pyrefly: ignore [bad-argument-type]

    # Skip self / cls
    node4 = ast.parse("def f(self, x: int) -> str: pass").body[0]
    assert analyze_function(node4) == (1, 1, True)  # pyrefly: ignore [bad-argument-type]

    # Complex args (vararg/kwarg)
    node5 = ast.parse("def f(*args: int, **kwargs: str) -> None: pass").body[0]
    assert analyze_function(node5) == (2, 2, True)  # pyrefly: ignore [bad-argument-type]


def test_type_coverage_analyze_file(tmp_path):
    from logic.validation.check_type_coverage import analyze_file

    py_file = tmp_path / "cov.py"
    py_file.write_text("""
def f1(x: int) -> str:
    pass

def f2(x):
    pass

def f3():
    pass
""")
    stats = analyze_file(py_file)
    assert stats["funcs"] == 3
    assert stats["full"] == 1
    assert stats["none"] == 2
    assert stats["partial"] == 0

    # Test syntax error
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("class 123Unbound:")
    assert analyze_file(bad_file) == {"funcs": 0, "full": 0, "partial": 0, "none": 0}


def test_type_coverage_markup():
    from logic.validation.check_type_coverage import _coverage_markup

    assert _coverage_markup(0, 0) == "[dim]—[/dim]"
    assert "green" in _coverage_markup(9, 10)
    assert "yellow" in _coverage_markup(7, 10)
    assert "red" in _coverage_markup(3, 10)


def test_type_coverage_main(tmp_path):
    from logic.validation.check_type_coverage import main as main_type_coverage

    # Create test files
    py_file1 = tmp_path / "test_cov1.py"
    py_file1.write_text("""
def f1(x: int) -> str: pass
def f2(x: int) -> str: pass
""")

    py_file2 = tmp_path / "test_cov2.py"
    py_file2.write_text("""
def f1(x): pass
def f2(x): pass
""")

    with patch("sys.argv", ["check_type_coverage.py", str(tmp_path), "--limit", "2", "--min-funcs", "2"]):
        main_type_coverage()
