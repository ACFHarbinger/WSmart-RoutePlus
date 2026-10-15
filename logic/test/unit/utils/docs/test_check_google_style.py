"""Unit tests for the Google Style Docstring Validator check_google_style.py."""

from unittest.mock import MagicMock, patch

import pytest
from logic.src.utils.docs.check_google_style import (
    GoogleStyleValidator,
    analyze_file,
    display_report,
    main,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



@pytest.fixture
def clean_module(tmp_path):
    """Creates a temporary Python module with perfect Google-style docstrings."""
    content = '''"""Clean Module.

Attributes:
    x (int): A dummy variable.

Example:
    >>> print("Hello")
"""

class CleanClass:
    """A clean class representation.

    Attributes:
        y (str): A class attribute.
    """

    def clean_method(self, param1: int, *, kwarg1: str = "val") -> int:
        """A clean method.

        Args:
            param1: The first parameter.
            kwarg1: Keyword argument.

        Returns:
            int: The return code.
        """
        return param1

    def clean_generator(self) -> float:
        """A clean generator function.

        Yields:
            float: Value between 0 and 1.
        """
        yield 0.5
'''
    f = tmp_path / "clean.py"
    f.write_text(content, encoding="utf-8")
    return str(f.resolve())


@pytest.fixture
def dirty_module(tmp_path):
    """Creates a temporary Python module with multiple Google docstring violations."""
    content = '''"""Dirty Module with missing sections."""

class DirtyClass:
    # Missing class docstring

    def missing_docstring(self, x):
        pass

    def missing_args_and_returns(self, x: int) -> int:
        """Just a single line docstring."""
        return x

    def restates_types(self, x: int) -> None:
        """Docstring.

        Args:
            x (int): Restates the hint.

        Returns:
            None
        """
'''
    f = tmp_path / "dirty.py"
    f.write_text(content, encoding="utf-8")
    return str(f.resolve())


@pytest.fixture
def syntax_error_module(tmp_path):
    """Creates a temporary Python module with invalid syntax."""
    content = "class InvalidClass\n    def error"
    f = tmp_path / "invalid.py"
    f.write_text(content, encoding="utf-8")
    return str(f.resolve())


@pytest.mark.unit
def test_clean_module_has_no_violations(clean_module):
    violations = analyze_file(clean_module)
    assert len(violations) == 0


@pytest.mark.unit
def test_dirty_module_violations(dirty_module):
    violations = analyze_file(dirty_module)
    messages = [v["message"] for v in violations]

    # Q-7b rule: public docstrings are required, sections are optional, Args must not restate
    # type hints and a Returns section that documents None is a defect.
    assert any("Missing class docstring" in m for m in messages)
    assert any("Missing function docstring" in m for m in messages)
    assert any("Args entry 'x' restates its type hint" in m for m in messages)
    assert any("'Returns:' section documents None" in m for m in messages)
    assert not any("Attributes" in m or "Example" in m for m in messages)


@pytest.mark.unit
def test_syntax_error_handling(syntax_error_module):
    violations = analyze_file(syntax_error_module)
    assert len(violations) == 1
    assert "Syntax error" in violations[0]["message"]


@pytest.mark.unit
def test_non_existent_file_handling():
    violations = analyze_file("non_existent_file.py")
    assert len(violations) == 1
    # Check that error is captured as a generic exception message
    assert violations[0]["context"] == "Error"


@pytest.mark.unit
def test_parse_sections_aliases():
    validator = GoogleStyleValidator("dummy.py")
    doc = """
    ARGS:
        x: dummy
    ARGUMENTS:
        y: dummy
    PARAMETERS:
        z: dummy
    PARAMS:
        w: dummy
    """
    sections = validator._parse_sections(doc)
    # They should all map to normalized "Args"
    assert "Args" in sections


@pytest.mark.unit
def test_display_report_empty():
    with pytest.raises(SystemExit) as exc:
        display_report([])
    assert exc.value.code == 0


@pytest.mark.unit
def test_display_report_with_violations():
    violations = [
        {"filepath": "dummy.py", "line": 10, "context": "Module", "name": "<module>", "message": "Missing attributes"}
    ]
    with pytest.raises(SystemExit) as exc:
        display_report(violations)
    assert exc.value.code == 1


@pytest.mark.unit
@patch("argparse.ArgumentParser.parse_args")
@patch("logic.src.utils.docs.check_google_style.display_report")
def test_main_scan_file(mock_display, mock_parse_args, clean_module):
    args = MagicMock()
    args.path = clean_module
    args.exclude_dir = []
    mock_parse_args.return_value = args

    main()
    mock_display.assert_called_once()
    violations = mock_display.call_args[0][0]
    assert len(violations) == 0


@pytest.mark.unit
@patch("argparse.ArgumentParser.parse_args")
@patch("logic.src.utils.docs.check_google_style.display_report")
def test_main_scan_directory(mock_display, mock_parse_args, tmp_path, clean_module):
    args = MagicMock()
    args.path = str(tmp_path.resolve())
    args.exclude_dir = []
    mock_parse_args.return_value = args

    # Also write a non-py file to ensure it's ignored
    txt_file = tmp_path / "ignore.txt"
    txt_file.write_text("Hello", encoding="utf-8")

    main()
    mock_display.assert_called_once()
