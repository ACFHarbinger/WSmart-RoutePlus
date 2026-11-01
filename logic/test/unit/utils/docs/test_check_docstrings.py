

from unittest.mock import mock_open, patch

import pytest
from logic.docs.check_docstrings import check_docstrings_recursive, check_path

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestCheckDocstrings:
    @patch("builtins.open", new_callable=mock_open, read_data="def a():\n    pass")
    @patch("os.path.isfile", return_value=True)
    def test_check_path_missing(self, mock_isfile, m_open):
        missing = check_path("dummy.py")

        assert len(missing) == 2
        assert missing[0]["type"] == "Module"
        assert missing[1]["type"] == "Function"

    @patch("builtins.open", new_callable=mock_open, read_data='"""Module."""\ndef a():\n    """Doc."""\n    pass')
    @patch("os.path.isfile", return_value=True)
    def test_check_path_present(self, mock_isfile, m_open):
        missing = check_path("dummy.py")
        assert len(missing) == 0

    @patch("builtins.open", new_callable=mock_open, read_data='"""Module."""\nclass A:\n    pass')
    @patch("os.path.isfile", return_value=True)
    def test_check_path_class_missing(self, mock_isfile, m_open):
        missing = check_path("dummy.py")
        assert len(missing) == 1
        assert missing[0]["type"] == "Class"

    @patch("builtins.open", new_callable=mock_open, read_data="invalid syntax @#$")
    @patch("os.path.isfile", return_value=True)
    @patch("logic.docs.check_docstrings.console.print")
    def test_check_path_syntax_error(self, mock_print, mock_isfile, m_open):
        missing = check_path("dummy.py")
        assert len(missing) == 0
        mock_print.assert_called()

    def test_check_path_ignore_non_py(self):
        missing = check_path("dummy.txt")
        assert len(missing) == 0

    @patch("os.walk")
    @patch("logic.docs.check_docstrings.check_path")
    def test_recursive(self, mock_check, mock_walk):
        mock_walk.return_value = [("root", [], ["a.py", "b.py"])]
        mock_check.return_value = ["error"]

        missing = check_docstrings_recursive("root")
        assert len(missing) == 2
        assert mock_check.call_count == 2
