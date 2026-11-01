"""Unit tests for logic/docs/add_docstrings_batch.py."""

import tempfile
from pathlib import Path

import pytest
from logic.docs.add_docstrings_batch import DocstringInjector

pytestmark = [pytest.mark.unit, pytest.mark.fast]






@pytest.mark.unit
@pytest.mark.fast
def test_docstring_injector_generate_and_apply():
    sample_code = (
        "class Dummy:\n"
        "    def foo(self, a: int) -> str:\n"
        "        return 'bar'\n"
    )

    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
        tmp.write(sample_code)
        tmp_path = tmp.name

    try:
        injector = DocstringInjector(tmp_path)
        injector.scan_and_queue()
        assert len(injector.modifications) > 0

        injector.apply()
        new_content = "\n".join(injector.lines)
        assert "foo" in new_content
        assert "Args:" in new_content or "Returns:" in new_content
    finally:
        Path(tmp_path).unlink(missing_ok=True)
