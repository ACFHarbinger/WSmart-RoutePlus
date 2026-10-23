"""Reject output symlinks that escape a directory operation's destination."""

import pytest
from logic.src.utils.security import directories


@pytest.mark.parametrize("operation", ["encrypt", "decrypt"])
@pytest.mark.parametrize("link_kind", ["directory", "file"])
def test_output_symlink_cannot_escape_to_prefix_sibling(tmp_path, monkeypatch, make_symlink, operation, link_kind):
    source = tmp_path / "input"
    output = tmp_path / "out"
    outside = tmp_path / "outside"
    (source / "sub").mkdir(parents=True)
    output.mkdir()
    outside.mkdir()
    suffix = ".enc" if operation == "decrypt" else ""
    (source / "sub" / ("item" + suffix)).write_text("payload")
    target_name = "item.enc" if operation == "encrypt" else "item"
    target = outside / target_name
    target.write_text("untouched")
    if link_kind == "directory":
        make_symlink(output / "sub", outside, target_is_directory=True)
    else:
        (output / "sub").mkdir()
        make_symlink(output / "sub" / target_name, target)
    calls = []
    monkeypatch.setattr(directories, operation + "_file_data", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="outside"):
        getattr(directories, operation + "_directory")(b"unused", source, output)
    assert calls == []
    assert target.read_text() == "untouched"
