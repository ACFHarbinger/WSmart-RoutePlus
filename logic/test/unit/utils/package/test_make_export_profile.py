"""make_export_profile.moved_tree: moves applied to a commit's tree, as apply_moves does on disk."""

import subprocess

import pytest
from logic.package.make_export_profile import moved_tree

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def test_moved_tree_replaces_parent_entries(tmp_path):
    (tmp_path / "pkg" / "export").mkdir(parents=True)
    (tmp_path / "pkg" / "__init__.py").write_text("old\n")
    (tmp_path / "pkg" / "keep.py").write_text("keep\n")
    (tmp_path / "pkg" / "export" / "__init__.py").write_text("new\n")
    (tmp_path / "pkg" / "export" / "grid.py").write_text("grid\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    tree = moved_tree(tmp_path, "HEAD", [{"from": "pkg/export", "to": "pkg"}])
    assert _git(tmp_path, "ls-tree", "-r", "--name-only", tree).split() == ["pkg/__init__.py", "pkg/grid.py", "pkg/keep.py"]
    assert _git(tmp_path, "show", f"{tree}:pkg/__init__.py") == "new\n"
    assert moved_tree(tmp_path, "HEAD", []) == "HEAD"
