"""Build (or refresh) an export profile from two commits of an export branch.

``base`` is the export tree before the profile's changes and ``target`` the tree after them,
with every policy engine still present (engine selection is applied separately by
apply_export_profile.py). Files deleted between the two become ``remove_paths`` (a directory is
listed when it disappears completely; ``main.py`` and ``__main__.py`` are left to the
``entrypoints`` option); added and modified files become the profile patch. ``moves`` (from
``--move FROM:TO`` or the existing profile) are applied to ``base`` first, so moved files are
not removed and re-added. The profile entry in
``ci/export_config.json`` is created or updated.

Usage (from the main repository)::

    python logic/package/make_export_profile.py --repo ../wsr-package/wt --base <commit> --target <commit> \\
        --name minimal-ptp --entrypoints dunder --engines "swc_tcf=gurobi" --description "..."

Check the result by applying it to ``base`` and diffing against the package::

    git -C ../wsr-package/wt worktree add --detach /tmp/check <base>
    python logic/package/apply_export_profile.py --root /tmp/check --profile minimal-ptp
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = {"main.py", "__main__.py"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def moved_tree(repo: Path, base: str, moves: List[Dict[str, str]]) -> str:
    """Return a tree id for ``base`` with ``moves`` applied (same semantics as apply_moves)."""
    if not moves:
        return base
    entries: Dict[str, str] = {}
    for rec in filter(None, _git(repo, "ls-tree", "-r", "-z", base).split("\0")):
        meta, path = rec.split("\t", 1)
        entries[path] = meta
    for move in moves:
        src, dst = move["from"].rstrip("/") + "/", move["to"].rstrip("/")
        dst = dst + "/" if dst else ""
        children = {p[len(src):].split("/", 1)[0] for p in entries if p.startswith(src)}
        for child in children:
            for p in [p for p in entries if p == dst + child or p.startswith(dst + child + "/")]:
                del entries[p]
        for p in [p for p in entries if p.startswith(src)]:
            entries[dst + p[len(src):]] = entries.pop(p)
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": os.path.join(tmp, "index")}
        info = "".join(f"{meta}\t{path}\0" for path, meta in entries.items())
        subprocess.run(["git", "-C", str(repo), "update-index", "-z", "--index-info"], input=info, text=True,
                       env=env, check=True)
        return subprocess.run(["git", "-C", str(repo), "write-tree"], env=env, capture_output=True, text=True,
                              check=True).stdout.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, required=True, help="Export checkout holding both commits.")
    p.add_argument("--base", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--entrypoints", default="both", choices=["both", "main", "dunder", "none"])
    p.add_argument("--engines", default="", help="Profile engine selection, e.g. 'swc_tcf=gurobi'.")
    p.add_argument("--description", default="")
    p.add_argument("--move", action="append", default=None, metavar="FROM:TO",
                   help="Move FROM's contents into TO before diffing (repeatable; default: the profile's moves).")
    p.add_argument("--config", type=Path, default=_ROOT / "ci" / "export_config.json")
    args = p.parse_args()

    config = json.loads(args.config.read_text())
    old = config.get("export_profiles", {}).get(args.name, {})
    if args.move is None:
        moves = old.get("moves", [])
    else:
        moves = [{"from": m.partition(":")[0], "to": m.partition(":")[2]} for m in args.move]
    base = moved_tree(args.repo, args.base, moves)

    base_files = set(filter(None, _git(args.repo, "ls-tree", "-r", "-z", "--name-only", base).split("\0")))
    target_files = set(filter(None, _git(args.repo, "ls-tree", "-r", "-z", "--name-only", args.target).split("\0")))
    removed = set()
    for f in sorted(base_files - target_files):
        if f in _ENTRY:
            continue
        parts = f.split("/")
        entry = f
        for i in range(1, len(parts)):
            d = "/".join(parts[:i])
            if not any(t.startswith(d + "/") for t in target_files):
                entry = d
                break
        removed.add(entry)

    patch = _git(args.repo, "diff", "--no-renames", "--binary", "--diff-filter=AM", base, args.target)
    patch_rel = f"export_profiles/{args.name}.patch"
    (args.config.parent / patch_rel).parent.mkdir(parents=True, exist_ok=True)
    (args.config.parent / patch_rel).write_text(patch)

    profiles = config.setdefault("export_profiles", {})
    # Keys this script does not derive (e.g. ``include``) are kept from the existing entry.
    profiles[args.name] = {
        **old,
        "description": args.description or old.get("description", ""),
        "entrypoints": args.entrypoints,
        "engines": args.engines,
        **({"moves": moves} if moves else {}),
        "remove_paths": sorted(removed),
        "patch": patch_rel,
    }
    args.config.write_text(json.dumps(config, indent=2) + "\n")
    print(f"{args.name}: {len(removed)} remove paths, {patch.count(chr(10) + 'diff --git') + bool(patch)} patched files")


if __name__ == "__main__":
    main()
