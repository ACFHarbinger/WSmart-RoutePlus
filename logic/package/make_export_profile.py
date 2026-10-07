"""Build (or refresh) an export profile from two commits of an export branch.

``base`` is the export tree before the profile's changes and ``target`` the tree after them,
with every policy engine still present (engine selection is applied separately by
apply_export_profile.py). Files deleted between the two become ``remove_paths`` (a directory is
listed when it disappears completely; ``main.py`` and ``__main__.py`` are left to the
``entrypoints`` option); added and modified files become the profile patch. The profile entry in
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
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = {"main.py", "__main__.py"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, required=True, help="Export checkout holding both commits.")
    p.add_argument("--base", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--entrypoints", default="both", choices=["both", "main", "dunder", "none"])
    p.add_argument("--engines", default="", help="Profile engine selection, e.g. 'swc_tcf=gurobi'.")
    p.add_argument("--description", default="")
    p.add_argument("--config", type=Path, default=_ROOT / "ci" / "export_config.json")
    args = p.parse_args()

    base_files = set(_git(args.repo, "ls-tree", "-r", "--name-only", args.base).split())
    target_files = set(_git(args.repo, "ls-tree", "-r", "--name-only", args.target).split())
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

    patch = _git(args.repo, "diff", "--no-renames", "--binary", "--diff-filter=AM", args.base, args.target)
    patch_rel = f"export_profiles/{args.name}.patch"
    (args.config.parent / patch_rel).parent.mkdir(parents=True, exist_ok=True)
    (args.config.parent / patch_rel).write_text(patch)

    config = json.loads(args.config.read_text())
    profiles = config.setdefault("export_profiles", {})
    old = profiles.get(args.name, {})
    # Keys this script does not derive (e.g. ``include``) are kept from the existing entry.
    profiles[args.name] = {
        **old,
        "description": args.description or old.get("description", ""),
        "entrypoints": args.entrypoints,
        "engines": args.engines,
        "remove_paths": sorted(removed),
        "patch": patch_rel,
    }
    args.config.write_text(json.dumps(config, indent=2) + "\n")
    print(f"{args.name}: {len(removed)} remove paths, {patch.count(chr(10) + 'diff --git') + bool(patch)} patched files")


if __name__ == "__main__":
    main()
