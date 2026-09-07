"""
Download dataset files from a cloud drive (OneDrive/SharePoint, Google Drive
or Dropbox) into data/datasets and/or data/simulator.

Mirrors logic.store.upload.datasets: it expects the remote folder to contain
'datasets/' and/or 'simulator/' subfolders (as produced by the uploader)
and re-creates the remote structure locally.

Credentials per provider are documented in logic/store/config.py.

Usage
-----
    # Fetch all NPZ simulator datasets from Google Drive
    uv run python -m logic.store.download.datasets --provider gdrive \\
        --source simulator --pattern "*.npz"

    # Fetch everything from both dataset roots on OneDrive
    uv run python -m logic.store.download.datasets --provider onedrive --source both
"""

from __future__ import annotations

import argparse
from pathlib import Path

from logic.store.providers import get_provider
from logic.store.transfer import download_tree

DATA_ROOT = Path("data")
SOURCES = ("datasets", "simulator")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", required=True, choices=["onedrive", "sharepoint", "gdrive", "dropbox"])
    p.add_argument("--source", default="both", choices=[*SOURCES, "both"],
                   help="Which dataset root(s) to download into")
    p.add_argument("--pattern", default="*", help="Filename glob (e.g. '*.npz', '*.td')")
    p.add_argument("--remote", default="WSmart-Route/data",
                   help="Remote folder holding the dataset roots")
    p.add_argument("--target", default=str(DATA_ROOT),
                   help="Local directory to download into (default: data/)")
    p.add_argument("--dry-run", action="store_true", help="List the selection without downloading")
    args = p.parse_args()

    sources = list(SOURCES) if args.source == "both" else [args.source]
    provider = get_provider(args.provider)
    target = Path(args.target)
    total = 0
    for src in sources:
        remote_dir = f"{args.remote.strip('/')}/{src}"
        print(f"Fetching {provider.name}:{remote_dir} → {target / src}")
        try:
            total += download_tree(provider, remote_dir, target / src, args.pattern, dry_run=args.dry_run)
        except FileNotFoundError as exc:
            print(f"  [WARN] {exc}")
        except Exception as exc:  # surface provider 404s without a stack trace
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                print(f"  [WARN] Remote folder not found: {remote_dir}")
            else:
                raise
    if not total:
        raise SystemExit("Nothing downloaded — check --remote, --source and --pattern.")
    print(f"Done — {total} file(s).")


if __name__ == "__main__":
    main()
