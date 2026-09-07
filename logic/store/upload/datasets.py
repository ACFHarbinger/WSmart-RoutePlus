"""
Upload dataset files from data/datasets and/or data/simulator to a cloud
drive (OneDrive/SharePoint, Google Drive or Dropbox).

Local directory structure is preserved under the destination folder, e.g.
data/simulator/datasets/riomaior100_emp_wsr30_N1_seed42.npz uploads to
<dest>/simulator/datasets/riomaior100_emp_wsr30_N1_seed42.npz.

Credentials per provider are documented in logic/store/config.py.

Usage
-----
    # All NPZ simulator datasets, to Google Drive
    uv run python -m logic.store.upload.datasets --provider gdrive \\
        --source simulator --pattern "*.npz"

    # Everything from both dataset roots, to OneDrive
    uv run python -m logic.store.upload.datasets --provider onedrive --source both
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from logic.store.providers import get_provider
from logic.store.transfer import collect_files, upload_tree

DATA_ROOT = Path("data")
SOURCES = {
    "datasets": DATA_ROOT / "datasets",
    "simulator": DATA_ROOT / "simulator",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", required=True, choices=["onedrive", "sharepoint", "gdrive", "dropbox"])
    p.add_argument("--source", default="both", choices=[*SOURCES, "both"],
                   help="Which dataset root(s) to upload from")
    p.add_argument("--pattern", default="*", help="Filename glob (e.g. '*.npz', '*.td')")
    p.add_argument("--dest", default="WSmart-Route/data", help="Destination folder on the drive")
    p.add_argument("--dry-run", action="store_true", help="List the selection without uploading")
    args = p.parse_args()

    sources = list(SOURCES) if args.source == "both" else [args.source]
    # dry runs never touch the network, so skip credential checks entirely
    provider = SimpleNamespace(name=args.provider) if args.dry_run else get_provider(args.provider)
    total = 0
    for src in sources:
        base = SOURCES[src]
        if not base.is_dir():
            print(f"  [WARN] Skipping missing directory: {base}")
            continue
        files = collect_files(base, args.pattern)
        if not files:
            print(f"  [WARN] No files matching '{args.pattern}' under {base}")
            continue
        print(f"{base}: {len(files)} files ({sum(f.stat().st_size for f in files) / 1e6:.1f} MB)")
        total += upload_tree(provider, files, base, f"{args.dest.strip('/')}/{src}", dry_run=args.dry_run)
    if not total:
        raise SystemExit("Nothing to upload.")
    print(f"Done — {total} file(s).")


if __name__ == "__main__":
    main()
