#!/bin/bash
set -e

# Ensure mutants directory exists and is clean-ish (we don't want to delete it if it has mutation history, but we need fresh source)
# mutmut generates mutants in 'mutants' directory.
# We need to ensure the FULL source code is there so pytest can run.

echo "Preparing mutants environment..."
mkdir -p mutants

# Copy logic directory to mutants/logic
# We use rsync to update existing files but not delete mutants/mutmut stuff if possible?
# Actually mutmut relies on file timestamps to know if it needs to regenerate.
# If we overwrite `boolmask.py` with original, mutmut will see it changed?
# mutmut `copy_src_dir` runs at start of `run`.
# But `mutmut` checks hashes.

# Simplest approach: Sync logic to mutants/logic
# We exclude __pycache__
rsync -a --exclude '__pycache__' logic mutants/

echo "Running mutmut..."
uv run mutmut run "$@"
