#!/usr/bin/env bash
# WSmart+ Route package setup for Linux.
#
#   bash scripts/linux/setup.sh
#
# 1. installs uv if it is missing;
# 2. unpacks the code archive (code.zip or wsmart_route_source.zip) when logic/ is not there yet;
# 3. creates the virtual environment (.venv), or updates it if it already exists;
# 4. unpacks data.zip into data/;
# 5. makes the Linux executable runnable.
#
# The package root is the nearest parent folder that holds pyproject.toml. This script is written for
# the exported package (the main repository keeps a copy under desktop/linux/scripts; do not run it
# against a development checkout, where it would drop the development dependencies from .venv).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ ! -f "$ROOT/pyproject.toml" ]; do
  [ "$ROOT" = "/" ] && { echo "[setup] no pyproject.toml above $(dirname "${BASH_SOURCE[0]}")" >&2; exit 1; }
  ROOT="$(dirname "$ROOT")"
done
cd "$ROOT"
echo "[setup] package root: $ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found: installing it"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "[setup] curl or wget is needed to install uv" >&2; exit 1
  fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo "[setup] $(uv --version)"

# unzip <archive> <folder the archive's content belongs in, . for the package root>
unpack() { uv run --no-project --quiet python -c "import sys,zipfile,pathlib;r=pathlib.Path(sys.argv[1]);s=sys.argv[3].strip('.');z=zipfile.ZipFile(r/sys.argv[2]);n=[x for x in z.namelist() if not x.startswith('__MACOSX/')];t=r if (not s or all(x.startswith(s+'/') for x in n)) else r/s;t.mkdir(parents=True,exist_ok=True);z.extractall(t,n);print('[setup]',sys.argv[2],'unpacked:',len(n),'entries ->',t)" "$ROOT" "$1" "$2"; }

if [ ! -f logic/pyproject.toml ]; then
  for archive in code.zip wsmart_route_source.zip; do
    if [ -f "$archive" ]; then unpack "$archive" .; break; fi
  done
  [ -f logic/pyproject.toml ] || { echo "[setup] logic/ is missing and no code archive was found" >&2; exit 1; }
fi

# Creates .venv on the first run and brings it up to date afterwards.
uv sync --package wsmart-route --extra solvers --no-dev

if [ -f data.zip ]; then
  unpack data.zip data
else
  echo "[setup] no data.zip: skipping (place the runtime data in $ROOT/data yourself)"
fi

[ -f executables/WSmartRoute ] && chmod +x executables/WSmartRoute

echo "[setup] done. Activate the environment with:  source .venv/bin/activate"
