#!/usr/bin/env bash
# Build WSmartRoute.exe (single file, CUDA torch) for the minimal-ptp export with Windows Python under Wine.
#
#   ci/build_windows_exe.sh <package dir> [out dir]
#
# <package dir> is an export tree (__main__.py + logic/) that also holds ci/simulator.spec (a copy of
# ci/simulator_minimal.spec) and ci/pyi_rth_builtins_help.py. Needs Wine and a Windows Python 3.10 at
# C:\Python310 in $WINEPREFIX. The same three pip/PyInstaller commands work in a real Windows shell.
#
# GPU support comes from the CUDA build of torch: it is installed first from the PyTorch index so the
# package's torch pin does not pull the CPU-only wheel from PyPI. The .exe then uses the GPU on any
# Windows machine with a recent NVIDIA driver (no CUDA toolkit needed) and falls back to CPU otherwise.
set -euo pipefail
PKG="$(cd "${1:?package dir}" && pwd)"
OUT="$(mkdir -p "${2:-$PKG/dist}" && cd "${2:-$PKG/dist}" && pwd)"
: "${WINEPREFIX:?set WINEPREFIX to the Wine prefix that holds C:\\Python310}"
export WINEPREFIX WINEDEBUG="${WINEDEBUG:--all}"
TORCH="${TORCH:-2.13.0}"       # must match the pin in logic/pyproject.toml
CUDA_TAG="${CUDA_TAG:-cu130}"  # PyTorch wheel index: https://download.pytorch.org/whl/<tag>
py() { wine 'C:\Python310\python.exe' "$@"; }

cd "$PKG"
py -m pip install --upgrade pip
py -m pip install "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
# Only the dependencies are installed: PyInstaller reads logic/ straight from the package folder.
python3 - > build_requirements.txt <<'PYEOF'
import re
text = open("logic/pyproject.toml").read()
deps = re.search(r"^dependencies = \[(.*?)^\]", text, re.S | re.M).group(1)
solvers = re.search(r"^solvers = \[(.*?)^\]", text, re.S | re.M).group(1)
print("\n".join(re.findall(r'"([^"]+)"', deps + solvers)))
PYEOF
py -m pip install -r build_requirements.txt pyinstaller==6.19.0
# Stop before the long build if the CUDA build is not the one installed.
py -c "import torch; assert torch.version.cuda, 'CPU-only torch installed'; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"
WSR_ONEFILE=1 py -m PyInstaller ci/simulator.spec --clean --noconfirm --distpath "$(winepath -w "$OUT")" --workpath "$(winepath -w "$PKG/build")"
ls -la "$OUT/WSmartRoute.exe"
