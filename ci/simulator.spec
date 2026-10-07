# -*- mode: python ; coding: utf-8 -*-
#
# WSmart-Route minimal-export executable (PyInstaller one-dir build).
#
# Bundles the training / evaluation pipeline (Attention Model + REINFORCE), the
# nine simulator policies and the simulator. Build from the repository root with
# an environment that has the core + solvers dependencies and PyInstaller:
#
#   uv sync --frozen --package wsmart-route --extra solvers --no-dev
#   uv pip install pyinstaller==6.19.0
#   pyinstaller ci/simulator.spec --clean --noconfirm
#
# Output: dist/wsmart_route/WSmartRoute (launcher) + dist/wsmart_route/_internal/.
# Ship the whole dist/wsmart_route/ folder. Usage, from a folder that contains
# data/simulator (or with WSMART_ROUTE_ROOT pointing at one):
#
#   /path/to/wsmart_route/WSmartRoute gen_data  [hydra overrides...]
#   /path/to/wsmart_route/WSmartRoute train     [hydra overrides...]
#   /path/to/wsmart_route/WSmartRoute eval      [hydra overrides...]
#   /path/to/wsmart_route/WSmartRoute test_sim  [hydra overrides...]
#
# data/simulator is not bundled; outputs (assets/, logs, outputs/) are written to
# that same working root (see logic/src/constants/paths.py::_frozen_root).

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

# Paths are resolved relative to this spec's folder (ci/); anchor them at the repo root.
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))  # noqa: F821 (SPECPATH is set by PyInstaller)
sys.path.insert(0, ROOT)  # make `logic` importable for collect_submodules

# Policies, models and configs are resolved through string-keyed registries and
# Hydra _target_ strings, so collect every logic module explicitly.
hiddenimports = collect_submodules("logic")
for pkg in ("hydra", "omegaconf", "tensordict", "torchrl", "pytorch_lightning", "lightning_fabric", "pyomo"):
    hiddenimports += collect_submodules(pkg)
hiddenimports += ["shapely.geometry", "openpyxl", "jinja2"]

# Compiled solver packages load submodules and native libraries dynamically
# (e.g. gurobipy._attrutil); bundle them whole.
binaries = []
_solver_datas = []
for pkg in ("gurobipy", "ortools", "fast_tsp"):
    _d, _b, _h = collect_all(pkg)
    _solver_datas += _d
    binaries += _b
    hiddenimports += _h

datas = [
    (os.path.join(ROOT, "logic/configs"), "logic/configs"),
    (os.path.join(ROOT, "logic/src/tracking/logging/modules/popup.html"), "logic/src/tracking/logging/modules"),
    (os.path.join(ROOT, "logic/src/pipeline/simulations/wsmart_bin_analysis/LICENSE"), "logic/src/pipeline/simulations/wsmart_bin_analysis"),
]
datas += _solver_datas
# Hydra ships its own config files; Lightning reads version files at import.
for pkg in ("hydra", "pytorch_lightning", "lightning_fabric", "lightning_utilities"):
    datas += collect_data_files(pkg)
# Distribution metadata queried through importlib.metadata at runtime.
for dist in (
    "torch", "tensordict", "torchrl", "pytorch-lightning", "lightning-fabric", "lightning-utilities",
    "hydra-core", "omegaconf", "numpy", "pandas", "scipy", "tqdm", "rich", "loguru", "networkx",
    "gurobipy", "ortools", "fast-tsp", "pyomo", "shapely", "jinja2", "openpyxl", "protobuf", "packaging",
):
    try:
        datas += copy_metadata(dist)
    except Exception:  # not installed in the build environment
        pass

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(ROOT, "ci", "pyi_rth_builtins_help.py")],
    excludes=[
        # GUI toolkits and interactive tooling are not used by the CLI build
        "tkinter", "PySide6", "shiboken6", "PyQt5", "PyQt6", "IPython", "jupyter", "notebook",
        # plotting is optional in the export (vendored GridBase falls back without it)
        "matplotlib",
        # test suites
        "pytest", "hypothesis", "expecttest",
        # unused torch companions
        "torchaudio", "torchvision",
    ],
    noarchive=False,
    optimize=0,
    # Lightning reads its version only if __version__.py exists on disk next to
    # __init__; keep these packages' sources on disk as well as in the archive.
    module_collection_mode={"pytorch_lightning": "pyz+py", "lightning_fabric": "pyz+py"},
)

pyz = PYZ(a.pure)

ONEFILE = os.environ.get("WSR_ONEFILE", "") not in ("", "0")
_common = dict(
    name="WSmartRoute",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-compressing CUDA libraries corrupts them
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if ONEFILE:
    # one-file (WSR_ONEFILE=1): everything is packed into the launcher and unpacked at start-up
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], runtime_tmpdir=None, **_common)
else:
    # one-dir: binaries go to _internal/ via COLLECT
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **_common)
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, upx_exclude=[], name="wsmart_route")
