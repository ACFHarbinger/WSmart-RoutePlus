# -*- mode: python ; coding: utf-8 -*-
#
# WSmart-Route 1.0.0 release executable (PyInstaller; one-dir by default, one-file with WSR_ONEFILE=1).
#
# Build spec of the release tree that logic/package/build_release.py writes. That tree does not carry
# ci/: copy this file (as ci/simulator.spec) and ci/pyi_rth_builtins_help.py into it, then build from
# its root with an environment that has the release and its solvers and neural extras plus PyInstaller:
#
#   uv venv --python 3.12 .venv && uv pip install ".[solvers,neural]" pyinstaller==6.19.0
#   WSR_ONEFILE=1 pyinstaller ci/simulator.spec --clean --noconfirm   # single file: dist/WSmartRoute
#
# Bundles the released routing algorithms (SWC-TCF, BPC, ALNS, SANS, HGS, PSOMA, ACO-HH, PG-CLNS, AM,
# CVRP), packing, improvers and acceptance criteria, the simulator, and the AM training stack.
# Not bundled: Pyomo (SWC-TCF uses Gurobi by default), data/, model_weights/. Usage, from a folder that
# contains data/simulator (or with WSMART_ROUTE_ROOT pointing at one):
#
#   /path/to/WSmartRoute test_sim  [hydra overrides...]
#   /path/to/WSmartRoute train     [hydra overrides...]
#
# Outputs (outputs/, logs/) are written to that same working root
# (see logic/src/constants/paths.py::_frozen_root).

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

# Paths are resolved relative to this spec's folder (ci/); anchor them at the repo root.
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))  # noqa: F821 (SPECPATH is set by PyInstaller)
sys.path.insert(0, ROOT)  # make `logic` importable for collect_submodules

# Policies, models and configs are resolved through string-keyed registries and
# Hydra _target_ strings, so collect every logic module explicitly.
hiddenimports = collect_submodules("logic") + collect_submodules("packages")
# SciPy's vendored array-API layer imports its namespaces dynamically (e.g. ...numpy.fft).
hiddenimports += collect_submodules("scipy._external")
for pkg in ("hydra", "omegaconf", "tensordict", "torchrl", "pytorch_lightning", "lightning_fabric"):
    hiddenimports += collect_submodules(pkg)
hiddenimports += ["shapely.geometry", "openpyxl", "jinja2"]

# Compiled solver packages load submodules and native libraries dynamically
# (e.g. gurobipy._attrutil); bundle them whole.
binaries = []
_solver_datas = []
for pkg in ("gurobipy", "ortools", "fast_tsp", "pyvrp"):
    _d, _b, _h = collect_all(pkg)
    _solver_datas += _d
    binaries += _b
    hiddenimports += _h

datas = [
    (os.path.join(ROOT, "logic/configs"), "logic/configs"),
    (os.path.join(ROOT, "logic/release.yaml"), "logic"),
    (os.path.join(ROOT, "logic/src/tracking/logging/modules/popup.html"), "logic/src/tracking/logging/modules"),
    (os.path.join(ROOT, "logic/src/tracking/core/schema.sql"), "logic/src/tracking/core"),
    (os.path.join(ROOT, "logic/src/tracking/core/queries.json"), "logic/src/tracking/core"),
    (os.path.join(ROOT, "logic/src/tracking/database/sql"), "logic/src/tracking/database/sql"),
]
datas += _solver_datas
# Hydra ships its own config files; Lightning reads version files at import.
for pkg in ("hydra", "pytorch_lightning", "lightning_fabric", "lightning_utilities"):
    datas += collect_data_files(pkg)
# Distribution metadata queried through importlib.metadata at runtime.
for dist in (
    "torch", "tensordict", "torchrl", "pytorch-lightning", "lightning-fabric", "lightning-utilities",
    "hydra-core", "omegaconf", "numpy", "pandas", "scipy", "tqdm", "rich", "loguru", "networkx",
    "gurobipy", "shapely", "jinja2", "openpyxl", "protobuf", "packaging",
    "ortools", "pyvrp", "alns", "fast-tsp", "matplotlib", "plotext", "seaborn",
):
    try:
        datas += copy_metadata(dist)
    except Exception:  # not installed in the build environment
        pass

a = Analysis(
    [os.path.join(ROOT, "__main__.py")],
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
        # test suites
        "pytest", "hypothesis", "expecttest",
        # unused torch companions
        "torchaudio", "torchvision",
        # solver backends this export does not ship
        "pyomo",
        # optional extras of torchrl/Lightning that a development environment may hold; a partial
        # copy breaks start-up (torchrl imports ray when it finds it)
        "ray", "wandb", "mlflow", "zenml", "optuna",
    ],
    noarchive=False,
    optimize=0,
    # Lightning reads its version only if __version__.py exists on disk next to
    # __init__; keep these packages' sources on disk as well as in the archive.
    # torch_geometric compiles TorchScript at import (needs its .py sources); lightning inspects source too.
    module_collection_mode={
        "pytorch_lightning": "pyz+py",
        "lightning_fabric": "pyz+py",
        "lightning": "pyz+py",
        "torch_geometric": "pyz+py",
    },
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
    # one-file: everything is packed into the launcher and unpacked at start-up
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], runtime_tmpdir=None, **_common)
else:
    # one-dir: binaries go to _internal/ via COLLECT
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **_common)
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, upx_exclude=[], name="wsmart_route")
