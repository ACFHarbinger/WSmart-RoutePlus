# -- mode: python ; coding: utf-8 --
#
# WSmart-Route stripped-simulator executable spec.
#
# Produces a single-file executable ``WSmartRoute`` for the pruned codebase,
# which retains only the core simulation and training pipeline.  Modules that
# are always removed during export (CLI, eval, testing, HPO, plotting utils)
# are explicitly excluded here so PyInstaller does not attempt to bundle them.
#
# Supported entry points after pruning:
#   WSmartRoute train_lightning  model=am env.name=vrpp ...
#   WSmartRoute test_sim  --policies regular gurobi alns --days 31
#   WSmartRoute gen_data  virtual --problem vrpp --graph_sizes 50
#
# Build from the project root (after running the pruner):
#   pyinstaller ci/simulator.spec --clean
#
# PyInstaller resolves all imports relative to the CWD at build time (pathex=[.]).

import os
import sys
from glob import glob

# ── Optional heavy dependencies — skip gracefully if not installed ──────────
try:
    import wandb
    _wandb_path = os.path.dirname(wandb.__file__)
except ImportError:
    _wandb_path = None

try:
    import osmnx
    _osmnx_path = os.path.dirname(osmnx.__file__)
    try:
        _osmnx_meta = glob(os.path.join(os.path.dirname(_osmnx_path), "osmnx-*.dist-info"))[0]
    except IndexError:
        _osmnx_meta = None
except ImportError:
    _osmnx_path = None
    _osmnx_meta = None

try:
    import hexaly
    _hexaly_path = os.path.dirname(hexaly.__file__)
    _hexaly_lib = os.path.join(_hexaly_path, "libhexaly140.so")
except ImportError:
    _hexaly_path = None
    _hexaly_lib = None

# ── Binaries (Hexaly native lib) ────────────────────────────────────────────
_binaries = []
if _hexaly_lib and os.path.exists(_hexaly_lib):
    _binaries.append((_hexaly_lib, "."))

# ── Analysis ────────────────────────────────────────────────────────────────
a = Analysis(
    ["__main__.py"],
    pathex=["."],
    binaries=_binaries,
    datas=[
        # Simulation data and pre-trained weights
        ("data/simulator/*", "data/simulator"),
        ("assets/model_weights/*", "assets/model_weights"),
        ("assets/logs/*", "assets/logs"),
        # Hydra YAML configs (must mirror the installed layout)
        ("logic/configs", "logic/configs"),
    ],
    hiddenimports=[
        # ── Core Python / PyTorch ──────────────────────────────────────────
        "torch",
        "torch.nn",
        "torch.optim",
        "torch.cuda",
        "torch.multiprocessing",
        "torch.distributed",
        "numpy",
        "argparse",
        "multiprocessing",
        "multiprocessing.pool",
        "multiprocessing.managers",

        # ── PyTorch Lightning / training pipeline ─────────────────────────
        "pytorch_lightning",
        "pytorch_lightning.callbacks",
        "pytorch_lightning.loggers",
        "pytorch_lightning.trainer",
        "pytorch_lightning.utilities",
        "lightning",
        "lightning.pytorch",
        "lightning.pytorch.callbacks",
        "lightning.pytorch.loggers",

        # ── Hydra / OmegaConf ─────────────────────────────────────────────
        "hydra",
        "hydra._internal",
        "hydra._internal.conf.hydra_conf",
        "hydra._internal.conf.user_conf",
        "hydra.core.global_hydra",
        "hydra.core.plugins",
        "omegaconf",

        # ── Optimisation libraries ─────────────────────────────────────────
        "gurobipy",
        "pyvrp",
        "alns",
        "ortools",
        "ortools.constraint_solver.pywrapcp",
        "ortools.graph.pywrapgraph",
        "hexaly",

        # ── Graph / geometric ──────────────────────────────────────────────
        "torch_geometric",
        "torch_geometric.nn",
        "torch_geometric.data",
        "torch_geometric.utils",
        "networkx",

        # ── Data science ───────────────────────────────────────────────────
        "pandas",
        "scipy",
        "sklearn",
        "sklearn.preprocessing",

        # ── Logging & tracking ─────────────────────────────────────────────
        "wandb",
        "tensorboard",
        "tensorboardX",

        # ── Geo / mapping ──────────────────────────────────────────────────
        "osmnx",
        "shapely",
        "geopandas",

        # ── WSmart-Route entry points (ensures sub-commands are bundled) ───
        # NOTE: logic.src.cli and logic.src.pipeline.features.eval are always
        # removed during export and must NOT be listed here.
        "logic.src.pipeline.features.train",
        "logic.src.pipeline.features.test",
        "logic.src.pipeline.simulations.simulator",
        "logic.src.pipeline.rl",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Modules removed during export — always absent in the stripped codebase
        "logic.src.cli",
        "logic.src.cli.parser_dispatch",
        "logic.src.cli.ts_parser",
        "logic.src.pipeline.features.eval",
        # GUI toolkits — not needed in the CLI build
        "tkinter",
        "PySide6",
        "shiboken6",
        "PyQt5",
        "PyQt6",
        # Heavy test suites
        "pytest",
        "hypothesis",
        "test",
        "unittest",
        "matplotlib.tests",
        "numpy.tests",
        "scipy.tests",
        "pandas.tests",
        "torch.test",
        "PIL.tests",
        # Unused audio/video
        "torchaudio",
        "torchvision",
        # Database drivers not in use
        "pysqlite2",
        "MySQLdb",
        "psycopg2",
        # ONNX internals
        "onnxscript",
        "torch.onnx._internal.fx.passes",
        "expecttest",
        "importlib_resources.trees",
    ],
    noarchive=False,
    optimize=0,
    cipher=None,
    key=None,
    collect_all=[
        # Logic layer — collect everything so dynamic imports work after pruning.
        # logic.src.cli is excluded: it is always removed during export.
        "logic.src.policies",
        "logic.src.models",
        "logic.src.pipeline",
        "logic.src.utils",
        "logic.src.interfaces",
        "logic.src.envs",
        "logic.src.data",
        "logic.src.configs",
        "logic.src.constants",
    ],
    collect_submodules=[],
    collect_data=[],
    collect_entrypoints=[],
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_clean_pkgname=[],
)

# ── Bundle large data trees (wandb, osmnx) ──────────────────────────────────
if _wandb_path:
    a.datas += Tree(_wandb_path, prefix="wandb")

if _osmnx_path:
    a.datas += Tree(_osmnx_path, prefix="osmnx")
    if _osmnx_meta:
        a.datas += Tree(_osmnx_meta, prefix=os.path.basename(_osmnx_meta))

# ── PYZ archive ─────────────────────────────────────────────────────────────
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ── Single-file executable ───────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="WSmartRoute",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ── One-dir collection (for systems that prefer an unpacked layout) ──────────
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="wsmart_route",
)
