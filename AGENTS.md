# AGENTS.md - Instructions for Coding Assistant LLMs

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Gurobi](https://img.shields.io/badge/Gurobi-11.0-ED1C24?logo=gurobi&logoColor=white)](https://www.gurobi.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MyPy](https://img.shields.io/badge/MyPy-checked-2f4f4f.svg)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/pytest-testing-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-50%25-green.svg)](https://coverage.readthedocs.io/)
[![CI](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml/badge.svg)](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml)

> **Version**: 3.0
> **Last Updated**: January 2026
> **Purpose**: Comprehensive guide for AI coding assistants working on WSmart+ Route

This document serves as the authoritative reference for AI assistants (Claude, GPT, Copilot, etc.) working on the WSmart+ Route codebase. It provides context about the project mission, architecture, coding standards, and operational guidelines.

---

## Table of Contents

1. [Project Overview & Mission](#1-project-overview--mission)
2. [Technical Stack & Environmental Governance](#2-technical-stack--environmental-governance)
3. [Core Architectural Boundaries](#3-core-architectural-boundaries)
4. [Key CLI Entry Points](#4-key-cli-entry-points-operational-playbook)
5. [External Access and Browser Usage Rules](#5-external-access-and-browser-usage-rules)
6. [Domain-Specific Coding Standards](#6-domain-specific-coding-standards)
7. [AI Review & Severity Protocol](#7-ai-review--severity-protocol)
8. [Known Constraints & "No-Go" Areas](#8-known-constraints--no-go-areas)
9. [Usage Note](#9-usage-note)
10. [Logic Submodule Architecture](#10-logic-submodule-architecture)
11. [Studio Submodule Architecture](#11-studio-submodule-architecture)
12. [Data Formats & Schemas](#12-data-formats--schemas)
13. [Testing Guidelines for AI](#13-testing-guidelines-for-ai)
14. [Common Patterns & Anti-Patterns](#14-common-patterns--anti-patterns)

---

## Detailed Module Documentation

This document provides a high-level overview of the codebase. For deep dives into specific subsystems, consult the [docs/](docs/) directory:

| Document                                                       | Module Path             | Description                                                                                  |
| -------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------- |
| **[CLI Module](docs/CLI_MODULE.md)**                           | `logic/controllers/cli/`        | Command-line interface, argument parsing, Hydra integration, entry points, and TUI           |
| **[Configuration Module](docs/CONFIGS_MODULE.md)**             | `logic/src/configs/`    | Config system architecture, Hydra composition, dataclass configurations, and overrides       |
| **[Configuration Guide](docs/CONFIGURATION_GUIDE.md)**         | -                       | Comprehensive guide to Hydra configuration, CLI syntax, multi-run sweeps, and best practices |
| **[Constants Module](docs/CONSTANTS_MODULE.md)**               | `logic/src/utils/`      | System-wide constants, problem definitions, enum types, and magic numbers                    |
| **[Data Module](docs/DATA_MODULE.md)**                         | `logic/src/data/`       | Dataset generation, loading utilities, data augmentation, and transforms                     |
| **[Environments Module](docs/ENVS_MODULE.md)**                 | `logic/src/envs/`       | Problem environments (VRPP, WCVRP, SCWCVRP), state management, and physics                   |
| **[Interfaces Module](docs/INTERFACES_MODULE.md)**             | `logic/src/interfaces/` | Abstract base classes, protocols, type definitions, and contracts                            |
| **[Models Module](docs/MODELS_MODULE.md)**                     | `logic/src/models/`     | Neural architectures (264KB doc): encoders, decoders, attention mechanisms, graph layers     |
| **[Pipeline Module](docs/PIPELINE_MODULE.md)**                 | `logic/src/pipeline/`   | Training pipeline, evaluation, simulation orchestration, RL algorithms, and HPO              |
| **[Policies Module](docs/POLICIES_MODULE.md)**                 | `logic/src/policies/`   | Classical solvers (Gurobi, ALNS, HGS), heuristics, and policy interfaces                     |
| **[Utilities Module](docs/UTILS_MODULE.md)**                   | `logic/src/utils/`      | Helper functions, I/O utilities, logging, debugging, encryption, and config loading          |
| **[Documentation Standards](docs/DOCUMENTATION_STANDARDS.md)** | -                       | Style guide, templates, and quality checklist for all documentation                          |

These module docs completely document the logic and GUI architectures. Please refer to them for class-level APIs and implementation details.

---

## 1. Project Overview & Mission

**WSmart+ Route** is a high-performance framework for solving complex Combinatorial Optimization (CO) problems, specifically:

- **Vehicle Routing Problem with Profits (VRPP)**: Select profitable subset of nodes to visit
- **Capacitated Waste Collection VRP (CWC VRP)**: Dynamic waste collection with capacity constraints
- **Stochastic Capacitated WCVRP (SCWCVRP)**: Waste generation with uncertainty modeling

### 1.1 Mission Statement

The project bridges **Deep Reinforcement Learning (DRL)** with **Operations Research (OR)**, providing:

1. **Benchmarking Environment**: Compare neural models against classical solvers
2. **Deployment Platform**: Production-ready routing optimization
3. **Research Testbed**: Experiment with novel architectures and algorithms
4. **Simulation Engine**: Test policies on realistic multi-day scenarios

### 1.2 Key Capabilities

| Capability                 | Description                                                     |
| -------------------------- | --------------------------------------------------------------- |
| **Neural Routing**         | Attention-based models (AM, TAM, DDAM) for constructive routing |
| **Classical Optimization** | Gurobi (exact), ALNS, HGS metaheuristics                        |
| **Hierarchical RL**        | Manager-Worker architecture for temporal decision-making        |
| **Meta-Learning**          | Cross-distribution generalization via MetaRNN                   |
| **Real-World Simulation**  | Multi-day scenarios with stochastic bin fill rates              |
| **Interactive GUI**        | WSmart-Route Studio — Tauri 2.0 desktop app (app/)              |

---

## 2. Technical Stack & Environmental Governance

### 2.1 Runtime Environment

| Component               | Specification | Notes                                   |
| ----------------------- | ------------- | --------------------------------------- |
| **Python**              | 3.9+          | Managed strictly via `uv`               |
| **Package Manager**     | `uv`          | Use `uv sync` for dependency resolution |
| **Virtual Environment** | `.venv/`      | Always activate before development      |

### 2.2 Primary Frameworks

| Framework             | Version | Purpose                            |
| --------------------- | ------- | ---------------------------------- |
| **PyTorch**           | 2.2.2   | Deep learning, CUDA-optimized      |
| **PyTorch Geometric** | 2.3.1   | Graph neural networks              |
| **Gurobi Optimizer**  | 11.0.3  | Exact optimization solver          |
| **Hexaly**            | 14.0+   | High-performance local search      |
| **OR-Tools**          | 9.4     | Google's optimization toolkit      |
| **PyVRP**             | 0.9.1+  | VRP solver library                 |
| **ALNS**              | 7.0+    | Adaptive Large Neighborhood Search |

### 2.3 Quality Control Tools

| Tool       | Purpose                       | Command               |
| ---------- | ----------------------------- | --------------------- |
| **ruff**   | Linter (Mandatory compliance) | `uv run ruff check .` |
| **black**  | Formatter                     | `uv run black .`      |
| **mypy**   | Type checking                 | `uv run mypy .`       |
| **pytest** | Testing                       | `uv run pytest`       |

### 2.4 Hardware Optimization

- **Target GPUs**: NVIDIA RTX 3090 Ti, RTX 4080 (laptop)
- **CUDA Version**: 11.8+ recommended
- **Memory**: Batch sizes tuned for 12GB+ VRAM

---

## 3. Core Architectural Boundaries

Maintain **strict separation of concerns** across these primary modules:

### 3.1 Logic Layer (`logic/src/`)

The computational engine, completely independent of UI concerns.

```
logic/src/
├── cli/              # Command-line interface & argument parsing
├── models/           # Neural network architectures
│   ├── modules/      # Atomic components (attention, normalization)
│   ├── subnets/      # Encoders, decoders, predictors
│   └── policies/     # Classical policies (HGS, local_search, split)
├── policies/         # Classical & heuristic algorithms
├── envs/             # Problem environments (replacing tasks/)
├── pipeline/         # Training, evaluation, simulation orchestration
│   ├── rl/           # Lightning-based RL pipeline
│   ├── features/     # Feature-specific implementations (train, eval, test)
│   └── simulations/  # Simulator engine
├── data/             # Data generation utilities
└── utils/            # Helper functions & utilities
```

### 3.2 Studio Layer (`app/`)

WSmart-Route Studio — Tauri 2.0 desktop application for visualization and interaction.

```
app/
├── src/              # React + TypeScript frontend
│   ├── pages/        # Monitor / Analysis / Launch / Files views
│   ├── components/   # Charts, maps, layout, live panels
│   ├── store/        # Zustand state (persisted launcher forms)
│   ├── hooks/        # Process monitor, DuckDB, file drop, theme
│   └── utils/        # Arrow pipeline, chart builders, log parsers
└── src-tauri/        # Rust backend (commands, process spawning, watchers)
```

### 3.3 Critical Boundaries

| Boundary      | Rule                                                       |
| ------------- | ---------------------------------------------------------- |
| Logic → Studio | Logic must **never** depend on the Studio app             |
| Studio → Logic | The Studio invokes Logic only via `main.py` CLI commands  |
| State Files   | `state_*.py` files are **critical**; test before modifying |
| Thread Safety | Heavy Studio work runs in Rust (tokio) or spawned CLI processes, never the WebView thread |

---

## 4. Key CLI Entry Points (Operational Playbook)

Always reference these commands when proposing code changes or workflows:

### 4.1 Environment Management

| Action             | Command                                                      |
| ------------------ | ------------------------------------------------------------ |
| Install uv         | `curl -LsSf https://astral.sh/uv/install.sh \| sh`           |
| Sync Environment   | `uv sync`                                                    |
| Activate venv      | `source .venv/bin/activate`                                  |
| Check Installation | `python -c "import torch; print(torch.cuda.is_available())"` |

### 4.2 Data Operations

| Action                   | Command                                                                      |
| ------------------------ | ---------------------------------------------------------------------------- |
| Generate Training Data   | `python main.py gen_data virtual --problem vrpp --graph_sizes 50`            |
| Generate Validation Data | `python main.py gen_data val --problem vrpp --graph_sizes 20 50 --seed 1234` |
| Generate Test Data       | `python main.py gen_data test --problem all --graph_sizes 20 50`             |

### 4.3 Training & Evaluation

| Action                      | Command                                                                                      |
| --------------------------- | -------------------------------------------------------------------------------------------- |
| Train Model                 | `python main.py train_lightning model=am env.name=vrpp env.num_loc=50`                       |
| Meta-RL Training            | `python main.py train_lightning experiment=meta_rl model=am env.name=vrpp train.n_epochs=50` |
| Hyperparameter Optimization | `python main.py train_lightning experiment=hpo env.name=wcvrp`                               |
| Evaluate Model              | `python main.py eval data/vrpp/test.pkl --model ./weights/best.pt`                           |

### 4.4 Simulation & Testing

| Action             | Command                                                            |
| ------------------ | ------------------------------------------------------------------ |
| Run Simulation     | `python main.py test_sim --policies regular gurobi alns --days 31` |
| Run Test Suite     | `python main.py test_suite`                                        |
| Run Specific Tests | `python main.py test_suite --module test_models`                   |
| Launch Studio      | `just studio`                                                      |
| Launch TUI         | `python main.py tui`                                               |

### 4.5 Code Quality

| Action      | Command                            |
| ----------- | ---------------------------------- |
| Lint Code   | `uv run ruff check .`              |
| Format Code | `uv run ruff format`               |
| Type Check  | `uv run mypy .`                    |
| Run Full CI | `just check` (if justfile present) |

---

## 5. External Access and Browser Usage Rules

The agent is authorized to use external tools to assist in development:

### 5.1 Web Search Authorization

| Purpose                  | Authorization Level                                   |
| ------------------------ | ----------------------------------------------------- |
| **Documentation Lookup** | ✅ Authorized - Gurobi 11+, Tauri 2.0, PyTorch 2.2+   |
| **API Verification**     | ✅ Authorized - Verify methods are not deprecated     |
| **Bug Investigation**    | ✅ Authorized - GitHub issues, Stack Overflow         |
| **CUDA/Driver Issues**   | ✅ Authorized - NVIDIA driver conflicts, Ubuntu fixes |

### 5.2 Knowledge Cutoff Management

**Directive**: Cross-reference internal training data with web search for technologies updated after January 2024:

- Gurobi performance tunings (v11+)
- PyTorch 2.x features (compile, inductor)
- Tauri 2.0 / React 19 recent API changes
- CUDA 12.x compatibility

### 5.3 Restricted Actions

| Action                    | Status        | Reason                |
| ------------------------- | ------------- | --------------------- |
| Modify production configs | ❌ Restricted | Requires human review |
| Delete data files         | ❌ Restricted | Potential data loss   |
| Push to main branch       | ❌ Restricted | Requires PR review    |
| Modify encryption keys    | ❌ Restricted | Security-critical     |

---

## 6. Domain-Specific Coding Standards

### 6.1 Mathematical & DRL Integrity

#### Invalid Move Prevention

Decoders **must** implement masking via `logic/src/utils/functions/boolmask.py` before sampling nodes:

```python
# CORRECT: Apply mask before softmax
logits = self.compute_logits(state)
logits = logits.masked_fill(mask, float('-inf'))
probs = F.softmax(logits, dim=-1)
action = torch.multinomial(probs, 1)

# WRONG: Sampling without masking can produce invalid routes
```

#### Exact Solvers (Branch-and-Price-and-Cut)

The BPC solver (`logic/src/policies/branch_and_price_and_cut/`) rigorously implements **Barnhart, Hane, and Vance (2000)** for ODIMCF problems.
**Strict Rules**:

1. **Phase I/II Constraints**: Farkas pricing MUST be explicitly resolved before normal pricing to guarantee initial LP feasibility.
2. **Mathematical Accuracy**: Subproblems must only inject columns using mathematically sound `reduced_cost` improvements; NEVER rank by raw initial profit.
3. **Bound Timing**: Lagrangian exact bounds ($z_{UB}$) must only be computed _after_ local Column Generation convergence.
4. **Cut Global Archival**: Cuts (e.g., Edge Clique) must be archived centrally (`GlobalCutPool`) and re-injected automatically at descendent B&B nodes.

#### Activation Scaling

Prefer custom modules in `logic/src/models/modules/normalization.py` over generic `nn.LayerNorm`:

```python
# PREFERRED
from logic.src.models.modules.normalization import Normalization
norm = Normalization(dim, normalization='instance')

# AVOID (unless specifically required)
norm = nn.LayerNorm(dim)
```

#### Configuration Sanitization

When passing Hydra/OmegaConf configs to Lightning modules (saved as hyperparameters):

1.  **Deep Sanitize**: Convert `DictConfig` and `ListConfig` to primitive `dict` and `list`.
2.  **Order Matters**: Sanitize _before_ injecting complex objects like `env` or `policy`.
3.  **Removal**: Remove non-serializable objects (like `env` instance) from `hparams` manually if needed in `__init__`.

```python
# CORRECT: Sanitize first, then inject
common_kwargs = deep_sanitize(cfg.rl)
common_kwargs["env"] = env  # injected after
model = MyModule(**common_kwargs)

# WRONG: Passing DictConfig directly causes YAML errors
model = MyModule(**cfg.rl)
```

#### State Transitions

**Never modify** `state_*.py` files without ensuring `logic/test/test_problems.py` passes:

```bash
# Before any state file modification
python main.py test_suite --module test_problems
```

### 6.2 Performance & Hardware

#### GPU Offloading

Ensure tensors are explicitly moved to device using `setup_sims.py`:

```python
from logic.src.utils.infrastructure.setup_sims import get_device

device = get_device(cuda_enabled=True)
tensor = tensor.to(device)
model = model.to(device)
```

#### Batch Size Guidelines

| GPU VRAM | Recommended Batch Size |
| -------- | ---------------------- |
| 8 GB     | 64-128                 |
| 12 GB    | 256                    |
| 24 GB    | 512-1024               |

#### GUI Threading

Heavy computations **must** inherit from `QThread`:

```python
# CORRECT: Background worker
class TrainingWorker(QThread):
    progress = Signal(int)

    def run(self):
        # Heavy computation here
        pass

# WRONG: Blocking main thread
def on_button_click(self):
    train_model()  # Freezes GUI
```

### 6.3 Code Style

#### Imports Organization

```python
# Standard library
import os
from typing import Dict, List, Optional

# Third-party
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

# Local imports
from logic.src.models.modules import MultiHeadAttention
from logic.src.utils import get_device
```

#### Type Hints

**Standard**: Use Python 3.8/3.9 compatible type hints from the `typing` module.

**Rationale**: While the project supports Python 3.9+, using `typing` module imports ensures:

- **Explicit clarity**: Imports clearly show typing dependencies
- **Backward compatibility**: Code works on Python 3.8-3.9 without `from __future__ import annotations`
- **Static analysis reliability**: Mypy and IDE autocomplete work consistently
- **Community convention**: Most Python typing documentation uses this style

**Required Style**:

```python
# CORRECT: Import from typing module (Python 3.8+ compatible)
from typing import Dict, List, Optional, Tuple, Union, Any

def compute_route_cost(
    route: List[int],
    distance_matrix: torch.Tensor,
    capacity: float = 100.0,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[float, List[int]]:
    """Compute total cost and optimized route."""
    ...

# WRONG: Lowercase built-in types (requires Python 3.10+ OR __future__ import)
def compute_route_cost(
    route: list[int],  # ❌ Don't use without __future__ import
    distance_matrix: torch.Tensor,
    capacity: float = 100.0
) -> tuple[float, list[int]]:  # ❌ Don't use without __future__ import
    ...
```

**Common Types**:

| Built-in | Typing Module   | Usage                                          |
| -------- | --------------- | ---------------------------------------------- | ------ |
| `list`   | `List[T]`       | `from typing import List`                      |
| `dict`   | `Dict[K, V]`    | `from typing import Dict`                      |
| `tuple`  | `Tuple[T, ...]` | `from typing import Tuple`                     |
| `set`    | `Set[T]`        | `from typing import Set`                       |
| `None`   | `Optional[T]`   | `from typing import Optional` (for `T          | None`) |
| -        | `Union[T, U]`   | `from typing import Union` (for `T             | U`)    |
| -        | `Any`           | `from typing import Any` (avoid when possible) |

**Protocol Classes**: For structural subtyping (duck typing with type safety):

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class PolicyLike(Protocol):
    """Structural type for policy objects."""
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        ...
```

**DO NOT use** `from __future__ import annotations` in new files unless absolutely necessary (e.g., circular import resolution).

#### Variable Naming Standards

To reduce ambiguity and maintain consistency across high-performance tensor code:

1.  **Single-Letter Tensors**: Acceptable ONLY in core algorithmic code and MUST be followed by a shape comment.
    ```python
    B, N = parent1.size()  # B=batch, N=nodes
    x = x.view(B, N, -1)
    ```
2.  **Weight Matrices**: Use prefix `W_` for learnable parameters (e.g., `W_query`, `W_out`).
3.  **Tensor Flattening**: Use underscore for clarity (e.g., `h_flat` instead of `hflat`).
4.  **Device-Awareness**: Use `setup_sims.get_device()` to avoid hardcoding `.cuda()`.

---

## 7. AI Review & Severity Protocol

Categorize feedback and edits using these severity levels:

### 7.1 CRITICAL (Must Fix Immediately)

- Breaking `state_*.py` transition logic
- Exposing credentials or API keys
- Cryptographic flaws in `crypto_utils.py`
- Data corruption in checkpoint files
- Security vulnerabilities (injection, XSS)

### 7.2 HIGH (Fix Before Merge)

- CUDA memory leaks
- Incorrect `skip_connection.py` usage
- `pyproject.toml` version mismatches
- Breaking changes to public APIs
- Test failures in CI pipeline

### 7.3 MEDIUM (Fix Soon)

- Suboptimal Pandas operations in `pandas_model.py`
- Deviations from ruff formatting
- Missing type hints on public functions
- Inefficient tensor operations
- Documentation inconsistencies

### 7.4 LOW (Nice to Have)

- Documentation typos
- Redundant imports
- UI padding/margin adjustments in `globals.py`
- Variable naming improvements
- Comment clarifications

---

## 8. Known Constraints & "No-Go" Areas

### 8.1 Legacy Preservation

| Pattern              | Rule                                  |
| -------------------- | ------------------------------------- |
| `*_copy.py` files    | Never edit - these are backups        |
| `legacy/` folders    | Read-only reference code              |
| Deprecated functions | Mark with `@deprecated`, don't delete |

### 8.2 Slurm Sensitivity

Cluster scripts (`scripts/slurm.sh`) use specific path mappings:

```bash
# Before modifying:
# 1. Verify SLURM_JOB_ID handling
# 2. Check module load commands
# 3. Validate output directory paths
```

### 8.3 Linux GUI Stability

When debugging the WSmart-Route Studio (Tauri) on Linux with WebKitGTK rendering issues:

```bash
# Recommended debug flags
export WEBKIT_DISABLE_COMPOSITING_MODE=1
just studio
```

### 8.4 Protected Files

| File/Directory          | Protection Level                |
| ----------------------- | ------------------------------- |
| `assets/keys/`          | Never commit to git             |
| `assets/model_weights/` | Large files - use Git LFS       |
| `.env` files            | Never commit - contains secrets |
| `*.lic` files           | License files - never commit    |

---

## 9. Usage Note

### 9.1 Session Initialization

When starting a new terminal session:

```bash
# 1. Kill any previous instances
pkill -f "python main.py"

# 2. Activate environment
source .venv/bin/activate

# 3. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 4. Update dependencies if needed
uv sync
```

### 9.2 Reference Files

| File                      | Purpose                              |
| ------------------------- | ------------------------------------ |
| `project_map.txt`         | Full project structure map           |
| `CLAUDE.md`               | This file (symlink to AGENTS.md)     |
| `docs/ARCHITECTURE.md`    | System design documentation          |
| `docs/TROUBLESHOOTING.md` | Common issues and fixes              |
| `docs/COMPATIBILITY.md`   | Model and Environment support matrix |

---

---

## 12. Data Formats & Schemas

### 12.1 Problem Instance Format

```python
{
    'loc': torch.Tensor,        # (batch, n_nodes, 2) - coordinates
    'depot': torch.Tensor,      # (batch, 2) - depot coordinates
    'capacity': float,          # Vehicle capacity
    'max_length': float,        # Maximum route length
}
```

### 12.2 Distance Matrix Format

```python
# Shape: (n_nodes, n_nodes), symmetric
# Values: Distance in kilometers
# Diagonal: 0.0
# File formats: .npy, .csv, .pkl
```

### 12.3 Waste Fill Data Format

```csv
timestamp,bin_id,fill_level,waste_type
2024-01-01,BIN001,0.45,plastic
2024-01-01,BIN002,0.72,glass
...
```

### 12.4 Simulation Output Format

```json
{
  "day": 1,
  "policy": "gurobi",
  "kg": 125.5,
  "km": 45.2,
  "cost": 22.6,
  "profit": 50.2,
  "overflows": 0,
  "tour": [0, 5, 12, 8, 0]
}
```

---

## 13. Testing Guidelines for AI

### 13.1 Before Making Changes

```bash
# Always run relevant tests first
python main.py test_suite --module test_models  # If modifying models
python main.py test_suite --module test_problems  # If modifying state/physics
python main.py test_suite --module test_policies  # If modifying policies
```

### 13.2 After Making Changes

```bash
# Run full test suite
python main.py test_suite

# Check code quality
uv run ruff check .

# Run type checks (non-blocking)
uv run mypy . || true
```

### 13.3 Test Markers

| Marker                     | Usage                   |
| -------------------------- | ----------------------- |
| `@pytest.mark.slow`        | Long-running tests      |
| `@pytest.mark.fast`        | Quick unit tests        |
| `@pytest.mark.integration` | Full system tests       |
| `@pytest.mark.train`       | Training pipeline tests |
| `@pytest.mark.gpu`         | Tests requiring CUDA    |

---

## 14. Common Patterns & Anti-Patterns

### 14.1 Preferred Patterns

```python
# ✅ Use factory methods
model = ModelFactory.create_model('am', problem='vrpp', opts=opts)

# ✅ Use device management utilities
device = get_device(cuda_enabled=True)
tensor = tensor.to(device)

# ✅ Use context managers for files
with open(filepath, 'r') as f:
    data = json.load(f)

# ✅ Use type hints
def compute_cost(route: List[int], distances: np.ndarray) -> float:
    ...
```

### 14.2 Anti-Patterns to Avoid

```python
# ❌ Don't hardcode device
tensor = tensor.cuda()  # Fails on CPU-only machines

# ❌ Don't modify state files without tests
# Always verify: python main.py test_suite --module test_problems

# ❌ Don't block the Studio WebView with synchronous work
# Long operations must run as spawned CLI processes (Rust tokio backend)

# ❌ Don't ignore masks in decoders
probs = F.softmax(logits, dim=-1)  # Can select invalid nodes
```

---

## Appendix: Quick Reference

### A.1 Key File Locations

| Purpose             | Location                          |
| ------------------- | --------------------------------- |
| Main entry point    | `main.py`                         |
| CLI parsers         | `logic/controllers/cli/`                  |
| Neural models       | `logic/src/models/`               |
| Classical policies  | `logic/src/policies/`             |
| Problem definitions | `logic/src/envs/`                 |
| Simulator engine    | `logic/src/pipeline/simulations/` |
| Studio desktop app  | `app/` (React + Tauri)            |
| Configuration       | `assets/configs/`                 |
| Model weights       | `assets/model_weights/`           |

### A.2 Important Constants

```python
# From logic/src/utils/definitions.py
MAX_WASTE = 1.0
VEHICLE_CAPACITY = 100.0
MAX_LENGTHS = {20: 2, 50: 3, 100: 4, 150: 5, 225: 6, 317: 7}
METRICS = ["overflows", "kg", "ncol", "kg_lost", "km", "kg/km", "cost", "profit"]
```

### A.3 Environment Variables

| Variable               | Purpose                  |
| ---------------------- | ------------------------ |
| `CUDA_VISIBLE_DEVICES` | GPU selection            |
| `GRB_LICENSE_FILE`     | Gurobi license path      |
| `WANDB_API_KEY`        | Weights & Biases logging |

---

**This guide is the authoritative reference for AI assistants working on WSmart+ Route. Keep it updated as the codebase evolves.**
