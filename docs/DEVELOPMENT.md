# Development Guide

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Gurobi](https://img.shields.io/badge/Gurobi-11.0-ED1C24?logo=gurobi&logoColor=white)](https://www.gurobi.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MyPy](https://img.shields.io/badge/MyPy-checked-2f4f4f.svg)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/pytest-testing-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-50%25-green.svg)](https://coverage.readthedocs.io/)
[![CI](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml/badge.svg)](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml)

> **Version**: 2.0
> **Last Updated**: January 2026
> **Purpose**: Comprehensive development setup and workflow documentation

This guide provides everything you need to set up a development environment for WSmart Route+ and understand the development workflows.

---

## ⚡ < 5 Minute Quickstart

**Get up and running with WSmart Route+ in under 5 minutes!**

```bash
# 1. Install uv package manager (< 1 min)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and enter repository (< 1 min)
git clone https://github.com/ACFPeacekeeper/WSmart-Route.git
cd WSmart-Route

# 3. Sync dependencies and activate environment (< 2 min)
uv sync
source .venv/bin/activate  # Linux/macOS
# OR: .venv\Scripts\activate  # Windows

# 4. Verify installation (< 30 seconds)
python -c "import torch; print(f'✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 5. Run a quick training example (< 1 min)
python main.py train_lightning model=am env.name=vrpp env.num_loc=20 train.n_epochs=1 train.batch_size=32
```

### What Just Happened?

- ✅ **uv** installed (fast Rust-based package manager)
- ✅ **Repository cloned** with all source code
- ✅ **Dependencies synced** (PyTorch, Lightning, solvers, etc.)
- ✅ **Virtual environment activated** (.venv)
- ✅ **Quick training run** completed (Attention Model on VRPP-20)

### Next Steps

- **Generate data**: `python main.py gen_data virtual --problem vrpp --graph_sizes 20 50`
- **Run simulation**: `python main.py test_sim --policies regular gurobi --days 7`
- **Launch Studio**: `just studio`
- **Run tests**: `python main.py test_suite`

### Troubleshooting Quick Fixes

| Issue                   | Quick Fix                                               |
| ----------------------- | ------------------------------------------------------- |
| `uv: command not found` | Restart terminal or run `source ~/.bashrc`              |
| `CUDA not available`    | CPU-only mode works fine for testing!                   |
| Import errors           | Run `uv sync --all-extras` to ensure all deps installed |
| Permission errors       | Don't use `sudo` - uv manages everything in userspace   |

**For detailed setup, continue reading below. For immediate help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).**

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Project Structure](#3-project-structure)
4. [Development Workflows](#4-development-workflows)
5. [CLI Commands Reference](#5-cli-commands-reference)
6. [Configuration System](#6-configuration-system)
7. [Debugging](#7-debugging)
8. [Performance Profiling](#8-performance-profiling)
9. [IDE Setup](#9-ide-setup)
10. [Common Tasks](#10-common-tasks)

---

## Documentation Resources

### Module Documentation

For detailed technical documentation of individual subsystems, see the [docs/](docs/) directory:

- **[CLI Module](docs/CLI_MODULE.md)** - Command-line interface, argument parsing, and Hydra integration
- **[Configuration Module](docs/CONFIGS_MODULE.md)** - Config system architecture and Hydra composition
- **[Configuration Guide](docs/CONFIGURATION_GUIDE.md)** - Comprehensive Hydra configuration guide
- **[Data Module](docs/DATA_MODULE.md)** - Dataset generation and loading utilities
- **[Environments Module](docs/ENVS_MODULE.md)** - Problem environments and state management
- **[Models Module](docs/MODELS_MODULE.md)** - Neural architectures and network components
- **[Pipeline Module](docs/PIPELINE_MODULE.md)** - Training, evaluation, and simulation orchestration
- **[Policies Module](docs/POLICIES_MODULE.md)** - Classical solvers and heuristic policies
- **[Utilities Module](docs/UTILS_MODULE.md)** - Helper functions and debugging tools

### Documentation Standards

When contributing documentation, follow the standards in [docs/DOCUMENTATION_STANDARDS.md](docs/DOCUMENTATION_STANDARDS.md):

- Consistent structure and formatting
- Required sections (header, TOC, overview, examples, best practices)
- Code example requirements
- Style guidelines (tone, headers, tables, cross-references)

---

## 1. Prerequisites

### 1.1 System Requirements

| Component    | Minimum                                 | Recommended   |
| ------------ | --------------------------------------- | ------------- |
| **OS**       | Ubuntu 20.04+ / Windows 10+ / macOS 12+ | Ubuntu 22.04+ |
| **Python**   | 3.9                                     | 3.9           |
| **RAM**      | 16 GB                                   | 32 GB         |
| **GPU VRAM** | 8 GB                                    | 12+ GB        |
| **Storage**  | 20 GB                                   | 50 GB SSD     |

### 1.2 Required Software

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y \
    git \
    curl \
    build-essential \
    python3-dev \
    python3-venv

# macOS (with Homebrew)
brew install git curl python@3.11

# Windows
# Install Git, Python 3.11+, and Visual Studio Build Tools
```

### 1.3 Optional Software

| Software               | Purpose                   |
| ---------------------- | ------------------------- |
| **CUDA Toolkit 11.8+** | GPU acceleration          |
| **Gurobi 11.0+**       | Exact optimization solver |
| **Hexaly 14.0+**       | High-performance solver   |
| **Docker**             | Containerized deployment  |

---

## 2. Environment Setup

### 2.1 Clone Repository

```bash
git clone https://github.com/ACFPeacekeeper/WSmart-Route.git
cd WSmart-Route
```

### 2.2 Install uv Package Manager

`uv` is the recommended package manager for this project.

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

### 2.3 Sync Dependencies

```bash
# Sync all dependencies (creates .venv automatically)
uv sync

# Include development dependencies
uv sync --all-extras --dev

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate.bat  # Windows CMD
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 2.4 Verify Installation

```bash
# Verify Python version
python --version

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Verify core dependencies
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

### 2.5 Alternative Setup Methods

#### Conda

```bash
conda env create -f env/environment.yml -n wsr
conda activate wsr
```

#### Standard venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
```

#### Setup Scripts

```bash
# Linux/macOS
bash scripts/setup_env.sh uv  # or 'conda' or 'venv'

# Windows
scripts\setup_env.bat uv
```

---

## 3. Project Structure

### 3.1 Directory Overview

```
WSmart-Route/
├── logic/                 # Core computational logic
│   ├── src/               # Source code
│   │   ├── cli/           # Command-line interface
│   │   ├── models/        # Neural network models
│   │   ├── policies/      # Classical optimization algorithms
│   │   ├── pipeline/      # Orchestration (learning, features, simulations); learning/envs holds the problem environments
│   │   ├── data/          # Data generation
│   │   └── utils/         # Utilities
│   └── test/              # Unit & integration tests
│
├── app/                   # WSmart-Route Studio (Tauri 2.0)
│   ├── src/               # React + TypeScript frontend
│   └── src-tauri/         # Rust backend
│
├── scripts/               # Shell/batch scripts
├── assets/                # Static assets (configs, images, weights)
├── data/                  # Datasets (git-ignored)
├── env/                   # Environment files
├── reports/               # Research reports
│
├── main.py                # Main entry point
├── pyproject.toml         # Project configuration
└── *.md                   # Documentation files
```

### 3.2 Key Files

| File                      | Purpose                                          |
| ------------------------- | ------------------------------------------------ |
| `main.py`                 | Application entry point                          |
| `pyproject.toml`          | Project metadata, dependencies, tool configs     |
| `.pre-commit-config.yaml` | Pre-commit hook configuration                    |
| `Makefile`                | Build and automation targets                     |
| `CLAUDE.md`               | AI assistant instructions (symlink to AGENTS.md) |

---

## 4. Development Workflows

### 4.1 Daily Development Cycle

```bash
# 1. Start a new session
source .venv/bin/activate

# 2. Pull latest changes
git fetch origin
git pull origin main

# 3. Sync dependencies (if pyproject.toml changed)
uv sync

# 4. Create feature branch
git checkout -b feature/my-feature

# 5. Make changes, run tests
python main.py test_suite --module test_models

# 6. Check code style
uv run ruff check .
uv run ruff format .

# 7. Commit and push
git add .
git commit -m "feat(models): add new encoder"
git push -u origin feature/my-feature
```

### 4.2 Running the Application

#### CLI Mode

```bash
# Train a model
python main.py train_lightning model=am env.name=vrpp env.num_loc=50

# Evaluate a model
python main.py eval data/vrpp/test.pkl --model ./weights/best.pt

# Run simulation
python main.py test_sim --policies regular gurobi --days 31
```

#### GUI Mode (WSmart-Route Studio)

```bash
# Launch the Studio desktop app (dev mode, hot reload)
just studio

# Build a release bundle
just studio-build
```

#### TUI Mode

```bash
# Interactive terminal UI
python main.py tui
```

### 4.3 Code Quality Workflow

```bash
# Run linter
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Type checking
uv run mypy logic/src/

# Run all checks (using pre-commit)
pre-commit run --all-files
```

### 4.4 Testing Workflow

```bash
# Run all tests
python main.py test_suite

# Run specific test module
python main.py test_suite --module test_models

# Run tests with coverage
uv run pytest --cov=logic/src --cov-report=html

# Run only fast tests
uv run pytest -m "not slow"

# Run GPU tests
uv run pytest -m "gpu"
```

### 4.5 Just Command Runner

If you have `just` installed, use these commands:

| Command           | Description                                   |
| ----------------- | --------------------------------------------- |
| `just check-docs` | Verify all modules have docstrings            |
| `just test`       | Run the full test suite                       |
| `just test-unit`  | Run unit tests only (faster)                  |
| `just lint`       | Run code linting (Ruff)                       |
| `just format`     | Format code (Black/Ruff)                      |
| `just build-docs` | Build Sphinx documentation                    |
| `just run-train`  | Run the training pipeline (Hydra + Lightning) |
| `just run-sim`    | Run the simulation pipeline                   |

---

## 5. CLI Commands Reference

### 5.1 Main Commands

| Command       | Description                 |
| ------------- | --------------------------- |
| `train`       | Train a neural model        |
| `mrl_train`   | Meta-RL training            |
| `hp_optim`    | Hyperparameter optimization |
| `eval`        | Evaluate a trained model    |
| `test_sim`    | Run simulation tests        |
| `gen_data`    | Generate datasets           |
| `tui`         | Launch terminal UI          |
| `test_suite`  | Run test suite              |
| `file_system` | File operations             |

### 5.2 Training Command

```bash
python main.py train_lightning [OVERRIDES]

# Model options
model=am                     # Model type (default: am)
model.embed_dim=128      # Embedding dimension
model.hidden_dim=512         # Hidden dimension
model.n_encode_layers=3      # Encoder layers

# Training options
env.name=vrpp                # Problem type
env.num_loc=20               # Graph size
train.batch_size=256         # Batch size
train.n_epochs=100           # Training epochs
optim.lr=1e-4                # Learning rate

# RL options
rl.baseline=rollout          # Baseline strategy (rollout, exponential, etc.)

# Hardware
train.accelerator=gpu        # Enable GPU (default: gpu)
seed=42                      # Random seed
```

### 5.3 Simulation Command

```bash
python main.py test_sim [OPTIONS]

--policies LIST              # Policies to test (e.g., regular gurobi alns)
--days INT                   # Simulation days (default: 31)
--area STRING                # Geographic area (default: riomaior)
--waste_type {plastic,glass,paper}  # Waste type
--n_samples INT              # Number of samples
--n_vehicles INT             # Number of vehicles
--output_dir PATH            # Output directory
--resume                     # Resume from checkpoint
```

### 5.4 Data Generation Command

```bash
python main.py gen_data [TYPE] [OPTIONS]

# Types: virtual, val, test

--problem {vrpp,wcvrp,all}   # Problem type
--graph_sizes LIST           # Graph sizes (e.g., 20 50 100)
--num_samples INT            # Samples per size
--data_distribution STRING   # Distribution type
--seed INT                   # Random seed
--output_dir PATH            # Output directory
```

---

## 6. Configuration System

### 6.1 Configuration Hierarchy

1. **Defaults** (hardcoded in parsers)
2. **YAML configs** (`assets/configs/*.yaml`)
3. **Environment variables**
4. **CLI arguments** (highest priority)

### 6.2 YAML Configuration

```yaml
# assets/configs/train.yaml
model:
  name: am
  encoder: gat
  embed_dim: 128
  hidden_dim: 512
  n_encode_layers: 3
  n_heads: 8

training:
  batch_size: 256
  n_epochs: 100
  learning_rate: 1e-4
  optimizer: adam

rl:
  algorithm: reinforce
  baseline: rollout
  entropy_weight: 0.01

problem:
  type: vrpp
  graph_size: 50
```

### 6.3 Environment Variables

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1  # For debugging

# Solver licenses
export GRB_LICENSE_FILE=/path/to/gurobi.lic
export LOCALSOLVER_HOME=/path/to/hexaly

# Logging
export WANDB_API_KEY=your_key
export WANDB_PROJECT=wsmart-route

```

### 6.4 Using Configs

````bash
# Start with default experiment config
python main.py train_lightning experiment=base

```bash
# Override specific options
python main.py train_lightning experiment=base train.batch_size=128 model.hidden_dim=256
````

### 6.5 Policy & Selection Configuration

Policies and selection strategies use a mix of YAML and XML configurations located in `assets/configs/policies/`.

- **YAML (\*.yaml)**: Defines classical or neural policy adaptors (e.g., `policy_alns.yaml`).
  - Can reference sub-components like Selection Strategies via XML pointers.
- **XML (\*.xml)**: Defines specific selection strategy behaviors (e.g., `ms_lookahead_days7.xml`).
  - Used for modular composition of strategies.

**Example Usage:**

To use a specific policy configuration during simulation:

```bash
# Run simulation with ALNS policy configured via YAML
python main.py test_sim --policies policy_alns
```

---

## 7. Debugging

### 7.1 Python Debugging

```python
# Insert breakpoint
import pdb; pdb.set_trace()

# Or use built-in breakpoint (Python 3.7+)
breakpoint()

# Rich traceback for better error display
from rich.traceback import install
install(show_locals=True)
```

### 7.2 PyTorch Debugging

```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Check for NaN/Inf
def check_tensor(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")

# Profile memory
print(torch.cuda.memory_summary())
```

### 7.3 Logging

```python
from loguru import logger

# Configure logging
logger.add("logs/debug.log", level="DEBUG")
logger.add("logs/error.log", level="ERROR")

# Usage
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### 7.4 Studio Debugging

```bash
# Launch with WebKitGTK compositing disabled (Linux rendering issues)
WEBKIT_DISABLE_COMPOSITING_MODE=1 just studio

# Type-check the frontend / lint the Rust backend
just studio-check
just studio-clippy
```

---

## 8. Performance Profiling

### 8.1 CPU Profiling

```python
import cProfile
import pstats

# Profile a function
profiler = cProfile.Profile()
profiler.enable()
# ... code to profile ...
profiler.disable()

# Print stats
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(20)
```

### 8.2 GPU Profiling

```python
import torch

# Basic timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# ... GPU operations ...
end.record()
torch.cuda.synchronize()
print(f"Time: {start.elapsed_time(end):.2f} ms")

# PyTorch Profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # ... code to profile ...

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 8.3 Memory Profiling

```bash
# Install memory profiler
uv pip install memory_profiler

# Profile memory usage
python -m memory_profiler main.py train_lightning train.batch_size=64
```

### 8.4 NVIDIA Tools

```bash
# nsight-systems profiling
nsys profile python main.py train_lightning

# nvidia-smi monitoring
watch -n 1 nvidia-smi
```

---

## 9. IDE Setup

### 9.1 VS Code

#### Extensions

- Python (Microsoft)
- Pylance
- Ruff
- Python Debugger
- GitLens
- Jupyter

#### Settings

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.autoImportCompletions": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit"
  },
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "ruff.lint.args": ["--config=pyproject.toml"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true,
    "**/*.egg-info": true
  }
}
```

#### Launch Configuration

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Train Model",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "args": [
        "train_lightning",
        "model=am",
        "env.num_loc=20",
        "train.n_epochs=2"
      ],
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Run Tests",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "args": ["test_suite", "--module", "test_models"],
      "console": "integratedTerminal"
    }
  ]
}
```

### 9.2 PyCharm

1. **Set interpreter**: File > Settings > Project > Python Interpreter > `.venv/bin/python`
2. **Enable Ruff**: Install Ruff plugin, configure in Settings > Tools > Ruff
3. **Configure tests**: Run > Edit Configurations > pytest
4. **Mark directories**: Right-click `logic/src` > Mark Directory as > Sources Root

### 9.3 Jupyter Notebooks

```bash
# Install Jupyter
uv pip install jupyter jupyterlab

# Launch
jupyter lab

# Use project kernel
python -m ipykernel install --user --name=wsmart-route
```

---

## 10. Common Tasks

### 10.1 Adding a New Model

1. Create model file:

   ```python
   # logic/src/models/my_model.py
   import torch.nn as nn
   from .attention_model import AttentionModel

   class MyModel(AttentionModel):
       def __init__(self, problem, **kwargs):
           super().__init__(problem, **kwargs)
           # Custom initialization
   ```

2. Register in factory:

   ```python
   # logic/src/models/model_factory.py
   from .my_model import MyModel

   MODEL_REGISTRY['my_model'] = MyModel
   ```

3. Add CLI argument:

   ```python
   # logic/src/configs/model/my_model.yaml
   # Define default parameters for Hydra
   ```

4. Write tests:
   ```python
   # logic/test/test_models.py
   def test_my_model():
       model = MyModel(problem='vrpp')
       # ... assertions
   ```

### 10.2 Adding a New Policy

1. Create policy file:

   ```python
   # logic/src/policies/my_policy.py
   from .adapters import Policy

   class MyPolicy(Policy):
       def solve(self, distances, wastes, capacity, depot=0):
           # Implementation
           return routes, profit, cost
   ```

2. Register in adapters:
   ```python
   # logic/src/policies/adapters.py
   POLICY_REGISTRY['my_policy'] = MyPolicy
   ```

### 10.3 Updating Dependencies

```bash
# Add a new dependency
uv add package_name

# Add dev dependency
uv add --dev package_name

# Remove dependency
uv remove package_name

# Update all dependencies
uv lock --upgrade
uv sync
```

### 10.4 Running on HPC Cluster

```bash
# Submit Slurm job
sbatch scripts/slurm.sh

# Check job status
squeue -u $USER

# View job output
tail -f logs/slurm-*.out
```

### 10.5 Building Distribution

```bash
# Build wheel
uv build

# Create executable (PyInstaller)
pyinstaller build.spec --clean
```

### 10.6 Building Documentation

```bash
# Build Sphinx docs
cd logic/docs
make html

# View docs
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

---

## Quick Reference

### Essential Commands

```bash
# Environment
source .venv/bin/activate
uv sync

# Development
python main.py train --model am --graph_size 20
python main.py test_suite
uv run ruff check .

# Studio
just studio
```

### Important Files

| File                                | Purpose            |
| ----------------------------------- | ------------------ |
| `main.py`                           | Entry point        |
| `logic/controllers/cli/base/registry.py`    | Command dispatcher |
| `logic/src/models/model_factory.py` | Model creation     |
| `logic/src/policies/adapters.py`    | Policy creation    |
| `pyproject.toml`                    | Project config     |

### Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

---

**Happy developing!**
