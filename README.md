<div align="center">

<img src="assets/images/logo-wsmartroute-white.png" alt="WSmart+ Route Logo" style="width: 35%; height: auto;">

# WSmart+ Route

**A High-Performance Framework for Combinatorial Optimization in Waste Collection Vehicle Routing**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Gurobi](https://img.shields.io/badge/Gurobi-11.0-ED1C24?logo=gurobi&logoColor=white)](https://www.gurobi.com/)
[![Hexaly](https://img.shields.io/badge/Hexaly-Optimizer-0078D7)](https://www.hexaly.com/)
[![OR-Tools](https://img.shields.io/badge/OR_Tools-9.4-4285F4?logo=google&logoColor=white)](https://developers.google.com/optimization)
[![PyVRP](https://img.shields.io/badge/PyVRP-0.9.1-blue)](https://pyvrp.org/)
[![TorchRL](https://img.shields.io/badge/TorchRL-0.1.0-blue)](https://github.com/acfharbinger/torchrl)
[![PyTorch Lightning](https://img.shields.io/badge/lightning-2.2+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</br>

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-24C8DB?logo=tauri&logoColor=white)](https://tauri.app/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![MyPy](https://img.shields.io/badge/MyPy-checked-2f4f4f.svg)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/pytest-testing-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-52%25-orange.svg)](https://coverage.readthedocs.io/)
[![Test](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml/badge.svg)](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml)

</br>

[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](https://www.kernel.org/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![Kubuntu](https://img.shields.io/badge/Kubuntu-0079C1?logo=kubuntu&logoColor=white)](https://kubuntu.org/)
[![Windows](https://img.shields.io/badge/Windows-0078D6?logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Intel](https://img.shields.io/badge/Intel-Hardware-0071C5?logo=intel&logoColor=white)](https://www.intel.com/)
[![AMD](https://img.shields.io/badge/AMD-Ryzen/EPYC-ED1C24?logo=amd&logoColor=white)](https://www.amd.com/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1/M2/M3-000000?logo=apple&logoColor=white)](https://www.apple.com/apple-silicon/)
[![CUDA RTX 4080](https://img.shields.io/badge/CUDA-RTX_4080-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![CUDA RTX 3090ti](https://img.shields.io/badge/CUDA-RTX_3090ti-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

</br>

[![Hydra](https://img.shields.io/badge/Hydra-1.4.0-blue)](https://github.com/acfharbinger/hydra)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.4-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.2.1-orange?logo=networkx&logoColor=white)](https://networkx.org/)
[![WandB](https://img.shields.io/badge/WandB-Logging-gold?logo=weightsandbiases&logoColor=white)](https://wandb.ai/)
[![ALNS](https://img.shields.io/badge/ALNS-7.0-purple)](https://github.com/alns/alns)

</br>

[![Make](https://img.shields.io/badge/Make-Build-blue?logo=gnu&logoColor=white)](https://www.gnu.org/software/make/)
[![Just](https://img.shields.io/badge/Just-Task_Runner-orange)](https://github.com/casey/just)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-025E8C?logo=dependabot&logoColor=white)](https://github.com/dependabot)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-blue?logo=sphinx&logoColor=white)](https://www.sphinx-doc.org/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-Viz-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tensorboard)

<p>
  <a href="#-documentation-hub"><strong>Documentation</strong></a> |
  <a href="docs/COMPATIBILITY.md"><strong>Compatibility</strong></a> |
  <a href="#-overview"><strong>Overview</strong></a> |
  <a href="#-key-features"><strong>Features</strong></a> |
  <a href="#-quickstart"><strong>Quickstart</strong></a> |
  <a href="#-model-ecosystem"><strong>Models</strong></a> |
  <a href="#-setup-dependencies"><strong>Setup</strong></a> |
  <a href="#-program-usage"><strong>Usage</strong></a> |
  <a href="#-contributing"><strong>Contributing</strong></a>
</p>

</div>

---

## Documentation Hub

Our comprehensive documentation covers every aspect of the WSmart+ Route system:

### Core Documentation

| Document                                          | Description                                                                                                                             | Target Audience                          |
| :------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------- |
| **[AGENTS.md](AGENTS.md)**                          | Complete registry of neural models, classical policies, and environment physics. The AI assistant guide for understanding the codebase. | Researchers, ML Engineers, AI Assistants |
| **[ARCHITECTURE.md](docs/moon/ARCHITECTURE.md)**  | High-level system design, data flow diagrams, design patterns, and module boundaries.                                                   | Architects, Senior Developers            |
| **[CONTRIBUTING.md](CONTRIBUTING.md)**            | Code style, Git workflow, PR process, and development guidelines.                                                                       | Contributors                             |
| **[DEVELOPMENT.md](docs/DEVELOPMENT.md)**         | Environment setup, CLI reference, development workflows, and debugging guides.                                                          | Developers                               |
| **[TESTING.md](docs/TESTING.md)**                 | Test suite organization, fixtures, coverage requirements, and best practices.                                                           | QA Engineers, Developers                 |
| **[COMPATIBILITY.md](docs/COMPATIBILITY.md)**     | Detailed matrix of supported models, environments, and configuration pairs.                                                             | Researchers, Developers                  |
| **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** | Common issues, diagnostic steps, error reference, and quick fixes.                                                                      | Everyone                                 |
| **[TUTORIAL.md](docs/TUTORIAL.md)**               | Deep dives into algorithms, code examples, and implementation guides.                                                                   | Developers, Researchers                  |

### Module Documentation

Detailed technical documentation for each subsystem. Essential for deep dives into specific components:

| Document                                                       | Description                                                                    | Module Path             |
| :------------------------------------------------------------- | :----------------------------------------------------------------------------- | :---------------------- |
| **[CLI Module](docs/CLI_MODULE.md)**                           | Command-line interface, argument parsing, Hydra integration, and entry points. | `logic/controllers/cli/`        |
| **[Configuration Module](docs/CONFIGS_MODULE.md)**             | Config system architecture, Hydra composition, and config classes.             | `logic/src/configs/`    |
| **[Configuration Guide](docs/CONFIGURATION_GUIDE.md)**         | Comprehensive guide to Hydra configuration, CLI overrides, and best practices. | -                       |
| **[Constants Module](docs/CONSTANTS_MODULE.md)**               | System-wide constants, problem definitions, and enum types.                    | `logic/src/utils/`      |
| **[Data Module](docs/DATA_MODULE.md)**                         | Dataset generation, loading utilities, and data augmentation.                  | `logic/src/data/`       |
| **[Environments Module](docs/ENVS_MODULE.md)**                 | Problem environments (VRPP, WCVRP, SCWCVRP) and state management.              | `logic/src/envs/`       |
| **[Interfaces Module](docs/INTERFACES_MODULE.md)**             | Abstract base classes, protocols, and type definitions.                        | `logic/src/interfaces/` |
| **[Models Module](docs/MODELS_MODULE.md)**                     | Neural architectures, encoders, decoders, and network components.              | `logic/src/models/`     |
| **[Pipeline Module](docs/PIPELINE_MODULE.md)**                 | Training, evaluation, simulation orchestration, and RL algorithms.             | `logic/src/pipeline/`   |
| **[Policies Module](docs/POLICIES_MODULE.md)**                 | Classical solvers (Gurobi, ALNS, HGS) and heuristic policies.                  | `logic/src/policies/`   |
| **[Utilities Module](docs/UTILS_MODULE.md)**                   | Helper functions, I/O utilities, logging, and debugging tools.                 | `logic/src/utils/`      |
| **[Documentation Standards](docs/DOCUMENTATION_STANDARDS.md)** | Style guide and templates for all documentation.                               | -                       |

---

## Overview

**WSmart+ Route** is a high-performance framework for solving complex Combinatorial Optimization (CO) problems, specifically the **Vehicle Routing Problem with Profits (VRPP)** and **Capacitated Waste Collection VRP (CWC VRP)**.

The project bridges **Deep Reinforcement Learning (DRL)** with **Operations Research (OR)**, providing a benchmarking and deployment environment where neural models (PyTorch) compete with traditional solvers (Gurobi, Hexaly).

### Mission

| Goal                        | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| **Research Platform**       | Benchmark neural routing agents against classical OR solvers |
| **Real-World Application**  | Optimize waste collection routes for municipalities          |
| **Simulation Environment**  | Test policies on realistic multi-day scenarios               |
| **User-Friendly Interface** | WSmart-Route Studio (Tauri) desktop app for training, evaluation, and analytics |

### Why WSmart+ Route?

1. **Neural + Classical**: Compare attention-based models with exact solvers and metaheuristics
2. **Production-Ready**: CLI and the WSmart-Route Studio desktop app for researchers and practitioners
3. **Extensible**: Add new models, policies, and problems with minimal boilerplate
4. **Well-Tested**: Comprehensive test suite with 50%+ code coverage
5. **Well-Documented**: Extensive documentation for all skill levels

---

## Key Features

### Neural Intelligence

| Capability                 | Description                                                                    |
| -------------------------- | ------------------------------------------------------------------------------ |
| **Attention-Based Models** | Transformer architectures (AM, TransGCN, DeepDecoder) for constructive routing |
| **Graph Neural Networks**  | GAT, GCN, GGAC encoders for spatial relationship modeling                      |
| **Hierarchical RL**        | Manager-Worker architecture with GAT-LSTM for temporal decision-making         |
| **Meta-Learning**          | MetaRNN for generalization across different problem distributions              |
| **Policy Gradients**       | REINFORCE, PPO, POMO with multiple baseline strategies                         |

### Optimization Solvers

| Solver       | Type          | Description                                |
| ------------ | ------------- | ------------------------------------------ |
| **Gurobi**   | Exact         | Branch-Cut-and-Price for optimal solutions |
| **Hexaly**   | Hybrid        | High-performance local search optimization |
| **ALNS**     | Metaheuristic | Adaptive Large Neighborhood Search         |
| **HGS**      | Genetic       | Hybrid Genetic Search with local search    |
| **OR-Tools** | Constraint    | Google's constraint programming solver     |

### Simulation Engine

| Feature                   | Description                                               |
| ------------------------- | --------------------------------------------------------- |
| **Multi-Day Scenarios**   | Test policies over extended time horizons (31-365 days)   |
| **Stochastic Fill Rates** | Gamma and empirical distributions for bin level modeling  |
| **Real Road Networks**    | OpenStreetMap integration for realistic distance matrices |
| **Parallel Execution**    | Multi-core simulation with checkpointing support          |
| **Overflow Tracking**     | Monitor and penalize bin overflows                        |

### User Interface

| Component              | Description                                      |
| ---------------------- | ------------------------------------------------ |
| **WSmart-Route Studio** | Tauri 2.0 desktop application (React + Rust)     |
| **Training Dashboard** | Real-time loss curves and validation metrics     |
| **Simulation Viewer**  | Interactive deck.gl route visualization          |
| **Analysis Tools**     | Comparative policy evaluation and statistics     |

---

## Quickstart

Get up and running in 3 steps:

```bash
# 1. Clone and sync dependencies
git clone https://github.com/ACFHarbinger/WSmart-Route.git
cd WSmart-Route
uv sync

# 2. Activate the environment
source .venv/bin/activate

# 3. Launch the WSmart-Route Studio desktop app
just studio
# Or run a quick simulation
python main.py test_sim --policies regular --size 20 --days 7
```

### Verify Installation

```bash
# Run the test suite
python main.py test_suite

# Check code quality
uv run ruff check .

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### System Requirements

| Component | Minimum  | Recommended |
| --------- | -------- | ----------- |
| Python    | 3.9      | 3.11        |
| RAM       | 16GB     | 32GB        |
| GPU       | 8GB VRAM | 16GB+ VRAM  |
| CUDA      | 11.8     | 12.x        |
| Disk      | 10GB     | 50GB        |

---

## Model Ecosystem

We provide a comprehensive library of neural architectures and classical policies:

### Neural Models

| Model                         | Architecture                  | Use Case                                      |
| ----------------------------- | ----------------------------- | --------------------------------------------- |
| **AttentionModel (AM)**       | Transformer (Encoder-Decoder) | General-purpose constructive routing          |
| **MandatorySelectionManager** | GNN + RNN                     | High-level temporal gating for HRL            |
| **TemporalAM**                | Transformer                   | Time-dependent attention mechanism            |
| **MetaRNN**                   | RNN/LSTM                      | Meta-learning for distribution generalization |
| **DeepDecoderAM**             | Deep Transformer              | Enhanced decoder for large instances          |
| **PointerNetwork**            | RNN + Attention               | Traditional pointer mechanism baseline        |
| **TransGCN**                  | Transformer + GCN             | Hybrid spatial-sequential encoding            |

### Graph Encoders

| Encoder         | Type                 | Description                              |
| --------------- | -------------------- | ---------------------------------------- |
| **GATEncoder**  | Graph Attention      | Multi-head attention for node embeddings |
| **GCNEncoder**  | Graph Convolution    | Standard GCN with aggregation            |
| **GGACEncoder** | Gated Graph          | Edge-node interaction with gating        |
| **TGCEncoder**  | Transformer-GCN      | Hybrid spatial encoding                  |
| **GACEncoder**  | Graph Attention Conv | Edge-aware attention mechanism           |
| **MLPEncoder**  | MLP                  | Non-graph baseline encoder               |

### Classical Policies

| Policy         | Type            | Description                                    |
| -------------- | --------------- | ---------------------------------------------- |
| **LookAhead**  | Rolling Horizon | N-day planning with sub-optimization           |
| **ALNS**       | Metaheuristic   | Destroy-repair operators with adaptive weights |
| **BPC**        | Exact           | Branch-and-Price-and-Cut via Gurobi/OR-Tools   |
| **HGS**        | Genetic         | Evolutionary operators with local search       |
| **Regular**    | Baseline        | Fixed-schedule periodic collection             |
| **LastMinute** | Reactive        | Threshold-triggered collection                 |
| **Gurobi**     | Exact           | Direct Gurobi MIP solver                       |
| **Hexaly**     | Hybrid          | Hexaly local search optimizer                  |

---

## Setup Dependencies

Choose your preferred installation method:

### UV (Recommended)

Fastest setup using the [uv package manager](https://github.com/astral-sh/uv):

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync the project and create a virtual environment
uv sync

# Activate the environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate.bat # Windows CMD
.venv\Scripts\Activate.ps1 # Windows PowerShell

# Verify installation
uv pip list
```

### Anaconda

```bash
conda env create --file env/environment.yml -y --name wsr
conda activate wsr
conda list
```

### Virtual Environment (Standard)

> **Note**: Requires Python 3.9+ pre-installed.

```bash
python3 -m venv env/.wsr
source env/.wsr/bin/activate
pip install -r env/requirements.txt
pip install -r env/pip_requirements.txt
```

### Setup Scripts

For automated setup:

```bash
# Linux
bash scripts/setup_env.sh <uv|conda|venv>

# Windows
scripts\setup_env.bat <uv|conda|venv>
```

---

## Setup Optimizers

### Gurobi (Academic License Available)

1. [Create an account](https://portal.gurobi.com/iam/login/) on the Gurobi website
2. [Request a license](https://portal.gurobi.com/iam/licenses/list) (free academic licenses available)
3. [Download the software](https://www.gurobi.com/downloads/) and install
4. Activate: `grbgetkey <your-license-key>`

### Hexaly

1. [Create an account](https://www.hexaly.com/login) on the Hexaly website
2. [Request a license](https://www.hexaly.com/account/on-premise-licenses)
3. Install: `pip install hexaly`

### CUDA Drivers

> **Important**: Required for GPU acceleration with NVIDIA hardware.

Download from the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and follow the [installation guide](https://docs.nvidia.com/cuda/index.html).

---

## Program Usage

### Generating Data

```bash
# Pre-generate training data for 50-vertex graphs
python main.py generate_data virtual --problem vrpp --graph_sizes 50 --n_epochs 10 --seed 42 --data_distribution gamma1

# Generate validation and test data
python main.py generate_data val --problem all --graph_sizes 20 50 --seed 42 --data_distribution gamma1
python main.py generate_data test --problem all --graph_sizes 20 50 --seed 42 --data_distribution gamma1
```

### Training

```bash
# Train Attention Model on VRPP with 50 vertices (Hydra + Lightning)
python main.py train_lightning model.name=am env.name=vrpp env.num_loc=50 train.n_epochs=100 train.batch_size=512

# Legacy Training (Old Pipeline)
python main.py train --model am --problem vrpp --graph_size 50 --baseline rollout
```

#### Resume Training

```bash
python main.py train --model am --graph_size 20 --n_epochs 50 --epoch_start 20 --load_path "results/vrpp_20/run_*/epoch-19.pt"
```

#### Multiple GPUs

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python main.py train

# Disable CUDA
python main.py train --no_cuda
```

### Evaluation

```bash
# Greedy evaluation
python main.py eval data/vrpp/vrpp20_test_seed1234.pkl --model assets/model_weights/vrpp_20/am --decode_strategy greedy

# Sampling evaluation (best of 1280)
python main.py eval data/vrpp/vrpp20_test_seed1234.pkl --model assets/model_weights/vrpp_20/am --decode_strategy sample --width 1280

# Beam search
python main.py eval data/vrpp/vrpp20_test_seed1234.pkl --model assets/model_weights/vrpp_20/am --decode_strategy bs --width 16
```

### Simulation Testing

```bash
# Test classical policies on 20-bin network for 31 days
python main.py test_sim --policies policy_last_minute policy_regular policy_look_ahead_a gurobi --problem vrpp --size 20 --days 31 --n_vehicles 1

# Test neural model on 365-day scenario
python main.py test_sim --policies am --problem vrpp --size 20 --days 365 --model_path assets/model_weights/vrpp_20/am/epoch-99.pt

# Multi-sample testing with resume support
python main.py test_sim --policies gurobi alns --problem vrpp --size 100 --days 365 --n_samples 10 --resume --cpu_cores -1
```

### Graphical User Interface (WSmart-Route Studio)

```bash
just studio          # dev mode (hot reload)
just studio-build    # release bundle
```

### Test Suite

```bash
# Run all tests
python main.py test_suite

# Run specific module/class/test
python main.py test_suite --module test_models
python main.py test_suite --class TestAttentionModel
python main.py test_suite --test test_forward_pass
python main.py test_suite --markers "unit and not slow"
```

### Hyperparameter Optimization

```bash
# Random search
python main.py hp_optim --model am --problem vrpp --search_strategy random --n_trials 50

# DEHB (Differential Evolution Hyperband)
python main.py hp_optim --model am --problem vrpp --search_strategy dehb --min_budget 1 --max_budget 50
```

### Meta-Reinforcement Learning

```bash
python main.py mrl_train --model meta_rnn --problem vrpp --graph_size 20 --n_tasks 10
```

---

## CLI Reference

| Command         | Description                 |
| --------------- | --------------------------- |
| `train`         | Train neural models         |
| `eval`          | Evaluate trained models     |
| `test_sim`      | Run simulation tests        |
| `generate_data` | Generate datasets           |
| `hp_optim`      | Hyperparameter optimization |
| `mrl_train`     | Meta-reinforcement learning |
| `test_suite`    | Run test suite              |
| `tui`           | Text user interface         |

Use `python main.py <command> --help` for detailed options.

---

## Scripts

| Script                                             | Description                                            |
| -------------------------------------------------- | ------------------------------------------------------ |
| [gen_data.sh](scripts/gen_data.sh)                 | Generate datasets for training, validation, or testing |
| [train.sh](scripts/train.sh)                       | Train Deep Learning models                             |
| [hyperparam_optim.sh](scripts/hyperparam_optim.sh) | Hyperparameter optimization                            |
| [test_sim.sh](scripts/test_sim.sh)                 | Test policies on the simulator                         |
| [slurm.sh](scripts/slurm.sh)                       | Run on Slurm cluster                                   |

Windows equivalents available as `.bat` files.

---

## Build Distribution

```bash
# Build source and binary distribution
uv build

# Create executable with PyInstaller
pyinstaller build.spec [--clean] [--noconsole]
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style and formatting (Ruff, Black)
- Git workflow and branching strategy
- Pull request process
- Testing requirements

### Quick Start for Contributors

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/WSmart-Route.git
cd WSmart-Route

# Set up development environment
uv sync
source .venv/bin/activate

# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes, then run tests and linting
uv run pytest
uv run ruff check .
uv run black .

# Commit and push
git commit -m "Add your feature"
git push origin feature/your-feature-name
```

---

## Project Structure

```
WSmart-Route/
├── logic/                    # Core Intelligence Layer
│   ├── src/
│   │   ├── models/          # Neural architectures
│   │   ├── policies/        # Classical solvers
│   │   ├── envs/            # Problem environments
│   │   ├── pipeline/        # Training/evaluation/simulation
│   │   └── utils/           # Shared utilities
│   └── test/                # Test suite
├── app/                      # WSmart-Route Studio (Tauri 2.0 desktop app)
│   ├── src/                 # React + TypeScript frontend
│   └── src-tauri/           # Rust backend (commands, process management)
├── data/                     # Datasets
├── assets/                   # Models, configs, outputs
├── scripts/                  # Automation scripts
└── main.py                   # CLI entry point
```

---

## Acknowledgments

This project adapts code and ideas from:

- [Attention, Learn to Solve Routing Problems](https://github.com/wouterkool/attention-learn-to-route)
- [POMO: Policy Optimization with Multiple Optima](https://github.com/yd-kwon/POMO/tree/master)
- [WSmart+ Bin Analysis](https://github.com/ACFPeacekeeper/wsmart_bin_analysis)
- [Do We Need Anisotropic Graph Neural Networks?](https://github.com/shyam196/egc)
- [Learning TSP Requires Rethinking Generalization](https://github.com/chaitjo/learning-tsp)
- [HGS-CVRP](https://github.com/vidalt/HGS-CVRP)
- [RL4CO](https://github.com/ai4co/rl4co)

---

## License

This project is dual-licensed under an open-core model:

- **Open source (free) — GNU AGPL-3.0.** Free to use, modify, and
  distribute for hobbyists, students, researchers, non-profits, and any
  other use that complies with the [AGPL-3.0](LICENSE) (Section A)'s
  copyleft and network source-disclosure terms.
- **Commercial (paid).** For proprietary, closed-source, or SaaS use that
  can't comply with the AGPL's obligations, a paid
  [commercial license](LICENSE) (Section B) is available — contact
  ACFHarbinger <afonso.fernandes100@gmail.com> for pricing and terms.
