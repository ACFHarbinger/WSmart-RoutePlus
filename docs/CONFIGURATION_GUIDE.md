# Comprehensive Configuration Guide

**Version**: 3.0
**Last Updated**: February 2026
**Framework**: [Hydra](https://hydra.cc/) 1.3+

This comprehensive guide covers all aspects of the WSmart-Route configuration system, from quick start commands to advanced Hydra features. Configurations are modular, composable, and fully overridable via command-line or YAML files.

---

## Table of Contents

1.  [**Quick Reference**](#1-quick-reference)
2.  [**Introduction to Hydra**](#2-introduction-to-hydra)
3.  [**Directory Structure**](#3-directory-structure)
4.  [**Configuration Groups**](#4-configuration-groups)
5.  [**CLI Override Syntax**](#5-cli-override-syntax)
6.  [**Multi-Run & Sweeps**](#6-multi-run--sweeps)
7.  [**Usage Examples**](#7-usage-examples)
8.  [**Advanced Features**](#8-advanced-features)
9.  [**Best Practices**](#9-best-practices)
10. [**Troubleshooting**](#10-troubleshooting)
11. [**Contributing**](#11-contributing)
12. [**Typical Workflows**](#12-typical-workflows)
13. [**References**](#13-references)
14. [**Version History**](#14-version-history)

---

## 1. Quick Reference

### Common Commands

```bash
# Training
python main.py train                                    # Default (CWCVRP, AM, HGS-ALNS expert)
python main.py train envs@train.env=vrpp model=tam     # Override environment & model
python main.py train envs@train.env=ttop               # TTOP (CVRPP + time budget)
python main.py train rl.algorithm=ppo                  # Change RL algorithm
python main.py train train.n_epochs=50                 # Override parameters

# Evaluation
python main.py eval eval.model=checkpoints/best.pt    # Evaluate model
python main.py eval eval.decoding.strategy=sampling    # Change decoding

# Simulation
python main.py test_sim sim.days=31                    # 31-day simulation
python main.py test_sim sim.policies=[hgs,alns]        # Compare policies

# Data Generation
python main.py gen_data data.problem=cwcvrp            # Generate CWCVRP data
```

### Configuration Cheat Sheet

#### Environments (`envs@<task>.env=`)

| Code      | Problem                              | Description                                          |
| --------- | ------------------------------------ | ---------------------------------------------------- |
| `cwcvrp`  | **Capacitated Waste Collection VRP** | Default. Multi-day, capacity, overflows              |
| `wcvrp`   | Waste Collection VRP                 | No capacity constraint                               |
| `vrpp`    | VRP with Profits                     | Select profitable nodes                              |
| `cvrpp`   | Capacitated VRP with Profits         | VRPP + capacity                                      |
| `ttop`    | Temporal Team Orienteering Problem   | CVRPP + independent per-trip time budget             |
| `scwcvrp` | Stochastic Capacitated WCVRP         | Stochastic waste generation                          |

#### Models (`model=`)

| Code           | Model                    | Best For                      |
| -------------- | ------------------------ | ----------------------------- |
| `am`           | **Attention Model**      | Default. General VRP problems |
| `tam`          | Temporal Attention Model | Multi-day temporal problems   |
| `deep_decoder` | Deep Decoder AM          | Large instances (200+ nodes)  |
| `ptr`          | Pointer Network          | Classical baseline            |
| `symnco`       | SymNCO                   | Symmetry-aware problems       |
| `moe`          | Mixture of Experts       | Multi-task learning           |

#### RL Algorithms (`rl.algorithm=`)

| Code                 | Algorithm                    | Use Case                  |
| -------------------- | ---------------------------- | ------------------------- |
| `adaptive_imitation` | **Adaptive Imitation**       | Default. IL→RL transition |
| `reinforce`          | REINFORCE                    | Policy gradient baseline  |
| `ppo`                | Proximal Policy Optimization | Stable training           |
| `imitation`          | Pure Imitation Learning      | Learn from expert         |

#### Expert Policies for RL (`rl.*.policy_config@=`)

| Path                       | Policy       | Speed      | Quality              |
| -------------------------- | ------------ | ---------- | -------------------- |
| `/tasks/policies/hgs_alns` | **HGS-ALNS** | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ (default) |
| `/tasks/policies/hgs`      | HGS          | ⭐⭐⭐⭐   | ⭐⭐⭐⭐             |
| `/tasks/policies/alns`     | ALNS         | ⭐⭐⭐     | ⭐⭐⭐⭐             |
| `/tasks/policies/ils`      | ILS          | ⭐⭐⭐⭐   | ⭐⭐⭐               |
| `/tasks/policies/rls`      | RLS          | ⭐⭐⭐⭐⭐ | ⭐⭐                 |
| `/tasks/policies/aco`      | ACO          | ⭐⭐       | ⭐⭐⭐⭐             |

**Usage:**

```bash
# Use HGS for faster training
python main.py train rl.imitation.policy_config@=/tasks/policies/rl/hgs

# Override expert time limit
python main.py train rl.imitation.policy_config.time_limit=10.0
```

#### Simulation Policies (`sim.policies=`)

| Code       | Policy                | Speed      | Quality    |
| ---------- | --------------------- | ---------- | ---------- |
| `hgs`      | Hybrid Genetic Search | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ |
| `alns`     | Adaptive LNS          | ⭐⭐⭐     | ⭐⭐⭐⭐   |
| `hgs_alns` | HGS-ALNS Hybrid       | ⭐⭐       | ⭐⭐⭐⭐⭐ |
| `gurobi`   | Exact Solver          | ⭐         | ⭐⭐⭐⭐⭐ |
| `neural`   | Neural Model          | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     |
| `regular`  | Fixed Frequency       | ⭐⭐⭐⭐⭐ | ⭐⭐       |

### Override Syntax Quick Reference

```bash
# Basic overrides
key=value                    # Simple override
outer.inner.key=value        # Nested override
key=[val1,val2]             # List override (no spaces!)

# Config group overrides
envs@train.env=vrpp          # Change training environment (group swap)
envs@train.env=ttop          # TTOP: CVRPP + per-trip time budget
model=tam                    # Change model
rl.algorithm=ppo            # Change RL algorithm

# Special overrides
+new_key=value              # Add new key
~key_to_remove              # Remove key
key@=/path/to/config        # Config composition
```

### Common Parameter Ranges

| Parameter       | Small (≤50) | Medium (50-150) | Large (150+) |
| --------------- | ----------- | --------------- | ------------ |
| `batch_size`    | 64-128      | 128-256         | 256-512      |
| `embed_dim`     | 64-128      | 128             | 128-256      |
| `n_layers`      | 2-3         | 3-4             | 4-6          |
| `learning_rate` | 1e-4        | 5e-5            | 1e-5         |
| `n_heads`       | 4-8         | 8               | 8-16         |

---

## 2. Introduction to Hydra

### What is Hydra?

[Hydra](https://hydra.cc/) is a framework for elegantly configuring complex applications. WSmart-Route uses Hydra to provide:

- **Composability**: Mix and match configurations from different groups
- **Reproducibility**: All experiment settings logged automatically
- **Sweeps**: Run hyperparameter searches with simple syntax
- **Type Safety**: Configs validated against Python dataclasses
- **CLI Overrides**: Change any parameter without editing files

### Why Two CLI Systems?

The codebase has a **dual CLI system** (see `main.py:212-256`):

| System     | Commands                                                         | Status            |
| ---------- | ---------------------------------------------------------------- | ----------------- |
| **Hydra**  | `train`, `eval`, `test_sim`, `gen_data`, `mrl_train`, `hp_optim` | ✅ Primary        |
| **Legacy** | `gui`, `test_suite`, `file_system`                               | ⚠️ Being migrated |

All new features use Hydra. Legacy commands will be migrated over time.

### Configuration Hierarchy

Configurations are loaded in this order (later overrides earlier):

1. **`config.yaml`** - Root configuration with global defaults
2. **Task config** - Task-specific settings (`tasks/*.yaml`)
3. **Environment config** - Problem definition (`envs/*.yaml`)
4. **Model config** - Neural architecture (`model/*.yaml`)
5. **CLI overrides** - Command-line arguments

### Composition Order

Hydra composes configs based on the `defaults` list in `assets/configs/config.yaml`:

```yaml
defaults:
  - _self_ # 1. Base settings from config.yaml
  - tasks: train # 2. Task-specific config
  - envs: cwcvrp # 3. Environment config
  - model: am # 4. Model architecture
```

**Priority**: Later configs override earlier ones. So `model/am.yaml` can override settings from `tasks/train.yaml`.

**Example:**

```bash
python main.py train  # Uses defaults: train + cwcvrp + am
```

This loads and merges:

1. `config.yaml` (base)
2. `tasks/train.yaml` (training parameters)
3. `envs/cwcvrp.yaml` (Capacitated Waste Collection VRP)
4. `model/am.yaml` (Attention Model architecture)

---

## 3. Directory Structure

```
assets/configs/
├── config.yaml                 # Root configuration (entry point)
│
├── tasks/                      # Task-specific configurations
│   ├── train.yaml             # Training configuration
│   ├── evaluation.yaml        # Model evaluation
│   ├── test_sim.yaml          # Multi-day simulation
│   ├── gen_data.yaml          # Dataset generation
│   ├── meta_train.yaml        # Meta-learning training
│   ├── hpo.yaml               # Hyperparameter optimization
│   ├── slurm.yaml             # HPC cluster execution
│   └── policies/              # Expert policy configs for RL training
│       ├── rl/                # RL-specific expert policies
│       │   ├── hgs.yaml       # Hybrid Genetic Search
│       │   ├── alns.yaml      # Adaptive Large Neighborhood Search
│       │   ├── hgs_alns.yaml  # HGS-ALNS hybrid (default)
│       │   ├── rls.yaml       # Random Local Search
│       │   ├── ils.yaml       # Iterated Local Search
│       │   ├── aco.yaml       # Ant Colony Optimization
│       │   └── README.md      # Expert policy usage guide
│       ├── hgs.yaml           # HGS for simulation
│       ├── alns.yaml          # ALNS for simulation
│       ├── hgs_alns.yaml      # HGS-ALNS for simulation
│       ├── aco.yaml           # ACO for simulation
│       ├── rls.yaml           # RLS for simulation
│       └── ils.yaml           # ILS for simulation
│
├── envs/                       # Environment (problem) definitions
│   ├── cwcvrp.yaml            # Capacitated Waste Collection VRP (default)
│   ├── wcvrp.yaml             # Waste Collection VRP
│   ├── vrpp.yaml              # VRP with Profits
│   ├── cvrpp.yaml             # Capacitated VRP with Profits
│   └── scwcvrp.yaml           # Stochastic Capacitated WCVRP
│
├── model/                      # Neural network architectures
│   ├── am.yaml                # Attention Model (default)
│   ├── tam.yaml               # Temporal Attention Model
│   ├── deep_decoder.yaml      # Deep Decoder AM
│   ├── ptr.yaml               # Pointer Network
│   ├── symnco.yaml            # SymNCO (Symmetry-aware)
│   └── moe.yaml               # Mixture of Experts
│
└── policies/                   # Routing policy configurations
    ├── policy_hgs.yaml        # Hybrid Genetic Search
    ├── policy_alns.yaml       # Adaptive Large Neighborhood Search
    ├── policy_hgs_alns.yaml   # HGS-ALNS Hybrid
    ├── policy_bpc.yaml        # Branch-and-Price-and-Cut (exact)
    ├── policy_hh_aco.yaml     # Hyper-Heuristic ACO
    ├── policy_ks_aco.yaml     # K-Sparse ACO
    ├── policy_lkh.yaml        # Lin-Kernighan-Helsgaun
    ├── policy_neural.yaml     # Neural policy wrapper
    ├── policy_sans.yaml       # SANS (Scheduling ANS)
    ├── policy_sisr.yaml       # SISR (Simple Insertion)
    ├── policy_tsp.yaml        # TSP solvers
    ├── policy_swc_tcf.yaml    # SWC-TCF policies
    ├── policy_cvrp.yaml       # CVRP-specific policies
    └── other/                 # Auxiliary configurations
        ├── ms_lookahead.yaml       # mandatory: Lookahead strategy
        ├── ms_last_minute_*.yaml   # mandatory: Threshold strategies
        ├── ms_regular_*.yaml       # mandatory: Fixed-frequency
        ├── ms_service_level*.yaml  # mandatory: Service level
        ├── pp_cls.yaml             # route improvement: CLS
        ├── pp_ftsp.yaml            # route improvement: Fast TSP
        └── pp_rds.yaml             # route improvement: RDS
```

---

## 4. Configuration Groups

### 1. Root Configuration: `config.yaml`

**Purpose:** Entry point for all configurations. Defines global defaults and composition order.

**Key Settings:**

- `seed`: Random seed for reproducibility (default: 42)
- `device`: Compute device (`cuda`, `cpu`, `cuda:0`)
- `task`: Active task type (`train`, `eval`, `test_sim`, `gen_data`)
- `wandb_mode`: Weights & Biases logging (`online`, `offline`, `disabled`)

**Defaults List:**

```yaml
defaults:
  - _self_
  - tasks: train # Default task
  - envs: cwcvrp # Default environment
  - model: am # Default model
```

---

### 2. Tasks: `tasks/*.yaml`

Task configurations define **what** the system does (training, evaluation, simulation, etc.).

#### 2.1 Training: `tasks/train.yaml`

**Purpose:** Configure reinforcement learning training with PyTorch Lightning.

**Key Sections:**

- **Training Settings:** Epochs, batch sizes, data loading
- **RL Algorithm:** REINFORCE, PPO, Imitation Learning, Adaptive Imitation
- **Optimizer:** Learning rate, scheduler, gradient clipping
- **Baselines:** Exponential, rollout, critic
- **Expert Policies:** Loaded from `tasks/policies/rl/`
- **mandatory Strategy:** Bin selection for WCVRP

**Example Configuration:**

```yaml
train:
  n_epochs: 100
  batch_size: 128
  train_data_size: 1280

rl:
  algorithm: "adaptive_imitation"
  baseline: "exponential"

  adaptive_imitation:
    policy_config: # Loaded from defaults
      # From tasks/policies/rl/hgs_alns.yaml
    il_weight: 1.0
    il_decay: 0.95
```

**CLI Overrides:**

```bash
# Change RL algorithm
python main.py train rl.algorithm=ppo

# Use different expert policy
python main.py train rl.imitation.policy_config@=/tasks/policies/rl/hgs

# Override expert policy parameters
python main.py train rl.imitation.policy_config.time_limit=60.0
```

#### 2.2 Evaluation: `tasks/evaluation.yaml`

**Purpose:** Evaluate trained models on test datasets.

**Key Settings:**

- **Decoding Strategy:** `greedy`, `sampling`, `beam_search`
- **Batch Size:** Evaluation batch size
- **Datasets:** List of test data paths
- **Model Path:** Checkpoint to evaluate

**Example:**

```bash
python main.py eval \
  eval.model=checkpoints/best_model.pt \
  eval.decoding.strategy=beam_search \
  eval.decoding.beam_width=[5,10,20]
```

#### 2.3 Simulation Testing: `tasks/test_sim.yaml`

**Purpose:** Multi-day waste collection simulation with multiple policies.

**Key Settings:**

- **Simulation Horizon:** `days` (7, 31, 365)
- **Problem Size:** `size` (20, 50, 100, 150)
- **Policies:** List of policies to compare
- **Random Seeds:** `n_samples` for statistical analysis

**Example:**

```bash
python main.py test_sim \
  sim.days=31 \
  sim.size=100 \
  sim.policies=[hgs,alns,neural] \
  sim.n_samples=20
```

#### 2.4 Data Generation: `tasks/gen_data.yaml`

**Purpose:** Generate training/validation/test datasets.

**Key Settings:**

- **Dataset Type:** `train`, `val`, `test`
- **Problem:** Environment type
- **Graph Sizes:** List of problem sizes
- **Number of Samples:** Instances per size
- **Data Distribution:** Waste generation patterns

**Example:**

```bash
python main.py gen_data \
  data.dataset_type=train \
  data.problem=cwcvrp \
  data.graph_sizes=[20,50,100] \
  data.num_samples=10000
```

#### 2.5 Meta-Learning: `tasks/meta_train.yaml`

**Purpose:** Train models for cross-distribution generalization.

**Key Features:**

- Multi-distribution training
- MetaRNN context encoding
- Distribution-specific adaptation

**Example:**

```bash
python main.py meta_train \
  train.data_distributions=[gamma1,gamma2,uniform]
```

#### 2.6 Hyperparameter Optimization: `tasks/hpo.yaml`

**Purpose:** Automated hyperparameter search with Optuna.

**Search Spaces:**

- Learning rate
- Batch size
- Model dimensions
- RL hyperparameters

**Example:**

```bash
python main.py hpo \
  hpo.n_trials=100 \
  hpo.search_space=learning_rate,batch_size
```

---

### 3. Environments: `envs/*.yaml`

Environment configurations define **what problem** is being solved.

#### 3.1 CWCVRP (Default): `envs/cwcvrp.yaml`

**Capacitated Waste Collection VRP** - Realistic waste collection with capacity constraints and temporal dynamics.

**Key Features:**

- Multi-day simulation
- Vehicle capacity constraints
- Bin overflow penalties
- Real-world geographic areas

**Parameters:**

```yaml
env:
  name: "cwcvrp"
  num_loc: 100 # Number of bins

  graph:
    area: "riomaior" # Geographic area
    waste_type: "plastic" # Waste category
    distance_method: "gmaps" # Distance calculation
    vertex_method: "mmn" # Vertex placement

  reward:
    cost_weight: 10.0 # Distance penalty
    overflow_penalty: 10.0 # Overflow penalty
    waste_weight: 10.0 # Collection reward
```

#### 3.2 VRPP: `envs/vrpp.yaml`

**Vehicle Routing Problem with Profits** - Select profitable subset of nodes to visit.

**Key Features:**

- Node wastes/rewards
- Maximum route length constraint
- No capacity constraints
- Maximize profit - cost

#### 3.3 SCWCVRP: `envs/scwcvrp.yaml`

**Stochastic Capacitated WCVRP** - Waste generation with uncertainty.

**Key Features:**

- Probabilistic fill rates
- Multiple waste scenarios
- Robust routing under uncertainty

#### 3.4 Other Environments

- **WCVRP** (`wcvrp.yaml`): Basic waste collection (no capacity)
- **CVRPP** (`cvrpp.yaml`): Capacitated VRP with profits
- **SCWCVRP** (`scwcvrp.yaml`): Selective capacitated waste collection

---

### 4. Models: `model/*.yaml`

Model configurations define **neural network architecture**.

#### 4.1 Attention Model (Default): `model/am.yaml`

**Purpose:** Transformer-based constructor for VRP.

**Architecture:**

```yaml
model:
  encoder:
    type: "gat" # Graph Attention Network
    embed_dim: 128 # Embedding dimension
    hidden_dim: 512 # FFN hidden size
    n_layers: 3 # Encoder depth
    n_heads: 8 # Attention heads
    normalization:
      norm_type: "instance"
    activation:
      name: "gelu"
    dropout: 0.1
    connection_type: "residual"

  decoder:
    type: "attention"
    n_layers: 2 # Decoder depth
```

**Encoder Types:**

- **`gat`**: Graph Attention Network (default, best for VRP)
- **`gcn`**: Graph Convolutional Network
- **`tgc`**: Transformer Graph Convolution
- **`mlp`**: Structure-agnostic MLP

**CLI Overrides:**

```bash
# Increase model capacity
python main.py train model.encoder.embed_dim=256 model.encoder.n_layers=6

# Change encoder type
python main.py train model.encoder.type=gcn

# Modify attention
python main.py train model.encoder.n_heads=16 model.encoder.spatial_bias=true
```

#### 4.2 Temporal Attention Model: `model/tam.yaml`

**Purpose:** Time-aware attention for multi-day problems.

**Key Features:**

- Temporal horizon encoding
- GRF predictor for future states
- History-aware attention

**Use Cases:**

- WCVRP with multi-day simulation
- Predictive routing
- Temporal dependency modeling

#### 4.3 Deep Decoder AM: `model/deep_decoder.yaml`

**Purpose:** Deeper decoder for complex action spaces.

**Features:**

- Increased decoder depth (4-6 layers)
- Enhanced action selection
- Better for large instances (200+ nodes)

#### 4.4 Other Models

- **Pointer Network** (`ptr.yaml`): Classic RNN-based pointer
- **SymNCO** (`symnco.yaml`): Symmetry-aware neural CO
- **MoE** (`moe.yaml`): Mixture of Experts routing

---

### 5. Policies: `policies/*.yaml`

Policy configurations define **classical/heuristic routing algorithms** for simulation and benchmarking.

#### 5.1 Hybrid Genetic Search: `policies/policy_hgs.yaml`

**Algorithm:** State-of-the-art metaheuristic combining genetic algorithm with local search.

**Parameters:**

```yaml
hgs:
  custom:
    - time_limit: 60 # Optimization time budget
    - population_size: 50 # Genetic population
    - elite_size: 10 # Elite preservation
    - mutation_rate: 0.2 # Mutation probability
    - engine: "custom" # PyVRP backend
    - mandatory_selection: ["other/ms_lookahead.yaml"]
    - route_improvement: []
```

**Use Cases:**

- Production deployment (60s yields excellent solutions)
- Benchmarking against commercial solvers
- Large instances (150-500 nodes)

#### 5.2 ALNS: `policies/policy_alns.yaml`

**Algorithm:** Adaptive Large Neighborhood Search with destroy-repair operators.

**Parameters:**

```yaml
alns:
  custom:
    - time_limit: 60
    - max_iterations: 10000
    - start_temp: 100.0
    - cooling_rate: 0.995
    - max_removal_pct: 0.3
```

**Use Cases:**

- Tight capacity constraints
- Time window problems
- Rapid solution refinement

#### 5.3 HGS-ALNS Hybrid: `policies/policy_hgs_alns.yaml`

**Algorithm:** Combines HGS global search with ALNS local refinement.

**Parameters:**

```yaml
hgs_alns:
  custom:
    - time_limit: 60
    - population_size: 50
    - elite_size: 10
    - alns_education_iterations: 50
```

**Performance:** Often 0.5-1.5% better than pure HGS, at 50-100% higher runtime.

#### 5.4 Branch-and-Price-and-Cut: `policies/policy_bpc.yaml`

**Algorithm:** Exact optimization via integer programming.

**Solvers:**

- Gurobi (commercial, fastest)
- OR-Tools (open-source)
- VRPy (column generation)

**Use Cases:**

- Small instances (≤50 nodes)
- Proving optimality
- Benchmarking upper bounds

#### 5.5 ACO Variants

- **HH-ACO** (`policy_hh_aco.yaml`): Hyper-heuristic ACO
- **KS-ACO** (`policy_ks_aco.yaml`): K-sparse ACO

#### 5.6 Neural Policy: `policies/policy_neural.yaml`

**Purpose:** Wrapper for trained neural models in simulation.

**Configuration:**

```yaml
neural:
  custom:
    - model_path: "checkpoints/best_model.pt"
    - decoding_strategy: "greedy"
    - mandatory_selection: ["other/ms_lookahead.yaml"]
```

---

### 6. Auxiliary Configurations: `policies/helpers/*.yaml`

#### 6.1 mandatory Strategies

**Purpose:** Determine which bins **must** be collected in waste collection problems.

| Strategy          | File                      | Description                                      |
| ----------------- | ------------------------- | ------------------------------------------------ |
| **Lookahead**     | `ms_lookahead.yaml`       | Predict future overflows with GRF                |
| **Last Minute**   | `ms_last_minute_cf*.yaml` | Collect when fill ≥ threshold (70%, 90%)         |
| **Regular**       | `ms_regular_lvl*.yaml`    | Fixed frequency (every 3, 4, 5 days)             |
| **Service Level** | `ms_service_level*.yaml`  | Statistical overflow prediction (84% confidence) |

**Example:**

```bash
# Change mandatory strategy for HGS
python main.py test_sim \
  sim.policies=[hgs] \
  +hgs.custom[0].mandatory_selection=["other/ms_last_minute_cf90.yaml"]
```

#### 6.2 route improvement

**Purpose:** Refine routes after construction.

| Method   | File           | Description           |
| -------- | -------------- | --------------------- |
| **CLS**  | `pp_cls.yaml`  | Chained Local Search  |
| **FTSP** | `pp_ftsp.yaml` | Fast TSP refinement   |
| **RDS**  | `pp_rds.yaml`  | Random Descent Search |

---

### 7. Expert Policies for RL Training: `tasks/policies/rl/*.yaml`

**Purpose:** Configure expert policies for **Imitation Learning** and **Adaptive Imitation Learning**.

These configs use `_target_` to instantiate Python dataclasses in `logic/src/configs/rl/policies/`.

**Available Policies:**

- **HGS** (`hgs.yaml`): Fast, good quality
- **ALNS** (`alns.yaml`): General-purpose
- **HGS-ALNS** (`hgs_alns.yaml`): Best quality (default)
- **RLS** (`rls.yaml`): Very fast, lower quality
- **ILS** (`ils.yaml`): Balanced
- **ACO** (`aco.yaml`): Research-oriented

**Usage:**

```bash
# Use HGS for faster training
python main.py train rl.imitation.policy_config@=/tasks/policies/rl/hgs

# Override expert policy time limit
python main.py train rl.imitation.policy_config.time_limit=10.0
```

**See:** `tasks/policies/rl/README.md` for detailed usage guide.

---

## 5. CLI Override Syntax

### Basic Overrides

```bash
# Override single values
python main.py train seed=1234
python main.py train device=cpu

# Override nested values (dot notation)
python main.py train model.embed_dim=256
python main.py train rl.learning_rate=1e-4
python main.py train env.num_loc=100
```

### Config Group Selection

```bash
# Change model architecture
python main.py train model=tam  # Temporal Attention Model
python main.py train model=ptr  # Pointer Network

# Change environment (swap the composed env group, not only env.name)
python main.py eval envs@eval.env=vrpp eval.env.name=vrpp eval.problem=vrpp
python main.py eval envs@eval.env=scwcvrp eval.env.name=scwcvrp eval.problem=scwcvrp
python main.py eval envs@eval.env=ttop eval.env.name=ttop eval.problem=ttop

# Change task
python main.py task=eval  # Evaluation task
python main.py task=test_sim  # Simulation testing
```

### Multiple Overrides

```bash
# Combine overrides
python main.py train \
    model=tam \
    envs@train.env=scwcvrp \
    seed=42 \
    model.n_encode_layers=6 \
    rl.batch_size=512
```

### List and Dict Overrides

```bash
# Override list elements
python main.py train env.graph_sizes='[20,50,100]'

# Override entire dicts
python main.py train 'rl.optimizer={name: adamw, lr: 1e-3}'
```

### Special Overrides

```bash
# Config composition (@ symbol)
rl.imitation.policy_config@=/tasks/policies/rl/hgs

# Add new key (+ prefix)
+new_key=value

# Remove key (~ prefix)
~key_to_remove
```

### Interpolation

Reference other config values:

```yaml
train:
  batch_size: 128
  eval_batch_size: ${train.batch_size} # Inherits from train
```

**CLI:**

```bash
python main.py train train.batch_size=256  # eval_batch_size becomes 256 too
```

### Environment Variable Integration

```yaml
wandb:
  api_key: ${oc.env:WANDB_API_KEY}
```

---

## 6. Multi-Run & Sweeps

### Basic Multi-Run

Use `-m` or `--multirun` flag:

```bash
# Sweep over seeds
python main.py -m train seed=1,2,3,4,5

# Sweep over model dimensions
python main.py -m train model.embed_dim=64,128,256,512

# Combinatorial sweep (Cartesian product)
python main.py -m train \
    model.embed_dim=128,256 \
    rl.learning_rate=1e-3,1e-4 \
    seed=1,2,3
# Runs 2×2×3 = 12 experiments
```

### Range Sweeps

```bash
# Numeric ranges
python main.py -m train seed=range(1,11)  # Seeds 1-10
python main.py -m train rl.batch_size=range(128,513,128)  # 128,256,384,512

# Glob patterns (for sweeping over models/envs)
python main.py -m train model=glob(*)  # All models
python main.py -m train envs@train.env=glob(*)  # All environments
```

### Sweep Output Organization

Hydra creates separate directories for each run:

```
outputs/
├── 2026-02-08/
│   ├── 10-30-15/  # First run
│   │   ├── .hydra/
│   │   │   ├── config.yaml  # Full merged config
│   │   │   └── overrides.yaml  # CLI overrides
│   │   └── train.log
│   ├── 10-31-22/  # Second run
│   └── ...
└── multirun/
    └── 2026-02-08/
        ├── 10-35-00/
        │   ├── 0/  # seed=1
        │   ├── 1/  # seed=2
        │   └── 2/  # seed=3
        └── ...
```

---

## 7. Usage Examples

### Example 1: Train AM on VRPP

```bash
python main.py train \
  envs@train.env=vrpp \
  model=am \
  train.n_epochs=100 \
  train.batch_size=256
```

### Example 1b: Train AM on TTOP (CVRPP + per-trip time budget)

```bash
python main.py train \
  envs@train.env=ttop \
  model=am \
  train.env.shift_hours=7.0
```

### Example 2: Train TAM on CWCVRP with Adaptive Imitation

```bash
python main.py train \
  envs@train.env=cwcvrp \
  model=tam \
  rl.algorithm=adaptive_imitation \
  rl.adaptive_imitation.policy_config@=/tasks/policies/rl/hgs_alns \
  rl.adaptive_imitation.il_weight=1.0 \
  rl.adaptive_imitation.il_decay=0.95
```

### Example 3: Evaluate Trained Model with Beam Search

```bash
python main.py eval \
  envs@eval.env=cwcvrp \
  model=am \
  eval.model=checkpoints/best_model.pt \
  eval.decoding.strategy=beam_search \
  eval.decoding.beam_width=[5,10,20] \
  eval.datasets=["data/cwcvrp_test_100.pkl"]
```

### Example 4: Compare Multiple Policies in Simulation

```bash
python main.py test_sim \
  sim.days=31 \
  sim.size=100 \
  sim.policies=[hgs,alns,hgs_alns,neural] \
  sim.n_samples=20 \
  sim.area=riomaior
```

### Example 5: Generate Training Data

```bash
python main.py gen_data \
  data.dataset_type=train \
  data.problem=cwcvrp \
  data.graph_sizes=[20,50,100] \
  data.num_samples=10000 \
  data.area=riomaior
```

### Example 6: Hyperparameter Optimization

```bash
python main.py hpo \
  envs@hpo.env=vrpp \
  model=am \
  hpo.n_trials=100 \
  hpo.search_space=all
```

### Example 7: Meta-Learning Training

```bash
python main.py meta_train \
  envs@meta_rl.env=cwcvrp \
  model=am \
  train.data_distributions=[gamma1,gamma2,uniform] \
  train.n_epochs=200
```

### Example 8: Ablation Studies

```bash
# Compare encoder types
python main.py -m train \
    model.encoder_type=gat,gcn,tgc,mlp \
    seed=range(1,6)  # 4×5 = 20 runs

# Compare normalization strategies
python main.py -m train \
    model.normalization=batch,layer,instance,group \
    model.embed_dim=128,256
```

---

## 8. Advanced Features

### 1. Structured Configs with `_target_`

Instantiate Python classes directly from YAML:

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
  weight_decay: 0.0001
```

### 2. Config Composition

Load multiple configs and merge them:

```yaml
defaults:
  - base_config
  - override envs: cwcvrp
  - override model: tam
  - _self_
```

### 3. Conditional Configs

```yaml
model:
  dropout: ${select:${env.name},vrpp:0.1,cwcvrp:0.2}
```

### 4. Config Groups Override

Override entire config groups:

```bash
# Use a custom task config
python main.py --config-name=my_custom_task

# Override with config from different location
python main.py train +experiment=my_experiment
```

### 5. Custom Resolvers

Register custom interpolation resolvers:

```python
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("double", lambda x: x * 2)
```

```yaml
value: 10
doubled: ${double:${value}} # 20
```

---

## 9. Best Practices

### 1. Config Organization

- **Keep configs focused:** One responsibility per file
- **Use descriptive names:** `train_large_batch.yaml` not `config1.yaml`
- **Document parameters:** Add inline comments explaining ranges and defaults
- **Version control:** Commit all config changes with descriptive messages

### 2. Parameter Tuning

| Parameter       | Small Problem (≤50) | Medium (50-150) | Large (150+) |
| --------------- | ------------------- | --------------- | ------------ |
| `batch_size`    | 64-128              | 128-256         | 256-512      |
| `embed_dim`     | 64-128              | 128             | 128-256      |
| `n_layers`      | 2-3                 | 3-4             | 4-6          |
| `learning_rate` | 1e-4                | 5e-5            | 1e-5         |

### 3. Reproducibility

Always set seeds in three places:

```bash
python main.py train seed=42  # Hydra level
```

Config:

```yaml
seed: 42 # Global seed
train:
  seed: 42 # Task-level seed
```

Python:

```python
torch.manual_seed(42)
np.random.seed(42)
```

### 4. GPU Memory Management

If OOM errors occur:

1. Reduce `batch_size`
2. Reduce `embed_dim` and `hidden_dim`
3. Reduce `n_layers`
4. Enable gradient checkpointing
5. Use mixed precision training (`train.precision=16`)

```bash
# GPU memory optimization example
python main.py train \
  train.batch_size=64 \
  model.encoder.embed_dim=64 \
  model.encoder.n_layers=2 \
  train.precision=16
```

### 5. Debugging Configs

```bash
# Print full resolved config
python main.py train --cfg job

# Print config to file
python main.py train --cfg job > resolved_config.yaml

# Check which config files are loaded
python main.py train --info defaults

# Print config schema
python main.py train --cfg hydra

# Resolve all interpolations
python main.py train --cfg job --resolve
```

### 6. Expert Policy Time Limits

| Use Case               | Time Limit | Quality    |
| ---------------------- | ---------- | ---------- |
| Fast prototyping       | 10-15s     | ⭐⭐       |
| **Balanced (default)** | 30s        | ⭐⭐⭐⭐   |
| High quality training  | 60s+       | ⭐⭐⭐⭐⭐ |

```bash
# Fast training
python main.py train rl.imitation.policy_config.time_limit=10.0

# High quality
python main.py train rl.imitation.policy_config.time_limit=60.0
```

---

## 10. Troubleshooting

### Common Errors

#### 1. "Could not find config group"

**Error:** `Could not find 'xyz.yaml'`

**Solution:**

- Check file exists in correct directory
- Verify config group name matches directory name
- Use relative paths: `tasks/policies/rl/hgs` not `rl/hgs`

```bash
# Check available models
ls assets/configs/model/*.yaml
python main.py train model=am  # Use correct name
```

#### 2. Override Not Applied

**Error:** Parameter unchanged after CLI override

**Solution:**

- Check override syntax: `key=value` (no spaces around `=`)
- For nested configs: `outer.inner.key=value`
- For lists: `key=[val1,val2,val3]` (no spaces)
- For config groups: `key@=/path/to/config`

#### 3. Type Mismatch

**Error:** `ValidationError: expected int, got str`

**Solution:**

```bash
# Correct
python main.py train train.n_epochs=100

# Incorrect (interpreted as string)
python main.py train train.n_epochs="100"
```

#### 4. Circular Dependencies

**Error:** `ConfigCompositionException: Circular reference`

**Solution:**

- Avoid `${key}` that reference each other
- Use explicit values instead of interpolation
- Restructure config hierarchy

#### 5. Missing Required Field

**Error:** `MissingMandatoryValue: field 'xyz' is mandatory`

**Solution:**

- Add field to config file
- Provide via CLI: `xyz=value`
- Set default in structured config

#### 6. Output Directory Conflicts

If Hydra complains about existing output directories:

```bash
# Option 1: Allow overwrites
python main.py train hydra.run.dir=./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}

# Option 2: Clear old outputs
rm -rf outputs/

# Option 3: Use unique run names
python main.py train +experiment_name=my_ablation_v2
```

### Debugging Tips

```bash
# Verify config loads without errors
python main.py --config-name=your_config --cfg job

# Check environment variables
python main.py train --cfg job | grep -A5 "wandb"

# Test multi-run without executing
python main.py -m train --multirun seed=1,2,3 --dry-run
```

---

## 11. Contributing

When adding new configurations:

### 1. Follow Naming Conventions

- Tasks: `verb_noun.yaml` (e.g., `train.yaml`, `gen_data.yaml`)
- Models: `model_name.yaml` (e.g., `am.yaml`, `tam.yaml`)
- Environments: `problem_acronym.yaml` (e.g., `cwcvrp.yaml`)
- Policies: `policy_name.yaml` (e.g., `policy_hgs.yaml`)

### 2. Add Comprehensive Comments

Include in config files:

- Purpose and use cases
- Parameter ranges and defaults
- Performance characteristics
- Example usage

### 3. Update Documentation

- Add entry to directory structure in this guide
- Document new parameters
- Provide usage example

### 4. Test Configurations

```bash
# Verify config loads without errors
python main.py --config-name=your_config --cfg job

# Run with new config
python main.py your_new_task
```

---

## 12. Typical Workflows

### Workflow 1: Train New Model

```bash
# 1. Generate data
python main.py gen_data data.problem=cwcvrp data.num_samples=10000

# 2. Train with adaptive imitation
python main.py train envs@train.env=cwcvrp model=am train.n_epochs=100

# 3. Evaluate
python main.py eval eval.model=checkpoints/best_model.pt
```

### Workflow 2: Benchmark Policies

```bash
# Compare policies on 31-day simulation
python main.py test_sim \
  sim.days=31 \
  sim.size=100 \
  sim.policies=[hgs,alns,hgs_alns,neural] \
  sim.n_samples=20
```

### Workflow 3: Hyperparameter Search

```bash
# Automated HPO with Optuna
python main.py hpo \
  envs@hpo.env=vrpp \
  model=am \
  hpo.n_trials=100
```

### Workflow 4: Meta-Learning

```bash
# Train on multiple distributions
python main.py meta_train \
  envs@meta_rl.env=cwcvrp \
  train.data_distributions=[gamma1,gamma2,uniform] \
  train.n_epochs=200
```

---

## 13. References

### Official Documentation

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Hydra Override Grammar](https://hydra.cc/docs/advanced/override_grammar/basic/)
- [Hydra Multi-Run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)
- [Structured Configs](https://hydra.cc/docs/tutorials/structured_config/intro/)

### Project Documentation

- **AI Assistant Guide:** `CLAUDE.md` - Comprehensive guide for AI coding assistants
- **Architecture Documentation:** `ARCHITECTURE.md` - System design and components
- **Model Compatibility:** `COMPATIBILITY.md` - Model-environment support matrix
- **Troubleshooting:** `TROUBLESHOOTING.md` - Common issues and fixes
- **Expert Policies for RL:** `tasks/policies/rl/README.md` - Detailed policy guide

### Code References

- **Config Loaders:** `logic/src/utils/configs/config_loader.py`
- **Model Factory:** `logic/src/utils/infrastructure/setup_sims.py`
- **Environment Factory:** `logic/src/envs/`
- **Policy Factory:** `logic/src/policies/adapters.py`

---

## 14. Version History

| Version | Date     | Changes                                        |
| ------- | -------- | ---------------------------------------------- |
| 3.0     | Feb 2026 | Comprehensive merged guide with Hydra features |
| 2.5     | Jan 2026 | Added meta-learning and HPO tasks              |
| 2.0     | Dec 2025 | Hydra 1.3 migration, structured configs        |
| 1.5     | Nov 2025 | Added temporal environments (TAM)              |
| 1.0     | Oct 2025 | Initial configuration system                   |

---

**For questions or issues:**

- GitHub Issues: https://github.com/ACFHarbinger/WSmart-Route/issues
- Documentation: `docs/` directory
- AI Assistant Guide: `CLAUDE.md`

---

_This guide is maintained as part of the WSmart-Route project. Contributions and improvements are welcome._
