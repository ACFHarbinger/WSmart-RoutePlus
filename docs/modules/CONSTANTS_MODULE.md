# Global Constants, Registries & Configuration Keys

**Module**: `logic/src/constants`
**Purpose**: Centralized technical specification of global constants, mapping registries, and configuration keys used throughout WSmart-Route.
**Version**: 3.0
**Last Updated**: February 2026

---

## Table of Contents

1.  [**Overview**](#1-overview)
2.  [**Module Organization**](#2-module-organization)
3.  [**Policy Constants**](#3-policy-constants)
4.  [**Metric Constants**](#4-metric-constants)
5.  [**Simulation Constants**](#5-simulation-constants)
6.  [**Routing Constants**](#6-routing-constants)
7.  [**Model Constants**](#7-model-constants)
8.  [**Waste Management Constants**](#8-waste-management-constants)
9.  [**HPO Constants**](#9-hpo-constants)
10. [**User Interface Constants**](#10-user-interface-constants)
11. [**System Constants**](#11-system-constants)
12. [**Path Constants**](#12-path-constants)
13. [**Testing Constants**](#13-testing-constants)
14. [**Statistics Constants**](#14-statistics-constants)
15. [**Dashboard Constants**](#15-dashboard-constants)
16. [**Legacy Constants**](#16-legacy-constants)
17. [**Best Practices**](#17-best-practices)
18. [**Migration Guide**](#18-migration-guide)
19. [**Cross-References**](#19-cross-references)
20. [**Appendix: Quick Reference**](#20-appendix-quick-reference)

---

## 1. Overview

The constants module provides a centralized registry of configuration values, naming conventions, and mappings used throughout the WSmart+ Route codebase. It ensures consistency across:

- **Configuration systems** (Hydra YAML files, CLI arguments)
- **Policy factories** (solver selection, hyperparameter parsing)
- **Metric logging** (RL training, simulation results, evaluation)
- **Visualization** (GUI themes, plotting colors, terminal output)
- **Problem definitions** (capacity constraints, optimization parameters)

### Design Principles

1. **Immutability**: Use tuples and uppercase naming for true constants
2. **Documentation**: Each constant includes usage context and valid ranges
3. **Deprecation Safety**: Legacy constants retained with migration warnings
4. **Type Safety**: Type hints for all mappings and function registries

### Import Strategy

```python
# Import from top-level module (re-exports all sub-modules)
from logic.src.constants import MAX_WASTE, VEHICLE_CAPACITY, METRICS

# Or import specific sub-module for clarity
from logic.src.constants.simulation import MAX_WASTE, VEHICLE_CAPACITY
from logic.src.constants.policies import SIMPLE_POLICIES, THRESHOLD_POLICIES
```

---

## 2. Module Organization

The constants module is split into 15 sub-modules for maintainability:

| File                | Category                           | Primary Use Cases                                |
| ------------------- | ---------------------------------- | ------------------------------------------------ |
| `policies.py`       | Policy naming & classification     | Policy factory, CLI parsing, config resolution   |
| `metrics.py`        | Metric name aliasing               | RL training, logging normalization, GUI charts   |
| `simulation.py`     | Physics & problem constraints      | Environment definitions, simulation orchestrator |
| `routing.py`        | Solver parameters & penalties      | Classical algorithms (HGS, ALNS, Gurobi)         |
| `models.py`         | Architecture hyperparameters       | Model factory, encoder/decoder instantiation     |
| `waste.py`          | Real-world waste management data   | Portugal case studies, empirical distributions   |
| `hpo.py`            | Hyperparameter optimization config | Optuna, DEHB, Ray Tune integration               |
| `user_interface.py` | Visual styling                     | CLI output, matplotlib plots, GUI themes         |
| `system.py`         | Infrastructure & operations        | File system, operator evaluation, threading      |
| `paths.py`          | Project directory resolution       | Asset loading, workspace detection               |
| `testing.py`        | Test suite registry                | pytest integration, CI/CD pipelines              |
| `stats.py`          | Statistical function mapping       | Data analysis, summary computation               |
| `dashboard.py`      | Visualization color schemes        | Simulation dashboard, map rendering              |
| `tasks.py`          | **DEPRECATED**                     | Legacy constants (backward compatibility)        |

---

## 3. Policy Constants

**Module**: `logic/src/constants/policies.py`

### Policy Classification System

Policies are categorized into 4 groups based on their configuration requirements:

#### 1. Engine Policies

**Purpose**: Solvers with multiple backend engines (exact/metaheuristic)

```python
ENGINE_POLICIES = {
    "vrpp": ["gurobi", "hexaly"],
}
```

**Usage**:

```bash
# CLI syntax
python main.py test_sim --policies vrpp --engine gurobi
python main.py test_sim --policies vrpp --engine hexaly
```

**When to use**: Your policy supports swappable solver backends for the same problem.

#### 2. Threshold Policies

**Purpose**: Algorithms with tunable numeric parameters parsed from policy name

```python
THRESHOLD_POLICIES = [
    "vrpp",   # MIP gap tolerance (e.g., "vrpp_0.01" → 1% gap)
    "sans",   # Cooling rate (e.g., "sans_0.95")
    "hgs",    # Max iterations (e.g., "hgs_10000")
    "alns",   # Max iterations (e.g., "alns_5000")
    "bpc",    # Time limit (e.g., "bpc_300" → 300 seconds)
]
```

**Naming Convention**: `{policy}_{threshold}`

**Usage**:

```python
# In logic/src/policies/__init__.py:get_adapter()
policy_name = "hgs_10000"
# Parses to: base_policy="hgs", threshold=10000.0
```

**When to use**: Policy has a single tunable numeric parameter (iterations, temperature, time limit).

#### 3. Config Char Policies

**Purpose**: Policies with named variants using alphabetic suffixes

```python
CONFIG_CHAR_POLICIES = {
    "lac": ["a", "b"],  # LAC variant 'a' (conservative) vs 'b' (aggressive)
}
```

**Naming Convention**: `{policy}_{char}_{threshold}`

**Example**:

- `lac_a_1.0` → LAC variant 'a' with threshold 1.0
- `lac_b_2.0` → LAC variant 'b' with threshold 2.0

**When to use**: Multiple named variants of the same algorithm with different hyperparameter profiles.

#### 4. Simple Policies

**Purpose**: Fixed-configuration policies with direct name mapping

```python
SIMPLE_POLICIES = {
    # Neural models → "neural" config
    ("am", "ddam", "transgcn"): "neural",

    # Selection strategies → direct mapping
    ("last_minute",): "last_minute",
    ("regular",): "regular",

    # Classical solvers → direct mapping
    ("bpc",): "bpc",
    ("lkh",): "lkh",
    ("tsp",): "tsp",
    ("cvrp",): "cvrp",
}
```

**Mapping**: `(alias1, alias2, ...) → config_file_name`

**Config Resolution**:

```
Policy "am" → assets/configs/policies/policy_neural.yaml
Policy "last_minute" → assets/configs/policies/policy_last_minute.yaml
```

**When to use**: Policy has fixed configuration with no runtime parameter tuning.

### Configuration File Resolution Example

For policy named `"sans_0.95"`:

1. Check `THRESHOLD_POLICIES` → `"sans"` found
2. Parse threshold: `0.95`
3. Load config: `assets/configs/policies/policy_sans.yaml`
4. Pass `threshold=0.95` to `PolicyConfig` constructor

For policy named `"last_minute"`:

1. Check `SIMPLE_POLICIES` → `("last_minute",)` found
2. Load config: `assets/configs/policies/policy_last_minute.yaml`
3. No threshold parsing needed

---

## 4. Metric Constants

**Module**: `logic/src/constants/metrics.py`

### Metric Name Aliasing

Unifies metric names across RL training, simulation, evaluation, and GUI subsystems.

```python
METRIC_MAPPING = {
    "collection": [
        "reward_waste",      # RL reward component
        "collection",        # Generic charts/tables
        "total_collected",   # Cumulative episode collection
        "collected_waste",   # Explicit waste (kg)
        "real_collection"    # Simulator tracking
    ],
    "cost": [
        "reward_cost",       # RL reward component (negative)
        "cost",              # Generic cost term
        "tour_length",       # Route distance (km)
        "total_cost"         # Cumulative episode cost
    ],
    "overflows": [
        "reward_overflow",   # RL penalty component
        "overflows",         # Generic overflow count
        "real_overflows"     # Simulator tracking
    ],
    "initial_overflows": [
        "cur_overflows",     # Current at initialization
        "reset_overflows"    # Captured during env reset
    ],
}
```

### Usage Context

#### 1. RL Training Pipeline

**File**: `logic/src/pipeline/rl/core/*.py`

```python
from logic.src.constants import METRIC_MAPPING

# Normalize reward component keys
for canonical, aliases in METRIC_MAPPING.items():
    if reward_key in aliases:
        normalized_key = canonical
```

#### 2. Simulation Analyzer

**File**: `logic/src/pipeline/simulations/actions/logging.py`

```python
# Unify metrics from different policies
# Classical: "real_collection", Neural: "collected_waste"
# → Both map to canonical "collection"
```

#### 3. Studio Visualization

**File**: `app/src/pages/analysis/*.tsx`

```python
# Consistent axis labels
label = f"Collection (kg)"  # User sees consistent names
# Internal: chart queries any alias in METRIC_MAPPING["collection"]
```

#### 4. Data Analysis

**File**: `notebooks/*.ipynb`

```python
# Simplify querying heterogeneous experiment logs
df[METRIC_MAPPING["cost"]]  # Captures all cost variants
```

### Ordering Convention

Aliases are ordered by usage frequency:

1. **First**: Most common name (typically RL training)
2. **Middle**: Alternative names from different subsystems
3. **Last**: Legacy names (backward compatibility)

---

## 5. Simulation Constants

**Module**: `logic/src/constants/simulation.py`

### Physical Constants

#### Earth Radius Models

**Spherical Approximation** (faster, ±0.5% error):

```python
EARTH_RADIUS: int = 6371  # km (mean radius)
```

**Use when**: Speed > precision (data generation, prototypes)

**WGS84 Ellipsoid** (sub-millimeter precision):

```python
EARTH_WMP_RADIUS: int = 6378137  # meters (equatorial radius)
```

**Use when**: Real-world GPS data (OpenStreetMap, Google Maps integration)

**Formula**:

```python
# Haversine distance
distance = 2 * EARTH_RADIUS * arcsin(√(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2)))
```

### Performance Metrics

#### Core Metrics (Daily + Overall)

```python
METRICS: List[str] = [
    "overflows",  # Bins exceeding capacity (integer count)
    "kg",         # Waste collected (float, kilograms)
    "ncol",       # Collection count (integer)
    "kg_lost",    # Waste lost to overflows (float, kg)
    "km",         # Distance traveled (float, kilometers)
    "kg/km",      # Collection efficiency (float, ratio)
    "cost",       # Operational cost (float, currency units)
    "profit",     # Net profit (float, revenue - cost)
]
```

**Units Specification**:

| Metric      | Type  | Unit       | Description                              |
| ----------- | ----- | ---------- | ---------------------------------------- |
| `overflows` | int   | count      | Constraint violations (minimize to 0)    |
| `kg`        | float | kilograms  | Waste collected (maximize, no overflows) |
| `ncol`      | int   | count      | Collection operations (minimize)         |
| `kg_lost`   | float | kilograms  | Overflow penalty (minimize to 0)         |
| `km`        | float | kilometers | Distance traveled (minimize)             |
| `kg/km`     | float | kg/km      | Efficiency ratio (maximize)              |
| `cost`      | float | currency   | Operational cost (minimize)              |
| `profit`    | float | currency   | Net profit (maximize, primary objective) |

#### Extended Metrics

```python
# Simulation metadata
SIM_METRICS: List[str] = METRICS + ["days", "time"]
# - days: int (simulation duration, e.g., 31)
# - time: float (wall-clock seconds)

# Daily log fields
DAY_METRICS: List[str] = ["day"] + METRICS + ["tour"]
# - day: int (day number, 1-indexed)
# - tour: List[int] (node sequence, e.g., [0, 5, 12, 8, 0])
```

#### Training Loss Components

```python
LOSS_KEYS: List[str] = ["nll", "reinforce_loss", "baseline_loss"]
```

**Usage**: RL training loops, logged to WandB/TensorBoard

| Key              | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `nll`            | Negative log-likelihood of actions (policy gradient loss) |
| `reinforce_loss` | REINFORCE loss with baseline                              |
| `baseline_loss`  | Value network MSE loss (critic training)                  |

### Problem Constraints

#### Bin Capacity

```python
MAX_WASTE: float = 1.0  # Normalized (100% capacity)
```

**Usage**: Overflow detection, reward penalty calculation

**Range**: [0.0, 1.0]
**Overflow**: Occurs when `fill > MAX_WASTE` after waste generation

#### Vehicle Capacity

```python
VEHICLE_CAPACITY: float = 200.0  # kg (default for synthetic instances)
```

**Usage**: Capacitated VRP variants (CVRP, CWCVRP, SCWCVRP)

**Route Termination**: When `cumulative_collected >= VEHICLE_CAPACITY`

**Real-world values**:

- Small trucks: 80-120 kg
- Large trucks: 200-300 kg

#### Route Length Limits

```python
MAX_LENGTHS: Dict[int, int] = {
    20: 2,   # 20 customers → max 2 node visits
    50: 3,   # 50 customers → max 3 node visits
    100: 4,  # 100 customers → max 4 node visits
    150: 5,  # 150 customers → max 5 node visits
    225: 6,  # 225 customers → max 6 node visits
    317: 7,  # 317 customers → max 7 node visits
}
```

**Purpose**: Prevent unbounded routes in waste-collecting problems

**Rationale**: Larger instances need proportionally longer routes (√n heuristic)

**Usage**: VRPP, CVRPP environments enforce these limits

### Supported Problems

```python
PROBLEMS: List[str] = [
    "vrpp",     # Vehicle Routing Problem with Profits
    "cvrpp",    # Capacitated VRPP
    "wcvrp",    # Waste Collection VRP (dynamic bins, no capacity)
    "cwcvrp",   # Capacitated WCVRP (standard WSmart+ problem)
    "scwcvrp",  # Stochastic Capacitated WCVRP
]
```

**Mapping**: `{problem_name}` → `logic/src/envs/{problem_name}.py`

---

## 6. Routing Constants

**Module**: `logic/src/constants/routing.py`

### Local Search Parameters

```python
IMPROVEMENT_EPSILON: float = 1e-3  # 0.1% of typical route cost
```

**Purpose**: Minimum cost improvement to accept local search move

**Prevents**: Cycling on near-equal solutions

**Usage**: 2-opt, swap, relocate operators

**Tuning**:

- Smaller (1e-6): More thorough, slower
- Larger (1e-2): Faster, may miss improvements

### Real-World Operational Constraints

#### Collection Time

```python
COLLECTION_TIME_MINUTES = 3.0  # minutes per bin
```

**Includes**: Approach, emptying, compaction, departure

**Real-world range**: 2-5 minutes (depends on bin type)

#### Vehicle Speed

```python
VEHICLE_SPEED_KMH = 40.0  # km/h
```

**Context**: Urban waste collection with traffic, turns, narrow streets

**Real-world range**: 30-50 km/h (not highway speed)

### Optimization Penalties

```python
PENALTY_MUST_GO_MISSED = 10000.0  # cost units
```

**Purpose**: Ensure mandatory bins are never skipped

**Magnitude**: Should be >> typical route cost (50-200 units)

```python
MAX_CAPACITY_PERCENT = 100.0  # percent (0-100 range)
```

**Usage**: Overflow detection, capacity feasibility checks

**Note**: Duplicates `simulation.MAX_WASTE` (1.0) in different units. Consider consolidating.

### Gurobi MIP Parameters

```python
MIP_GAP = 0.01  # 1% optimality gap
```

**Solver Termination**: When `(best_bound - incumbent) / incumbent ≤ 0.01`

**Trade-off**: Smaller gap = longer runtime, better solution quality

**Industry Standard**: 1% for VRP-class problems

```python
HEURISTICS_RATIO = 0.5  # balanced
```

**Range**: [0.0, 1.0]

- `0.0`: No heuristics (pure branch-and-bound)
- `0.5`: Balanced (default)
- `1.0`: Maximum heuristics

```python
NODEFILE_START_GB = 0.5  # GB
```

**Purpose**: When memory usage exceeds this, Gurobi writes nodes to disk (slower)

**Tuning**: Increase for large instances on high-RAM machines

```python
SOLVER_OUTPUT_FLAG = 0  # suppress output
```

**Values**: `0` = silent, `1` = enable solver output

### Simulated Annealing (SANS) Defaults

```python
DEFAULT_SHIFT_DURATION = 390  # minutes (6.5 hours)
```

**Context**: Typical waste collection shift: 6-8 hours

```python
DEFAULT_V_VALUE = 1.0  # weight
```

**Purpose**: Tradeoff between collection cost and overflow risk

```python
DEFAULT_COMBINATION = [500, 75, 0.95, 0, 0.095, 0, 0]
```

**Format**: `[max_iter, beam_width, cooling_rate, init_temp, pert_strength, ...]`

**Source**: OG SANS hyperparameters (7-tuple, see SANS paper)

```python
DEFAULT_TIME_LIMIT = 600  # seconds (10 minutes)
```

**Purpose**: Total optimization time before returning best solution

### Batch Size Defaults

```python
DEFAULT_EVAL_BATCH_SIZE: int = 1024  # instances per batch
```

**Usage**: HRL policy evaluation (`logic/src/models/hrl_manager/manager.py`)

**GPU Memory**: Optimal for 12GB VRAM; reduce to 512 for 8GB

```python
DEFAULT_ROLLOUT_BATCH_SIZE: int = 64  # episodes per rollout
```

**Usage**: REINFORCE/PPO baseline computation

**Rationale**: Smaller than eval batch (rollouts compute full episodes)

---

## 7. Model Constants

**Module**: `logic/src/constants/models.py`

### Problem Size Naming Conventions

**CRITICAL**: Three variable names describe "number of nodes" with distinct semantics. **DO NOT** rename one to another:

#### `num_loc` (Customer Locations)

**Excludes**: Depot
**Usage**: Configs (`EnvConfig`), generators, environments, CLI
**Example**: `num_loc = 50` → 50 customers

```python
# Config
env = EnvConfig(num_loc=50)

# Generator
dataset = generate_instances(num_loc=50)
```

#### `graph_size` (Total Nodes)

**Includes**: Depot
**Usage**: Model forward passes, encoder/decoder, simulation context
**Relationship**: `graph_size = num_loc + 1`

```python
# Model forward pass
def forward(self, x):
    batch_size, graph_size, embed_dim = x.size()
    # graph_size includes depot
```

#### `n_nodes` (Customer Nodes in Solvers)

**Excludes**: Depot
**Usage**: Classical solvers (HGS, ALNS, BPC) - local/instance variables
**Computed**: `n_nodes = len(dist_matrix) - 1`

```python
# Classical policy
def solve(self, dist_matrix):
    n_nodes = len(dist_matrix) - 1  # Exclude depot
```

### Propagation Flow

```
Config (num_loc)
  ↓
Generator (num_loc)
  ↓
Environment (num_loc)
  ↓
Model tensors (graph_size = num_loc + 1)
  ↓
Classical policies (n_nodes = num_loc)
  ↓
Simulator context (graph_size)
```

### Model Architecture Registries

```python
SUB_NET_ENCS: List[str] = ["tgc"]
```

**Purpose**: Models requiring sub-network encoders (edge feature encoding)

**Usage**: `model_factory.py` instantiates `encoder.sub_network`

**Example**: TGC (Transformer Graph Convolution) needs edge features

```python
PRED_ENC_MODELS: List[str] = ["tam"]
```

**Purpose**: Models with predictive encoders (future state prediction)

**Usage**: `model_factory.py` instantiates predictor networks

**Example**: TAM (Temporal Attention Model) predicts future bin fill levels

```python
ENC_DEC_MODELS: List[str] = ["ddam"]
```

**Purpose**: Models with separate encoder-decoder architecture

**Usage**: `model_factory.py` instantiates both encoder and decoder separately

**Example**: DDAM (Deep Decoder Attention Model) has deep transformer decoder

### Feature Dimensions

```python
NODE_DIM: int = 3  # [x, y, waste]
```

**Usage**: Embedding layers, input projections

**Components**: Coordinates (2D) + Node attribute (1D)

```python
STATIC_DIM: int = 2  # [x, y]
```

**Usage**: Distance matrix computation, spatial encoders

**Range**: [0, 1] (normalized Euclidean coordinates)

```python
DEPOT_DIM: int = 2  # [x, y]
```

**Usage**: Depot-specific encoding, route start/end embeddings

**Note**: Same as `STATIC_DIM`, kept separate for semantic clarity

### Step Context Dimensions

```python
WC_STEP_CONTEXT_OFFSET: int = 2  # [current_capacity, remaining_capacity]
```

**Usage**: `wcvrp.py` state embeddings for capacity-aware decoding

```python
VRPP_STEP_CONTEXT_OFFSET: int = 1  # [collected_profit]
```

**Usage**: `vrpp.py` state embeddings for profit-aware decoding

**Computation**: `total_context_dim = NODE_DIM + offset`

### Temporal Defaults

```python
DEFAULT_TEMPORAL_HORIZON: int = 10  # Days
```

**Usage**: TAM, HRL Manager network

**Context**: Typical waste collection planning horizon is 7-14 days

### Architecture Hyperparameters

```python
TANH_CLIPPING: float = 10.0
```

**Purpose**: Prevents `exp()` overflow in softmax

**Formula**: `logits = tanh_clipping * tanh(logits / tanh_clipping)`

**Source**: Kool et al. (2019) - standard for VRP models

```python
NORM_EPSILON: float = 1e-5
```

**Purpose**: Layer/batch/instance normalization stability

**Usage**: Prevents division by zero in variance denominator

**Default**: Same as PyTorch `nn.LayerNorm`

```python
NUMERICAL_EPSILON: float = 1e-8
```

**Purpose**: General computation stability

**Usage**: Probability clamping, division safety, `kg/km` efficiency

**Rationale**: Smaller than `NORM_EPSILON` to minimize impact on distributions

```python
FEED_FORWARD_EXPANSION: int = 4
```

**Formula**: `hidden_dim = embed_dim * 4`

**Source**: Vaswani et al. (2017) "Attention is All You Need"

**Usage**: `feed_forward.py` module instantiation

### Mixture of Experts Defaults

```python
DEFAULT_MOE_KWARGS = {
    "encoder": {
        "hidden_act": "ReLU",      # Expert activation function
        "num_experts": 4,           # Number of expert networks
        "k": 2,                     # Top-k routing (activate best 2)
        "noisy_gating": True,       # Exploration noise
    },
    "decoder": {
        "light_version": True,      # Lightweight MoE (shared routing)
        "num_experts": 4,
        "k": 2,
        "noisy_gating": True,
    },
}
```

**Usage**: `logic/src/models/moe_model.py` when config omits MoE parameters

**Purpose**: Model specialization - different experts for different instances

**Trade-off**: More experts = higher capacity, slower inference

---

## 8. Waste Management Constants

**Module**: `logic/src/constants/waste.py`

### Geographic Mappings

```python
MAP_DEPOTS: Dict[str, str] = {
    "mixrmbac": "CTEASO",      # Multi-municipality dataset
    "riomaior": "CTEASO",      # Rio Maior (central Portugal)
    "figueiradafoz": "CITVRSU", # Figueira da Foz (coastal)
}
```

**Context**: Real municipalities in Portugal served by WSmart+ system

**Depot Codes**:

- **CTEASO**: Centro de Triagem e Estação de RSU (Waste Sorting Center)
- **CITVRSU**: Centro Integrado de Tratamento e Valorização (Recovery Center)

**Usage**: `logic/src/pipeline/simulations/loader.py` for area-specific data

### Waste Type Classification

```python
WASTE_TYPES: Dict[str, str] = {
    "glass": "Embalagens de Vidro",              # Glass packaging (green bins)
    "plastic": "Mistura de embalagens",          # Mixed packaging (yellow bins)
    "paper": "Embalagens de papel e cartão",     # Paper/cardboard (blue bins)
}
```

**Authority**: Agência Portuguesa do Ambiente (APA) official nomenclature

**Context**: Selective waste collection (recycling), not mixed waste

**Characteristics**:

| Type    | Bin Color | Collection Frequency | Revenue |
| ------- | --------- | -------------------- | ------- |
| Glass   | Green     | Biweekly             | Lowest  |
| Plastic | Yellow    | Weekly               | Highest |
| Paper   | Blue      | Weekly               | Medium  |

**Usage**: Data loading, report generation, GUI labels, notebooks

### Critical Fill Threshold

```python
CRITICAL_FILL_THRESHOLD: float = 0.9  # 90% capacity
```

**Purpose**: Fill level triggering priority collection

**Usage**:

- mandatory bin selection (bins ≥ 0.9 flagged as mandatory)
- Service Level Agreement (SLA) compliance
- Look-ahead search (preventive collection)

**Industry Standard**: 0.8-0.9 (WSmart+ uses 0.9 for cost-overflow balance)

---

## 9. HPO Constants

**Module**: `logic/src/constants/hpo.py`

### Configuration Keys

```python
HOP_KEYS: Tuple[str, ...] = (
    # Core HPO settings
    "hpo_method",     # Algorithm: "optuna", "dehb", "ray", "grid"
    "hpo_range",      # Parameter search space
    "hpo_epochs",     # Training epochs per trial
    "metric",         # Objective: "cost", "profit", "kg/km"
    "cpu_cores",      # Parallel workers
    "verbose",        # Logging verbosity (0=silent, 1=progress, 2=debug)
    "train_best",     # Retrain best config on full data
    "local_mode",     # Run Ray Tune locally vs distributed

    # Optuna-specific
    "n_trials",           # Total optimization trials
    "timeout",            # Max HPO time (seconds)
    "n_startup_trials",   # Random trials before Bayesian optimization
    "n_warmup_steps",     # Steps before pruner starts
    "interval_steps",     # Pruning check frequency

    # DEHB-specific
    "eta",                # Successive halving reduction (default: 3)
    "max_tres",           # Max resources per trial
    "reduction_factor",   # Budget reduction (HyperBand)
    "fevals",             # Total function evaluations

    # Evolutionary (NSGA-II, NSGA-III)
    "indpb",      # Independent mutation probability
    "tournsize",  # Tournament selection size (higher = more elitism)
    "cxpb",       # Crossover probability
    "mutpb",      # Mutation probability
    "n_pop",      # Population size
    "n_gen",      # Number of generations

    # Ray Tune-specific
    "num_samples",   # Total trials
    "max_failures",  # Fault tolerance
    "max_conc",      # Max concurrent trials

    # Grid search
    "grid",  # Parameter grid definition
)
```

### Supported Methods

| Method   | Algorithm                                    | Use Case                                 |
| -------- | -------------------------------------------- | ---------------------------------------- |
| `optuna` | Bayesian optimization (TPE, NSGA-II, CMA-ES) | General-purpose, multi-objective         |
| `dehb`   | Differential Evolution Hyperband             | Efficient multi-fidelity optimization    |
| `ray`    | Distributed HPO (ASHA, HyperBand, Bayesian)  | Large-scale, cluster computing           |
| `grid`   | Exhaustive grid search                       | Small search spaces, baseline comparison |

### Usage Example

```yaml
# assets/configs/tasks/hpo.yaml
hpo_method: optuna
n_trials: 100
timeout: 3600 # 1 hour
metric: profit
hpo_range:
  learning_rate: [1e-5, 1e-3, log]
  batch_size: [64, 128, 256]
```

**Validation**: Missing keys trigger default value fallbacks in `logic/src/configs/hpo.py`

**CLI Integration**: Keys map to `logic/controllers/cli/ts_parser.py` arguments

---

## 10. User Interface Constants

**Module**: `logic/src/constants/user_interface.py`

### Terminal Output

```python
PBAR_WAIT_TIME: float = 0.1  # seconds (100ms refresh interval)
```

**Purpose**: tqdm progress bar update rate

**Trade-off**: Lower = smoother (higher CPU), higher = choppier (lower CPU)

**Default**: 10 Hz is smooth without CPU overhead

```python
TQDM_COLOURS = [
    "red",      # Worker 0 (or error indicator)
    "blue",     # Worker 1
    "green",    # Worker 2 (success indicator)
    "yellow",   # Worker 3 (warning indicator)
    "magenta",  # Worker 4
    "cyan",     # Worker 5
]
```

**Usage**: Parallel simulation workers (cycles through colors when >6 workers)

**Note**: "red" reserved for errors; avoid using for normal progress

### Matplotlib Styling

```python
MARKERS: List[str] = ["P", "s", "^", "8", "*"]
```

**Shapes**: Pentagon, Square, Triangle, Octagon, Star

**Purpose**: Visual distinction in scatter/line plots

```python
PLOT_COLOURS: List[str] = [
    "red",     # Policy 0 or worst-case
    "blue",    # Policy 1 or baseline
    "green",   # Policy 2 or best-case
    "yellow",  # Policy 3 (caution on white backgrounds)
    "magenta", # Policy 4
    "cyan",    # Policy 5
    "black",   # Reference lines, grid, text
]
```

**Palette**: ANSI 6-color + black

**Usage**: Multi-policy comparisons, Pareto fronts, time series

```python
LINESTYLES: List[Union[str, Tuple[int, Tuple[int, ...]]]] = [
    "dotted",               # · · · ·
    "dashed",               # - - - -
    "dashdot",              # -·-·-·-·
    (0, (3, 5, 1, 5, 1, 5)), # Custom: dash-dot-dot
    "solid",                # ________
]
```

**Purpose**: Distinguish >6 series (6 colors × 5 linestyles = 30 unique)

### Design Principles

1. **Immutability**: Use tuples and uppercase naming for true constants
2. **Documentation**: Each constant includes usage context and valid ranges
3. **Deprecation Safety**: Legacy constants retained with migration warnings
4. **Type Safety**: Type hints for all mappings and function registries

### Import Strategy

```python
# Import from top-level module (re-exports all sub-modules)
from logic.src.constants import MAX_WASTE, VEHICLE_CAPACITY, METRICS

# Or import specific sub-module for clarity
from logic.src.constants.simulation import MAX_WASTE, VEHICLE_CAPACITY
from logic.src.constants.policies import SIMPLE_POLICIES, THRESHOLD_POLICIES
```

---

## 2. Module Organization

The constants module is split into 15 sub-modules for maintainability:

| File                | Category                           | Primary Use Cases                                |
| ------------------- | ---------------------------------- | ------------------------------------------------ |
| `policies.py`       | Policy naming & classification     | Policy factory, CLI parsing, config resolution   |
| `metrics.py`        | Metric name aliasing               | RL training, logging normalization, GUI charts   |
| `simulation.py`     | Physics & problem constraints      | Environment definitions, simulation orchestrator |
| `routing.py`        | Solver parameters & penalties      | Classical algorithms (HGS, ALNS, Gurobi)         |
| `models.py`         | Architecture hyperparameters       | Model factory, encoder/decoder instantiation     |
| `waste.py`          | Real-world waste management data   | Portugal case studies, empirical distributions   |
| `hpo.py`            | Hyperparameter optimization config | Optuna, DEHB, Ray Tune integration               |
| `user_interface.py` | Visual styling                     | CLI output, matplotlib plots, GUI themes         |
| `system.py`         | Infrastructure & operations        | File system, operator evaluation, threading      |
| `paths.py`          | Project directory resolution       | Asset loading, workspace detection               |
| `testing.py`        | Test suite registry                | pytest integration, CI/CD pipelines              |
| `stats.py`          | Statistical function mapping       | Data analysis, summary computation               |
| `dashboard.py`      | Visualization color schemes        | Simulation dashboard, map rendering              |
| `tasks.py`          | **DEPRECATED**                     | Legacy constants (backward compatibility)        |

---

## 3. Policy Constants

**Module**: `logic/src/constants/policies.py`

### Policy Classification System

Policies are categorized into 4 groups based on their configuration requirements:

#### 1. Engine Policies

**Purpose**: Solvers with multiple backend engines (exact/metaheuristic)

```python
ENGINE_POLICIES = {
    "vrpp": ["gurobi", "hexaly"],
}
```

**Usage**:

```bash
# CLI syntax
python main.py test_sim --policies vrpp --engine gurobi
python main.py test_sim --policies vrpp --engine hexaly
```

**When to use**: Your policy supports swappable solver backends for the same problem.

#### 2. Threshold Policies

**Purpose**: Algorithms with tunable numeric parameters parsed from policy name

```python
THRESHOLD_POLICIES = [
    "vrpp",   # MIP gap tolerance (e.g., "vrpp_0.01" → 1% gap)
    "sans",   # Cooling rate (e.g., "sans_0.95")
    "hgs",    # Max iterations (e.g., "hgs_10000")
    "alns",   # Max iterations (e.g., "alns_5000")
    "bpc",    # Time limit (e.g., "bpc_300" → 300 seconds)
]
```

**Naming Convention**: `{policy}_{threshold}`

**Usage**:

```python
# In logic/src/policies/__init__.py:get_adapter()
policy_name = "hgs_10000"
# Parses to: base_policy="hgs", threshold=10000.0
```

**When to use**: Policy has a single tunable numeric parameter (iterations, temperature, time limit).

#### 3. Config Char Policies

**Purpose**: Policies with named variants using alphabetic suffixes

```python
CONFIG_CHAR_POLICIES = {
    "lac": ["a", "b"],  # LAC variant 'a' (conservative) vs 'b' (aggressive)
}
```

**Naming Convention**: `{policy}_{char}_{threshold}`

**Example**:

- `lac_a_1.0` → LAC variant 'a' with threshold 1.0
- `lac_b_2.0` → LAC variant 'b' with threshold 2.0

**When to use**: Multiple named variants of the same algorithm with different hyperparameter profiles.

#### 4. Simple Policies

**Purpose**: Fixed-configuration policies with direct name mapping

```python
SIMPLE_POLICIES = {
    # Neural models → "neural" config
    ("am", "ddam", "transgcn"): "neural",

    # Selection strategies → direct mapping
    ("last_minute",): "last_minute",
    ("regular",): "regular",

    # Classical solvers → direct mapping
    ("bpc",): "bpc",
    ("lkh",): "lkh",
    ("tsp",): "tsp",
    ("cvrp",): "cvrp",
}
```

**Mapping**: `(alias1, alias2, ...) → config_file_name`

**Config Resolution**:

```
Policy "am" → assets/configs/policies/policy_neural.yaml
Policy "last_minute" → assets/configs/policies/policy_last_minute.yaml
```

**When to use**: Policy has fixed configuration with no runtime parameter tuning.

### Configuration File Resolution Example

For policy named `"sans_0.95"`:

1. Check `THRESHOLD_POLICIES` → `"sans"` found
2. Parse threshold: `0.95`
3. Load config: `assets/configs/policies/policy_sans.yaml`
4. Pass `threshold=0.95` to `PolicyConfig` constructor

For policy named `"last_minute"`:

1. Check `SIMPLE_POLICIES` → `("last_minute",)` found
2. Load config: `assets/configs/policies/policy_last_minute.yaml`
3. No threshold parsing needed

---

## 4. Metric Constants

**Module**: `logic/src/constants/metrics.py`

### Metric Name Aliasing

Unifies metric names across RL training, simulation, evaluation, and GUI subsystems.

```python
METRIC_MAPPING = {
    "collection": [
        "reward_waste",      # RL reward component
        "collection",        # Generic charts/tables
        "total_collected",   # Cumulative episode collection
        "collected_waste",   # Explicit waste (kg)
        "real_collection"    # Simulator tracking
    ],
    "cost": [
        "reward_cost",       # RL reward component (negative)
        "cost",              # Generic cost term
        "tour_length",       # Route distance (km)
        "total_cost"         # Cumulative episode cost
    ],
    "overflows": [
        "reward_overflow",   # RL penalty component
        "overflows",         # Generic overflow count
        "real_overflows"     # Simulator tracking
    ],
    "initial_overflows": [
        "cur_overflows",     # Current at initialization
        "reset_overflows"    # Captured during env reset
    ],
}
```

### Usage Context

#### 1. RL Training Pipeline

**File**: `logic/src/pipeline/rl/core/*.py`

```python
from logic.src.constants import METRIC_MAPPING

# Normalize reward component keys
for canonical, aliases in METRIC_MAPPING.items():
    if reward_key in aliases:
        normalized_key = canonical
```

#### 2. Simulation Analyzer

**File**: `logic/src/pipeline/simulations/actions/logging.py`

```python
# Unify metrics from different policies
# Classical: "real_collection", Neural: "collected_waste"
# → Both map to canonical "collection"
```

#### 3. Studio Visualization

**File**: `app/src/pages/analysis/*.tsx`

```python
# Consistent axis labels
label = f"Collection (kg)"  # User sees consistent names
# Internal: chart queries any alias in METRIC_MAPPING["collection"]
```

#### 4. Data Analysis

**File**: `notebooks/*.ipynb`

```python
# Simplify querying heterogeneous experiment logs
df[METRIC_MAPPING["cost"]]  # Captures all cost variants
```

### Ordering Convention

Aliases are ordered by usage frequency:

1. **First**: Most common name (typically RL training)
2. **Middle**: Alternative names from different subsystems
3. **Last**: Legacy names (backward compatibility)

---

## 5. Simulation Constants

**Module**: `logic/src/constants/simulation.py`

### Physical Constants

#### Earth Radius Models

**Spherical Approximation** (faster, ±0.5% error):

```python
EARTH_RADIUS: int = 6371  # km (mean radius)
```

**Use when**: Speed > precision (data generation, prototypes)

**WGS84 Ellipsoid** (sub-millimeter precision):

```python
EARTH_WMP_RADIUS: int = 6378137  # meters (equatorial radius)
```

**Use when**: Real-world GPS data (OpenStreetMap, Google Maps integration)

**Formula**:

```python
# Haversine distance
distance = 2 * EARTH_RADIUS * arcsin(√(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2)))
```

### Performance Metrics

#### Core Metrics (Daily + Overall)

```python
METRICS: List[str] = [
    "overflows",  # Bins exceeding capacity (integer count)
    "kg",         # Waste collected (float, kilograms)
    "ncol",       # Collection count (integer)
    "kg_lost",    # Waste lost to overflows (float, kg)
    "km",         # Distance traveled (float, kilometers)
    "kg/km",      # Collection efficiency (float, ratio)
    "cost",       # Operational cost (float, currency units)
    "profit",     # Net profit (float, revenue - cost)
]
```

**Units Specification**:

| Metric      | Type  | Unit       | Description                              |
| ----------- | ----- | ---------- | ---------------------------------------- |
| `overflows` | int   | count      | Constraint violations (minimize to 0)    |
| `kg`        | float | kilograms  | Waste collected (maximize, no overflows) |
| `ncol`      | int   | count      | Collection operations (minimize)         |
| `kg_lost`   | float | kilograms  | Overflow penalty (minimize to 0)         |
| `km`        | float | kilometers | Distance traveled (minimize)             |
| `kg/km`     | float | kg/km      | Efficiency ratio (maximize)              |
| `cost`      | float | currency   | Operational cost (minimize)              |
| `profit`    | float | currency   | Net profit (maximize, primary objective) |

#### Extended Metrics

```python
# Simulation metadata
SIM_METRICS: List[str] = METRICS + ["days", "time"]
# - days: int (simulation duration, e.g., 31)
# - time: float (wall-clock seconds)

# Daily log fields
DAY_METRICS: List[str] = ["day"] + METRICS + ["tour"]
# - day: int (day number, 1-indexed)
# - tour: List[int] (node sequence, e.g., [0, 5, 12, 8, 0])
```

#### Training Loss Components

```python
LOSS_KEYS: List[str] = ["nll", "reinforce_loss", "baseline_loss"]
```

**Usage**: RL training loops, logged to WandB/TensorBoard

| Key              | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `nll`            | Negative log-likelihood of actions (policy gradient loss) |
| `reinforce_loss` | REINFORCE loss with baseline                              |
| `baseline_loss`  | Value network MSE loss (critic training)                  |

### Problem Constraints

#### Bin Capacity

```python
MAX_WASTE: float = 1.0  # Normalized (100% capacity)
```

**Usage**: Overflow detection, reward penalty calculation

**Range**: [0.0, 1.0]
**Overflow**: Occurs when `fill > MAX_WASTE` after waste generation

#### Vehicle Capacity

```python
VEHICLE_CAPACITY: float = 200.0  # kg (default for synthetic instances)
```

**Usage**: Capacitated VRP variants (CVRP, CWCVRP, SCWCVRP)

**Route Termination**: When `cumulative_collected >= VEHICLE_CAPACITY`

**Real-world values**:

- Small trucks: 80-120 kg
- Large trucks: 200-300 kg

#### Route Length Limits

```python
MAX_LENGTHS: Dict[int, int] = {
    20: 2,   # 20 customers → max 2 node visits
    50: 3,   # 50 customers → max 3 node visits
    100: 4,  # 100 customers → max 4 node visits
    150: 5,  # 150 customers → max 5 node visits
    225: 6,  # 225 customers → max 6 node visits
    317: 7,  # 317 customers → max 7 node visits
}
```

**Purpose**: Prevent unbounded routes in waste-collecting problems

**Rationale**: Larger instances need proportionally longer routes (√n heuristic)

**Usage**: VRPP, CVRPP environments enforce these limits

### Supported Problems

```python
PROBLEMS: List[str] = [
    "vrpp",     # Vehicle Routing Problem with Profits
    "cvrpp",    # Capacitated VRPP
    "wcvrp",    # Waste Collection VRP (dynamic bins, no capacity)
    "cwcvrp",   # Capacitated WCVRP (standard WSmart+ problem)
    "scwcvrp",  # Stochastic Capacitated WCVRP
]
```

**Mapping**: `{problem_name}` → `logic/src/envs/{problem_name}.py`

---

## 6. Routing Constants

**Module**: `logic/src/constants/routing.py`

### Local Search Parameters

```python
IMPROVEMENT_EPSILON: float = 1e-3  # 0.1% of typical route cost
```

**Purpose**: Minimum cost improvement to accept local search move

**Prevents**: Cycling on near-equal solutions

**Usage**: 2-opt, swap, relocate operators

**Tuning**:

- Smaller (1e-6): More thorough, slower
- Larger (1e-2): Faster, may miss improvements

### Real-World Operational Constraints

#### Collection Time

```python
COLLECTION_TIME_MINUTES = 3.0  # minutes per bin
```

**Includes**: Approach, emptying, compaction, departure

**Real-world range**: 2-5 minutes (depends on bin type)

#### Vehicle Speed

```python
VEHICLE_SPEED_KMH = 40.0  # km/h
```

**Context**: Urban waste collection with traffic, turns, narrow streets

**Real-world range**: 30-50 km/h (not highway speed)

### Optimization Penalties

```python
PENALTY_MUST_GO_MISSED = 10000.0  # cost units
```

**Purpose**: Ensure mandatory bins are never skipped

**Magnitude**: Should be >> typical route cost (50-200 units)

```python
MAX_CAPACITY_PERCENT = 100.0  # percent (0-100 range)
```

**Usage**: Overflow detection, capacity feasibility checks

**Note**: Duplicates `simulation.MAX_WASTE` (1.0) in different units. Consider consolidating.

### Gurobi MIP Parameters

```python
MIP_GAP = 0.01  # 1% optimality gap
```

**Solver Termination**: When `(best_bound - incumbent) / incumbent ≤ 0.01`

**Trade-off**: Smaller gap = longer runtime, better solution quality

**Industry Standard**: 1% for VRP-class problems

```python
HEURISTICS_RATIO = 0.5  # balanced
```

**Range**: [0.0, 1.0]

- `0.0`: No heuristics (pure branch-and-bound)
- `0.5`: Balanced (default)
- `1.0`: Maximum heuristics

```python
NODEFILE_START_GB = 0.5  # GB
```

**Purpose**: When memory usage exceeds this, Gurobi writes nodes to disk (slower)

**Tuning**: Increase for large instances on high-RAM machines

```python
SOLVER_OUTPUT_FLAG = 0  # suppress output
```

**Values**: `0` = silent, `1` = enable solver output

### Simulated Annealing (SANS) Defaults

```python
DEFAULT_SHIFT_DURATION = 390  # minutes (6.5 hours)
```

**Context**: Typical waste collection shift: 6-8 hours

```python
DEFAULT_V_VALUE = 1.0  # weight
```

**Purpose**: Tradeoff between collection cost and overflow risk

```python
DEFAULT_COMBINATION = [500, 75, 0.95, 0, 0.095, 0, 0]
```

**Format**: `[max_iter, beam_width, cooling_rate, init_temp, pert_strength, ...]`

**Source**: OG SANS hyperparameters (7-tuple, see SANS paper)

```python
DEFAULT_TIME_LIMIT = 600  # seconds (10 minutes)
```

**Purpose**: Total optimization time before returning best solution

### Batch Size Defaults

```python
DEFAULT_EVAL_BATCH_SIZE: int = 1024  # instances per batch
```

**Usage**: HRL policy evaluation (`logic/src/models/hrl_manager/manager.py`)

**GPU Memory**: Optimal for 12GB VRAM; reduce to 512 for 8GB

```python
DEFAULT_ROLLOUT_BATCH_SIZE: int = 64  # episodes per rollout
```

**Usage**: REINFORCE/PPO baseline computation

**Rationale**: Smaller than eval batch (rollouts compute full episodes)

---

## 7. Model Constants

**Module**: `logic/src/constants/models.py`

### Problem Size Naming Conventions

**CRITICAL**: Three variable names describe "number of nodes" with distinct semantics. **DO NOT** rename one to another:

#### `num_loc` (Customer Locations)

**Excludes**: Depot
**Usage**: Configs (`EnvConfig`), generators, environments, CLI
**Example**: `num_loc = 50` → 50 customers

```python
# Config
env = EnvConfig(num_loc=50)

# Generator
dataset = generate_instances(num_loc=50)
```

#### `graph_size` (Total Nodes)

**Includes**: Depot
**Usage**: Model forward passes, encoder/decoder, simulation context
**Relationship**: `graph_size = num_loc + 1`

```python
# Model forward pass
def forward(self, x):
    batch_size, graph_size, embed_dim = x.size()
    # graph_size includes depot
```

#### `n_nodes` (Customer Nodes in Solvers)

**Excludes**: Depot
**Usage**: Classical solvers (HGS, ALNS, BPC) - local/instance variables
**Computed**: `n_nodes = len(dist_matrix) - 1`

```python
# Classical policy
def solve(self, dist_matrix):
    n_nodes = len(dist_matrix) - 1  # Exclude depot
```

### Propagation Flow

```
Config (num_loc)
  ↓
Generator (num_loc)
  ↓
Environment (num_loc)
  ↓
Model tensors (graph_size = num_loc + 1)
  ↓
Classical policies (n_nodes = num_loc)
  ↓
Simulator context (graph_size)
```

### Model Architecture Registries

```python
SUB_NET_ENCS: List[str] = ["tgc"]
```

**Purpose**: Models requiring sub-network encoders (edge feature encoding)

**Usage**: `model_factory.py` instantiates `encoder.sub_network`

**Example**: TGC (Transformer Graph Convolution) needs edge features

```python
PRED_ENC_MODELS: List[str] = ["tam"]
```

**Purpose**: Models with predictive encoders (future state prediction)

**Usage**: `model_factory.py` instantiates predictor networks

**Example**: TAM (Temporal Attention Model) predicts future bin fill levels

```python
ENC_DEC_MODELS: List[str] = ["ddam"]
```

**Purpose**: Models with separate encoder-decoder architecture

**Usage**: `model_factory.py` instantiates both encoder and decoder separately

**Example**: DDAM (Deep Decoder Attention Model) has deep transformer decoder

### Feature Dimensions

```python
NODE_DIM: int = 3  # [x, y, waste]
```

**Usage**: Embedding layers, input projections

**Components**: Coordinates (2D) + Node attribute (1D)

```python
STATIC_DIM: int = 2  # [x, y]
```

**Usage**: Distance matrix computation, spatial encoders

**Range**: [0, 1] (normalized Euclidean coordinates)

```python
DEPOT_DIM: int = 2  # [x, y]
```

**Usage**: Depot-specific encoding, route start/end embeddings

**Note**: Same as `STATIC_DIM`, kept separate for semantic clarity

### Step Context Dimensions

```python
WC_STEP_CONTEXT_OFFSET: int = 2  # [current_capacity, remaining_capacity]
```

**Usage**: `wcvrp.py` state embeddings for capacity-aware decoding

```python
VRPP_STEP_CONTEXT_OFFSET: int = 1  # [collected_profit]
```

**Usage**: `vrpp.py` state embeddings for profit-aware decoding

**Computation**: `total_context_dim = NODE_DIM + offset`

### Temporal Defaults

```python
DEFAULT_TEMPORAL_HORIZON: int = 10  # Days
```

**Usage**: TAM, HRL Manager network

**Context**: Typical waste collection planning horizon is 7-14 days

### Architecture Hyperparameters

```python
TANH_CLIPPING: float = 10.0
```

**Purpose**: Prevents `exp()` overflow in softmax

**Formula**: `logits = tanh_clipping * tanh(logits / tanh_clipping)`

**Source**: Kool et al. (2019) - standard for VRP models

```python
NORM_EPSILON: float = 1e-5
```

**Purpose**: Layer/batch/instance normalization stability

**Usage**: Prevents division by zero in variance denominator

**Default**: Same as PyTorch `nn.LayerNorm`

```python
NUMERICAL_EPSILON: float = 1e-8
```

**Purpose**: General computation stability

**Usage**: Probability clamping, division safety, `kg/km` efficiency

**Rationale**: Smaller than `NORM_EPSILON` to minimize impact on distributions

```python
FEED_FORWARD_EXPANSION: int = 4
```

**Formula**: `hidden_dim = embed_dim * 4`

**Source**: Vaswani et al. (2017) "Attention is All You Need"

**Usage**: `feed_forward.py` module instantiation

### Mixture of Experts Defaults

```python
DEFAULT_MOE_KWARGS = {
    "encoder": {
        "hidden_act": "ReLU",      # Expert activation function
        "num_experts": 4,           # Number of expert networks
        "k": 2,                     # Top-k routing (activate best 2)
        "noisy_gating": True,       # Exploration noise
    },
    "decoder": {
        "light_version": True,      # Lightweight MoE (shared routing)
        "num_experts": 4,
        "k": 2,
        "noisy_gating": True,
    },
}
```

**Usage**: `logic/src/models/moe_model.py` when config omits MoE parameters

**Purpose**: Model specialization - different experts for different instances

**Trade-off**: More experts = higher capacity, slower inference

---

## 8. Waste Management Constants

**Module**: `logic/src/constants/waste.py`

### Geographic Mappings

```python
MAP_DEPOTS: Dict[str, str] = {
    "mixrmbac": "CTEASO",      # Multi-municipality dataset
    "riomaior": "CTEASO",      # Rio Maior (central Portugal)
    "figueiradafoz": "CITVRSU", # Figueira da Foz (coastal)
}
```

**Context**: Real municipalities in Portugal served by WSmart+ system

**Depot Codes**:

- **CTEASO**: Centro de Triagem e Estação de RSU (Waste Sorting Center)
- **CITVRSU**: Centro Integrado de Tratamento e Valorização (Recovery Center)

**Usage**: `logic/src/pipeline/simulations/loader.py` for area-specific data

### Waste Type Classification

```python
WASTE_TYPES: Dict[str, str] = {
    "glass": "Embalagens de Vidro",              # Glass packaging (green bins)
    "plastic": "Mistura de embalagens",          # Mixed packaging (yellow bins)
    "paper": "Embalagens de papel e cartão",     # Paper/cardboard (blue bins)
}
```

**Authority**: Agência Portuguesa do Ambiente (APA) official nomenclature

**Context**: Selective waste collection (recycling), not mixed waste

**Characteristics**:

| Type    | Bin Color | Collection Frequency | Revenue |
| ------- | --------- | -------------------- | ------- |
| Glass   | Green     | Biweekly             | Lowest  |
| Plastic | Yellow    | Weekly               | Highest |
| Paper   | Blue      | Weekly               | Medium  |

**Usage**: Data loading, report generation, GUI labels, notebooks

### Critical Fill Threshold

```python
CRITICAL_FILL_THRESHOLD: float = 0.9  # 90% capacity
```

**Purpose**: Fill level triggering priority collection

**Usage**:

- mandatory bin selection (bins ≥ 0.9 flagged as mandatory)
- Service Level Agreement (SLA) compliance
- Look-ahead search (preventive collection)

**Industry Standard**: 0.8-0.9 (WSmart+ uses 0.9 for cost-overflow balance)

---

## 9. HPO Constants

**Module**: `logic/src/constants/hpo.py`

### Configuration Keys

```python
HOP_KEYS: Tuple[str, ...] = (
    # Core HPO settings
    "hpo_method",     # Algorithm: "optuna", "dehb", "ray", "grid"
    "hpo_range",      # Parameter search space
    "hpo_epochs",     # Training epochs per trial
    "metric",         # Objective: "cost", "profit", "kg/km"
    "cpu_cores",      # Parallel workers
    "verbose",        # Logging verbosity (0=silent, 1=progress, 2=debug)
    "train_best",     # Retrain best config on full data
    "local_mode",     # Run Ray Tune locally vs distributed

    # Optuna-specific
    "n_trials",           # Total optimization trials
    "timeout",            # Max HPO time (seconds)
    "n_startup_trials",   # Random trials before Bayesian optimization
    "n_warmup_steps",     # Steps before pruner starts
    "interval_steps",     # Pruning check frequency

    # DEHB-specific
    "eta",                # Successive halving reduction (default: 3)
    "max_tres",           # Max resources per trial
    "reduction_factor",   # Budget reduction (HyperBand)
    "fevals",             # Total function evaluations

    # Evolutionary (NSGA-II, NSGA-III)
    "indpb",      # Independent mutation probability
    "tournsize",  # Tournament selection size (higher = more elitism)
    "cxpb",       # Crossover probability
    "mutpb",      # Mutation probability
    "n_pop",      # Population size
    "n_gen",      # Number of generations

    # Ray Tune-specific
    "num_samples",   # Total trials
    "max_failures",  # Fault tolerance
    "max_conc",      # Max concurrent trials

    # Grid search
    "grid",  # Parameter grid definition
)
```

### Supported Methods

| Method   | Algorithm                                    | Use Case                                 |
| -------- | -------------------------------------------- | ---------------------------------------- |
| `optuna` | Bayesian optimization (TPE, NSGA-II, CMA-ES) | General-purpose, multi-objective         |
| `dehb`   | Differential Evolution Hyperband             | Efficient multi-fidelity optimization    |
| `ray`    | Distributed HPO (ASHA, HyperBand, Bayesian)  | Large-scale, cluster computing           |
| `grid`   | Exhaustive grid search                       | Small search spaces, baseline comparison |

### Usage Example

```yaml
# assets/configs/tasks/hpo.yaml
hpo_method: optuna
n_trials: 100
timeout: 3600 # 1 hour
metric: profit
hpo_range:
  learning_rate: [1e-5, 1e-3, log]
  batch_size: [64, 128, 256]
```

**Validation**: Missing keys trigger default value fallbacks in `logic/src/configs/hpo.py`

**CLI Integration**: Keys map to `logic/controllers/cli/ts_parser.py` arguments

---

## 10. User Interface Constants

**Module**: `logic/src/constants/user_interface.py`

### Terminal Output

```python
PBAR_WAIT_TIME: float = 0.1  # seconds (100ms refresh interval)
```

**Purpose**: tqdm progress bar update rate

**Trade-off**: Lower = smoother (higher CPU), higher = choppier (lower CPU)

**Default**: 10 Hz is smooth without CPU overhead

```python
TQDM_COLOURS = [
    "red",      # Worker 0 (or error indicator)
    "blue",     # Worker 1
    "green",    # Worker 2 (success indicator)
    "yellow",   # Worker 3 (warning indicator)
    "magenta",  # Worker 4
    "cyan",     # Worker 5
]
```

**Usage**: Parallel simulation workers (cycles through colors when >6 workers)

**Note**: "red" reserved for errors; avoid using for normal progress

### Matplotlib Styling

```python
MARKERS: List[str] = ["P", "s", "^", "8", "*"]
```

**Shapes**: Pentagon, Square, Triangle, Octagon, Star

**Purpose**: Visual distinction in scatter/line plots

```python
PLOT_COLOURS: List[str] = [
    "red",     # Policy 0 or worst-case
    "blue",    # Policy 1 or baseline
    "green",   # Policy 2 or best-case
    "yellow",  # Policy 3 (caution on white backgrounds)
    "magenta", # Policy 4
    "cyan",    # Policy 5
    "black",   # Reference lines, grid, text
]
```

**Palette**: ANSI 6-color + black

**Usage**: Multi-policy comparisons, Pareto fronts, time series

```python
LINESTYLES: List[Union[str, Tuple[int, Tuple[int, ...]]]] = [
    "dotted",               # · · · ·
    "dashed",               # - - - -
    "dashdot",              # -·-·-·-·
    (0, (3, 5, 1, 5, 1, 5)), # Custom: dash-dot-dot
    "solid",                # ________
]
```

**Purpose**: Distinguish >6 series (6 colors × 5 linestyles = 30 unique)

### Design Principles

1. **Accessibility**: Colorblind-safe palettes, high contrast
2. **Consistency**: Same colors for same concepts (CLI/GUI/plots)
3. **Terminal Safety**: ANSI color compatibility (tqdm, rich)
4. **Print Compatibility**: Patterns distinguishable in grayscale

---

## 11. System Constants

**Module**: `logic/src/constants/system.py`

### Multi-core Processing

```python
CORE_LOCK_WAIT_TIME: int = 100  # milliseconds (base timeout)
```

**Purpose**: File/resource locking in single-core mode

```python
LOCK_TIMEOUT: int = CORE_LOCK_WAIT_TIME  # Dynamic timeout (modified at runtime)
```

**Scaling**: Modified by `update_lock_wait_time()` based on CPU count

**Prevents**: Deadlocks when many workers contend for files

```python
def update_lock_wait_time(num_cpu_cores: Optional[int] = None) -> int:
    """
    Updates LOCK_TIMEOUT based on CPU cores.

    Args:
        num_cpu_cores: Number of CPU cores to scale timeout by.

    Returns:
        New LOCK_TIMEOUT value.
    """
    global LOCK_TIMEOUT
    if num_cpu_cores is None:
        LOCK_TIMEOUT = CORE_LOCK_WAIT_TIME
    else:
        LOCK_TIMEOUT = CORE_LOCK_WAIT_TIME * num_cpu_cores
    return LOCK_TIMEOUT
```

**Thread Safety**: Safe for parallel data generation (each subprocess gets separate module globals)

### File System Operations

```python
CONFIRM_TIMEOUT: int = 30  # seconds
```

**Purpose**: GUI confirmation dialog timeout

**Behavior**: Destructive operations (delete, update) auto-cancelled if user doesn't respond

```python
FS_COMMANDS: List[str] = ["create", "read", "update", "delete", "cryptography"]
```

**Mapping**: Surfaced in the Studio System Tools page (`app/src/pages/files/SystemTools.tsx`)

| Command        | Operation                     | Confirmation Required |
| -------------- | ----------------------------- | --------------------- |
| `create`       | New file/directory generation | No                    |
| `read`         | File content display          | No                    |
| `update`       | In-place modification         | Yes                   |
| `delete`       | Permanent removal             | Yes                   |
| `cryptography` | Encrypt/decrypt data          | No                    |

### Operator Evaluation

```python
OPERATION_MAP: Dict[str, Callable[[Any, Any], Any]] = {
    # Assignment
    "=": lambda x, y: y,
    "": lambda x, y: x,

    # Arithmetic
    "+": lambda x, y: x + y,
    "+=": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "-=": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "*=": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "/=": lambda x, y: x / y,
    "**": lambda x, y: x**y,
    "**=": lambda x, y: x**y,
    "//": lambda x, y: x // y,
    "//=": lambda x, y: x // y,
    "%": lambda x, y: x % y,
    "%=": lambda x, y: x % y,
    "@": lambda x, y: x @ y,      # Matrix multiplication
    "@=": lambda x, y: x @ y,
    "divmod": lambda x, y: divmod(x, y),

    # Comparison
    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    "<": lambda x, y: x < y,
    "<=": lambda x, y: x <= y,
    ">": lambda x, y: x > y,
    ">=": lambda x, y: x >= y,

    # Bitwise
    "<<": lambda x, y: x << y,
    "<<=": lambda x, y: x << y,
    ">>": lambda x, y: x >> y,
    ">>=": lambda x, y: x >> y,
    "|": lambda x, y: x | y,
    "|=": lambda x, y: x | y,
    "&": lambda x, y: x & y,
    "&=": lambda x, y: x & y,
    "^": lambda x, y: x ^ y,
    "^=": lambda x, y: x ^ y,

    # Identity/Membership
    "is": lambda x, y: x is y,
    "isnot": lambda x, y: x is not y,
    "in": lambda x, y: x in y,
    "notin": lambda x, y: x not in y,
}
```

### Usage Context

#### 1. GUI Widgets

```python
# User selects "+=" and enters value 5
current_value = 100
user_input = 5
operation = "+="

op_func = OPERATION_MAP[operation]
new_value = op_func(current_value, user_input)  # 105
```

#### 2. Config File Processing

```python
# Dynamic value transformations during Hydra composition
# config.yaml: learning_rate *= 0.1
```

#### 3. File System Automation

```python
# Conditional logic for batch operations
if OPERATION_MAP[">"](file_size, threshold):
    compress_file(file)
```

### Security Note

**Purpose**: Enables runtime operator evaluation without `eval()` (safer than `eval()`)

**Scope**: Only mathematical/logical operators, no arbitrary code execution

---

## 12. Path Constants

**Module**: `logic/src/constants/paths.py`

### Root Directory Resolution

```python
# Dynamic search upward from cwd
path: Path = Path(os.getcwd())
parts: tuple[str, ...] = path.parts

try:
    root_dir = Path(*parts[: parts.index("WSmart-Route") + 1])
except ValueError:
    root_dir = Path(*parts[: parts.index("WSmart-Simulator") + 1])

ROOT_DIR: Path = root_dir
```

**Resolution Order**:

1. Get current working directory
2. Search upward for `"WSmart-Route"` or `"WSmart-Simulator"` in path parts
3. Set `ROOT_DIR` to that location
4. Derive asset paths relative to `ROOT_DIR`

**Supports**:

- Running from any subdirectory (`notebooks/`, `logic/test/`)
- Multiple project clones
- Virtual environment isolation

**Example**:

```python
# From /home/user/Repositories/WSmart-Route/logic/test/
# ROOT_DIR = Path("/home/user/Repositories/WSmart-Route")
```

### Application Icon

```python
ICON_FILE: str = os.path.join(ROOT_DIR, "assets", "images", "logo-wsmartroute-white.png")
```

**Usage**: Application icon for exported tooling and report generators

**Format**: PNG, white logo on transparent background

**Dimensions**: 512×512 px (scales down for UI)

---

## 13. Testing Constants

**Module**: `logic/src/constants/testing.py`

### Test Module Registry

```python
TEST_MODULES: Dict[str, str] = {
    # CLI Command Tests
    "parser": "test_configs_parser.py",
    "train": "test_train_command.py",
    "mrl": "test_mrl_train_command.py",
    "hp_optim": "test_hp_optim_command.py",
    "gen_data": "test_gen_data_command.py",
    "eval": "test_eval_command.py",
    "test_sim": "test_test_command.py",
    "file_system": "test_file_system_command.py",

    # Component Tests
    "actions": "test_custom_actions.py",
    "edge_cases": "test_edge_cases.py",
    "layers": "test_model_layers.py",
    "scheduler": "test_lr_scheduler.py",
    "optimizer": "test_optimizer.py",

    # Integration Tests
    "integration": "test_integration.py",
}
```

### Test Organization

#### 1. CLI Command Tests

**Purpose**: Validate `main.py` entry points

**Characteristics**:

- Fast smoke tests (do not train full models)
- Check argument parsing, config loading, orchestration
- Each command has dedicated test file

**Example**:

```bash
python main.py test_suite --module train
```

#### 2. Component Tests

**Purpose**: Unit tests for subsystems

**Characteristics**:

- Test individual classes/functions in isolation
- Mock external dependencies (environments, datasets, GPU)
- Fast execution (<1s per test)

**Example**:

```bash
python main.py test_suite --module layers
```

#### 3. Integration Tests

**Purpose**: End-to-end workflows

**Characteristics**:

- Test complete pipelines (data gen → train → eval)
- Use small instances (10-20 nodes, 2-3 epochs)
- Slow but comprehensive (30s-2min per test)

**Example**:

```bash
python main.py test_suite --module integration
```

### Usage Examples

```bash
# Run all tests
python main.py test_suite

# Run specific module
python main.py test_suite --module train

# Run multiple modules
python main.py test_suite --module train eval integration
```

---

## 14. Statistics Constants

**Module**: `logic/src/constants/stats.py`

### Statistical Function Registry

```python
STATS_FUNCTION_MAP: Dict[str, Callable[..., Any]] = {
    # Central tendency
    "mean": statistics.mean,        # Arithmetic average
    "median": statistics.median,    # Middle value (robust to outliers)
    "mode": statistics.mode,        # Most common value

    # Dispersion
    "stdev": statistics.stdev,      # Standard deviation (√variance)
    "var": statistics.variance,     # Variance (σ²)

    # Distribution
    "quant": statistics.quantiles,  # Quantile values (default n=4)

    # Aggregation
    "size": len,                    # Count of elements
    "sum": sum,                     # Total of all values
    "min": min,                     # Minimum value
    "max": max,                     # Maximum value
}
```

### Usage Context

#### 1. Simulation Analyzer

**File**: `logic/src/pipeline/simulations/actions/logging.py`

```python
# Compute daily/overall statistics
mean_cost = STATS_FUNCTION_MAP["mean"](daily_costs)
median_km = STATS_FUNCTION_MAP["median"](route_lengths)
```

#### 2. Studio Chart Utilities

**File**: `app/src/utils/trainingMetrics.ts`

```python
# Generate summary statistics for plots
summary = {
    "mean": STATS_FUNCTION_MAP["mean"](data),
    "median": STATS_FUNCTION_MAP["median"](data),
    "quartiles": STATS_FUNCTION_MAP["quant"](data, n=4),
}
```

#### 3. HPO Objective Functions

**File**: `logic/src/pipeline/rl/hpo/`

```python
# Reduce multi-run results to scalar
best_loss = STATS_FUNCTION_MAP["min"](validation_losses)
```

#### 4. Notebook Analysis

**File**: `notebooks/*.ipynb`

```python
# Quick statistical summaries
profit_quartiles = STATS_FUNCTION_MAP["quant"](profits, n=4)
```

### Function Descriptions

| Function | Input    | Output      | Use Case                  |
| -------- | -------- | ----------- | ------------------------- |
| `mean`   | Iterable | float       | Normal distributions      |
| `median` | Iterable | float       | Robust to outliers        |
| `mode`   | Iterable | Any         | Categorical/discrete data |
| `stdev`  | Iterable | float       | Spread around mean        |
| `var`    | Iterable | float       | Compare variability       |
| `quant`  | Iterable | List[float] | Distribution analysis     |
| `size`   | Iterable | int         | Sample size               |
| `sum`    | Iterable | float/int   | Cumulative metrics        |
| `min`    | Iterable | Any         | Best-case performance     |
| `max`    | Iterable | Any         | Worst-case performance    |

### Special Note: Quantiles

```python
# Default: quartiles (n=4)
quartiles = STATS_FUNCTION_MAP["quant"](data)  # [Q1, Q2, Q3]

# Custom: deciles (n=10)
deciles = STATS_FUNCTION_MAP["quant"](data, n=10)
```

---

## 15. Dashboard Constants

**Module**: `logic/src/constants/dashboard.py`

### Route Visualization Colors

```python
ROUTE_COLORS = [
    "#e41a1c",  # Red - Vehicle 0
    "#377eb8",  # Blue - Vehicle 1
    "#4daf4a",  # Green - Vehicle 2
    "#984ea3",  # Purple - Vehicle 3
    "#ff7f00",  # Orange - Vehicle 4
    "#ffff33",  # Yellow - Vehicle 5
    "#a65628",  # Brown - Vehicle 6
    "#f781bf",  # Pink - Vehicle 7
]
```

**Source**: ColorBrewer2 "Set1" qualitative palette

**Optimized For**:

- Categorical data visualization
- Print-friendly (maintains distinction in grayscale)
- Colorblind-safe (tested for deuteranopia, protanopia)
- Maximum visual distinction (8+ routes distinguishable)

**Cycling**:

```python
# For >8 routes
color = ROUTE_COLORS[route_id % len(ROUTE_COLORS)]
```

### Bin Status Colors

```python
BIN_COLORS = {
    "served": "#28a745",   # Green (Bootstrap success)
    "pending": "#dc3545",  # Red (Bootstrap danger)
    "depot": "#007bff",    # Blue (Bootstrap primary)
}
```

**Purpose**: Semantic consistency with GUI

**Usage**:

- Folium map popups
- Bin state heatmaps
- Status legends

**Color Scheme**: Bootstrap 4/5 semantic colors

### Usage Context

#### 1. Simulation Dashboard

**File**: `app/src/pages/monitor/SimulationMonitor.tsx`

```python
for route_id, tour in enumerate(tours):
    color = ROUTE_COLORS[route_id % len(ROUTE_COLORS)]
    plt.plot(tour, color=color, label=f"Vehicle {route_id}")
```

#### 2. Folium Map Visualizations

**File**: `notebooks/*.ipynb`

```python
for bin_id, status in bin_states.items():
    folium.CircleMarker(
        location=coords[bin_id],
        color=BIN_COLORS[status],
        popup=f"Bin {bin_id}: {status}"
    ).add_to(map)
```

#### 3. Matplotlib Charts

**File**: `logic/src/utils/logging/plotting/`

```python
plt.scatter(x, y, c=ROUTE_COLORS[0], label="Policy A")
```

---

## 16. Legacy Constants

**Module**: `logic/src/constants/tasks.py`

### ⚠️ DEPRECATION WARNING

This module contains **deprecated constants** retained only for backward compatibility with:

- Old simulation logs
- Legacy notebooks in `notebooks/archive/`
- Unit tests for backward compatibility validation

**DO NOT use in new code**. These will be removed in v4.0.

### Deprecated Constants

```python
COST_KM = 1.0        # Legacy default (use SimulationConfig.cost_per_km)
REVENUE_KG = 1.0     # Legacy default (use SimulationConfig.revenue_per_kg)
BIN_CAPACITY = 100.0 # Legacy default (use EnvConfig.bin_capacity)
VEHICLE_CAPACITY = 200.0  # DIFFERS from simulation.VEHICLE_CAPACITY (100.0)!
```

### Migration Path

#### Old Code

```python
from logic.src.constants.tasks import COST_KM, REVENUE_KG

profit = REVENUE_KG * kg_collected - COST_KM * km_traveled
```

#### New Code

```python
from logic.src.configs.simulation import SimulationConfig

cfg = SimulationConfig.load("assets/configs/sim/default.yaml")
profit = cfg.revenue_per_kg * kg_collected - cfg.cost_per_km * km_traveled
```

### Critical Note: VEHICLE_CAPACITY

The legacy `tasks.VEHICLE_CAPACITY` (200.0 kg) **differs** from `simulation.VEHICLE_CAPACITY` (100.0 kg):

- **Old simulations**: Used 200 kg
- **New simulations**: Use 100 kg

**Check your config** to ensure correct capacity!

---

## 19. Best Practices

### 1. Import Strategy

✅ **Prefer Top-Level Imports**:

```python
from logic.src.constants import MAX_WASTE, VEHICLE_CAPACITY, METRICS
```

✅ **Use Specific Sub-Module for Clarity**:

```python
from logic.src.constants.simulation import MAX_WASTE, VEHICLE_CAPACITY
from logic.src.constants.policies import SIMPLE_POLICIES, THRESHOLD_POLICIES
```

❌ **Avoid Wildcard Imports**:

```python
from logic.src.constants import *  # Don't do this
```

### 2. Type Safety

✅ **Use Type Hints**:

```python
from typing import Dict, List
from logic.src.constants import METRIC_MAPPING

def normalize_metric(key: str) -> str:
    for canonical, aliases in METRIC_MAPPING.items():
        if key in aliases:
            return canonical
    return key
```

### 3. Documentation

✅ **Document Usage Context**:

```python
# Good: Explains why and where constant is used
IMPROVEMENT_EPSILON: float = 1e-3  # 0.1% typical route cost, prevents cycling in 2-opt

# Bad: No context
EPSILON = 0.001
```

### 4. Deprecation

✅ **Warn Before Removing**:

```python
# Deprecated: Use SimulationConfig.cost_per_km instead
# Will be removed in v4.0
COST_KM = 1.0  # Legacy default
```

### 5. Configuration Over Constants

✅ **Prefer Config Files**:

```yaml
# assets/configs/sim/default.yaml
vehicle_capacity: 100.0
cost_per_km: 1.0
revenue_per_kg: 1.5
```

❌ **Avoid Hardcoded Values**:

```python
# Don't hardcode in source
VEHICLE_CAPACITY = 100.0  # Should be in config
```

### 6. Naming Conventions

✅ **Use Descriptive Names**:

```python
IMPROVEMENT_EPSILON: float = 1e-3       # Good
DEFAULT_EVAL_BATCH_SIZE: int = 1024     # Good
```

❌ **Avoid Ambiguous Names**:

```python
EPSILON = 1e-3  # Epsilon for what?
BATCH_SIZE = 1024  # Training or eval?
```

### 7. Units and Ranges

✅ **Document Units**:

```python
COLLECTION_TIME_MINUTES = 3.0  # minutes per bin
VEHICLE_SPEED_KMH = 40.0       # km/h (urban)
MIP_GAP = 0.01                 # 1% optimality gap (0.0-1.0)
```

### 8. Immutability

✅ **Use Uppercase for True Constants**:

```python
MAX_WASTE: float = 1.0  # Never changes
METRICS: List[str] = [...]  # Contents fixed
```

✅ **Use Tuples for Immutable Collections**:

```python
HOP_KEYS: Tuple[str, ...] = ("hpo_method", "metric", ...)
```

❌ **Avoid Mutable Globals**:

```python
# Only exception: LOCK_TIMEOUT (scaled by CPU count)
```

---

## 18. Migration Guide

### From tasks.py (Deprecated) to Config Files

#### Step 1: Identify Legacy Constants

Search your codebase for:

```bash
grep -r "from logic.src.constants.tasks import" .
grep -r "COST_KM\|REVENUE_KG\|BIN_CAPACITY" .
```

#### Step 2: Create Config File

```yaml
# assets/configs/sim/my_simulation.yaml
cost_per_km: 1.0
revenue_per_kg: 1.5
bin_capacity: 100.0
vehicle_capacity: 200.0
```

#### Step 3: Update Code

**Before**:

```python
from logic.src.constants.tasks import COST_KM, REVENUE_KG

def compute_profit(kg: float, km: float) -> float:
    return REVENUE_KG * kg - COST_KM * km
```

**After**:

```python
from logic.src.configs.simulation import SimulationConfig

def compute_profit(kg: float, km: float, cfg: SimulationConfig) -> float:
    return cfg.revenue_per_kg * kg - cfg.cost_per_km * km
```

#### Step 4: Load Config

```python
from logic.src.configs.simulation import SimulationConfig

cfg = SimulationConfig.load("assets/configs/sim/my_simulation.yaml")
profit = compute_profit(kg=125.0, km=45.0, cfg=cfg)
```

### From Hardcoded Values to Constants

#### Step 1: Find Magic Numbers

```bash
# Search for hardcoded values
grep -r "0.9\|100.0\|10000.0" logic/src/
```

#### Step 2: Replace with Named Constants

**Before**:

```python
if bin.fill_level > 0.9:  # What's 0.9?
    collect(bin)
```

**After**:

```python
from logic.src.constants.waste import CRITICAL_FILL_THRESHOLD

if bin.fill_level > CRITICAL_FILL_THRESHOLD:  # Clear meaning
    collect(bin)
```

### From simulation.VEHICLE_CAPACITY to Config

**Before**:

```python
from logic.src.constants.simulation import VEHICLE_CAPACITY

def check_capacity(load: float) -> bool:
    return load <= VEHICLE_CAPACITY
```

**After**:

```python
from logic.src.configs.env import EnvConfig

def check_capacity(load: float, cfg: EnvConfig) -> bool:
    return load <= cfg.vehicle_capacity
```

---

## 19. Cross-References

### Related Documentation

- [Configuration Guide](CONFIGURATION_GUIDE.md) - Hydra config system
- [Architecture Guide](ARCHITECTURE.md) - System design overview
- [CLAUDE.md](../CLAUDE.md) - AI coding assistant instructions

### Key Files Using Constants

| File                                          | Constants Used                          | Purpose                 |
| --------------------------------------------- | --------------------------------------- | ----------------------- |
| `logic/src/policies/__init__.py`              | `SIMPLE_POLICIES`, `THRESHOLD_POLICIES` | Policy factory          |
| `logic/src/envs/*.py`                         | `MAX_WASTE`, `VEHICLE_CAPACITY`         | Environment physics     |
| `logic/src/pipeline/rl/core/*.py`             | `METRIC_MAPPING`, `LOSS_KEYS`           | RL training             |
| `logic/src/pipeline/simulations/simulator.py` | `METRICS`, `SIM_METRICS`                | Simulation logging      |
| `logic/src/utils/logging/plotting/*.py`       | `PLOT_COLOURS`, `MARKERS`               | Matplotlib charts       |

### Configuration Files

| Config File                      | Constants Used                            | Purpose               |
| -------------------------------- | ----------------------------------------- | --------------------- |
| `assets/configs/policies/*.yaml` | Policy naming conventions                 | Solver configuration  |
| `assets/configs/envs/*.yaml`     | `NODE_DIM`, `DEPOT_DIM`                   | Environment setup     |
| `assets/configs/models/*.yaml`   | `FEED_FORWARD_EXPANSION`, `TANH_CLIPPING` | Model hyperparameters |
| `assets/configs/sim/*.yaml`      | `VEHICLE_CAPACITY`, `MAX_WASTE`           | Simulation parameters |

---

## 20. Appendix: Quick Reference

### Most Common Constants

```python
# Simulation
from logic.src.constants import MAX_WASTE, VEHICLE_CAPACITY, METRICS

# Model
from logic.src.constants import NODE_DIM, DEPOT_DIM, TANH_CLIPPING

# Routing
from logic.src.constants import MIP_GAP, IMPROVEMENT_EPSILON

# Waste
from logic.src.constants import CRITICAL_FILL_THRESHOLD

# Paths
from logic.src.constants import ROOT_DIR, ICON_FILE
```

### Import Cheat Sheet

```python
# Option 1: Top-level (recommended)
from logic.src.constants import MAX_WASTE, VEHICLE_CAPACITY

# Option 2: Specific sub-module
from logic.src.constants.simulation import MAX_WASTE, VEHICLE_CAPACITY

# Option 3: Import module
from logic.src import constants
capacity = constants.VEHICLE_CAPACITY
```

### Common Pitfalls

❌ **Using legacy constants**:

```python
from logic.src.constants.tasks import COST_KM  # Deprecated!
```

✅ **Use config files**:

```python
cfg = SimulationConfig.load("config.yaml")
cost = cfg.cost_per_km
```

❌ **Hardcoding values**:

```python
if fill > 0.9:  # Magic number
```

✅ **Use named constants**:

```python
from logic.src.constants.waste import CRITICAL_FILL_THRESHOLD
if fill > CRITICAL_FILL_THRESHOLD:
```

❌ **Confusing num_loc/graph_size/n_nodes**:

```python
graph_size = num_loc  # Wrong! graph_size = num_loc + 1
```

✅ **Use correct semantics**:

```python
graph_size = num_loc + 1  # Includes depot
n_nodes = len(dist_matrix) - 1  # Excludes depot
```

---

**Document Version**: 1.0
**Maintainer**: WSmart+ Route Development Team
**Last Review**: February 2026
**Next Review**: May 2026
