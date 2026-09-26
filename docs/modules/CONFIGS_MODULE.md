# Type-Safe Configuration System

**Module**: `logic/src/configs`
**Purpose**: Comprehensive technical specification of the type-safe dataclass-based configuration architecture—orchestrating environments, models, and RL workflows.
**Version**: 3.0
**Last Updated**: February 2026

---

## Table of Contents

1.  [**Overview**](#1-overview)
2.  [**Module Organization**](#2-module-organization)
3.  [**Root Configuration**](#3-root-configuration)
4.  [**Environment Configurations**](#4-environment-configurations)
5.  [**Model Configurations**](#5-model-configurations)
6.  [**Policy Configurations**](#6-policy-configurations)
7.  [**Reinforcement Learning Configurations**](#7-reinforcement-learning-configurations)
8.  [**Task Configurations**](#8-task-configurations)
9.  [**Integration Examples**](#9-integration-examples)
10. [**Best Practices**](#10-best-practices)
11. [**Quick Reference**](#11-quick-reference)

---

## 1. Overview

The `logic/src/configs` module provides a comprehensive, type-safe configuration architecture built on Python dataclasses. This system centralizes all parameters for:

- **Environment Physics**: Problem definitions, graph structure, data generation
- **Model Architecture**: Neural network design, optimization, decoding strategies
- **RL Algorithms**: Training objectives, baselines, policy updates
- **Policy Behaviors**: Classical solvers, metaheuristics, pre/route improvement
- **Workflow Orchestration**: Training loops, simulations, evaluations, HPO

### Key Features

- **Dataclass Foundation**: Automatic validation, readable `repr()`, easy serialization
- **Hierarchical Composition**: Deep nesting with default fallbacks
- **Type Safety**: Full type hints for IDE support and static analysis
- **CLI Integration**: Direct mapping to command-line overrides via Hydra
- **Immutable by Default**: Prevents accidental configuration mutation

### Architecture Principles

The configuration system follows a **hierarchical factory pattern**:

```
Config (Root)
├── EnvConfig (Problem World)
│   ├── GraphConfig (Connectivity)
│   ├── ObjectiveConfig (Multi-objective weights)
│   └── DataConfig (Generation parameters)
├── ModelConfig (Neural Architecture)
│   ├── EncoderConfig (Feature extraction)
│   ├── DecoderConfig (Action selection)
│   └── OptimConfig (Training hyperparameters)
├── RLConfig (Learning Algorithms)
│   ├── PPOConfig / POMOConfig / etc.
│   └── ImitationConfig
├── TrainConfig / EvalConfig / SimConfig (Tasks)
└── PolicyConfig variants (Classical solvers)
```

---

## 2. Module Organization

### Directory Structure

```
logic/src/configs/
├── __init__.py              # Root Config class
├── README.md                # Architecture overview
│
├── envs/                    # Problem environment configurations
│   ├── __init__.py
│   ├── env.py               # EnvConfig (core problem settings)
│   ├── graph.py             # GraphConfig (connectivity)
│   ├── data.py              # DataConfig (generation)
│   ├── objective.py         # ObjectiveConfig (reward weights)
│   └── README.md
│
├── models/                  # Neural network configurations
│   ├── __init__.py
│   ├── model.py             # ModelConfig (architecture assembly)
│   ├── encoder.py           # EncoderConfig (embedding)
│   ├── decoder.py           # DecoderConfig (policy head)
│   ├── decoding.py          # DecodingConfig (inference strategy)
│   ├── optim.py             # OptimConfig (optimization)
│   ├── activation_function.py
│   ├── normalization.py
│   └── README.md
│
├── policies/                # Classical solver configurations
│   ├── __init__.py
│   ├── alns.py              # ALNS metaheuristic
│   ├── hgs.py, hgs_alns.py  # Hybrid Genetic Search
│   ├── aco.py               # Ant Colony Optimization
│   ├── ils.py, sisr.py      # Local search variants
│   ├── bpc.py               # Branch-and-Price-and-Cut exact solver
│   ├── neural.py            # Neural policy wrapper
│   ├── other/               # Modular extensions
│   │   ├── mandatory_selection.py       # Pre-selection strategies
│   │   └── route_improvement.py
│   └── README.md
│
├── rl/                      # Reinforcement learning configurations
│   ├── __init__.py          # RLConfig
│   ├── README.md
│   ├── core/                # Learning algorithms
│   │   ├── ppo.py           # Proximal Policy Optimization
│   │   ├── pomo.py          # Policy Optimization w/ Multiple Optima
│   │   ├── symnco.py        # Symmetry-aware NCO
│   │   ├── imitation.py     # Supervised learning
│   │   ├── adaptive_imitation.py
│   │   ├── grpo.py, gdpo.py, sapo.py
│   │   └── README.md
│   └── policies/            # Expert policy wrappers
│       ├── aco.py, alns.py, hgs.py, ...
│       └── README.md
│
└── tasks/                   # Workflow orchestration
    ├── __init__.py
    ├── train.py             # Training loop configuration
    ├── eval.py              # Evaluation configuration
    ├── sim.py               # Multi-day simulation
    ├── meta_rl.py           # Meta-RL / HRL
    ├── hpo.py               # Hyperparameter optimization
    └── README.md
```

### Module Exports

The root `__init__.py` exports all major config classes:

```python
from logic.src.configs import (
    # Root
    Config,
    # Environments
    EnvConfig, GraphConfig, DataConfig, ObjectiveConfig,
    # Models
    ModelConfig, EncoderConfig, DecoderConfig, DecodingConfig, OptimConfig,
    # Tasks
    TrainConfig, EvalConfig, SimConfig, MetaRLConfig, HPOConfig,
    # Policies
    ALNSConfig, HGSConfig, ACOConfig, BPCConfig, ILSConfig,
    MandatorySelectionConfig, route improvementConfig,
    # RL
    RLConfig, PPOConfig, POMOConfig, ImitationConfig,
)
```

---

## 3. Root Configuration

**File**: `logic/src/configs/__init__.py`

The `Config` class is the single source of truth for any execution. It composes all sub-configurations into a unified interface.

### Config Class

```python
@dataclass
class Config:
    """Root configuration for WSmart+ Route.

    Attributes:
        env: Environment configuration.
        model: Model architecture configuration.
        train: Training configuration.
        optim: Optimizer configuration.
        rl: RL algorithm configuration.
        meta_rl: Meta-RL configuration.
        hpo: Hyperparameter optimization configuration.
        eval: Evaluation configuration.
        sim: Simulation configuration.
        data: Data generation configuration.
        mandatory_selection: mandatory selection strategy configuration.
        route_improvement: Route refinement configuration.
        seed: Random seed for reproducibility.
        device: Device to use ('cpu', 'cuda', 'cuda:0').
        experiment_name: Optional name for the experiment.
        task: Task to perform ('train', 'eval', 'test_sim', 'gen_data').
        wandb_mode: Weights & Biases logging mode ('offline', 'online', 'disabled').
        output_dir: Directory for model checkpoints.
        log_dir: Directory for logs.
        verbose: Enable detailed console output.
    """

    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    meta_rl: MetaRLConfig = field(default_factory=MetaRLConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mandatory_selection: MandatorySelectionConfig = field(default_factory=MandatorySelectionConfig)
    route_improvement: route improvementConfig = field(default_factory=route improvementConfig)

    seed: int = 42
    device: str = "cuda"
    experiment_name: Optional[str] = None
    task: str = "train"
    wandb_mode: str = "offline"
    no_tensorboard: bool = False
    no_progress_bar: bool = False
    output_dir: str = "assets/model_weights"
    log_dir: str = "logs"
    run_name: Optional[str] = None
    verbose: bool = True
    start: int = 0
    callbacks: Dict[str, Any] = field(default_factory=dict)
```

### Usage Examples

**Basic Configuration**

```python
from logic.src.configs import Config

# Use all defaults
config = Config()

# Override top-level fields
config = Config(
    task="train",
    device="cuda:0",
    seed=1234,
    wandb_mode="online"
)
```

**Nested Configuration**

```python
from logic.src.configs import Config, EnvConfig, ModelConfig

config = Config(
    env=EnvConfig(name="wcvrp", num_loc=100),
    model=ModelConfig(name="am"),
    device="cuda:0"
)
```

**Deep Nesting**

```python
from logic.src.configs import Config, EnvConfig, GraphConfig, ObjectiveConfig

config = Config(
    env=EnvConfig(
        name="wcvrp",
        num_loc=100,
        graph=GraphConfig(area="riomaior", waste_type="plastic"),
        reward=ObjectiveConfig(w_waste=2.0, w_overflows=50.0)
    )
)
```

---

## 4. Environment Configurations

**Directory**: `logic/src/configs/envs/`
**Purpose**: Define problem physics, graph structure, data generation, and optimization objectives

### EnvConfig

**File**: `envs/env.py`

Core configuration for a problem instance.

```python
@dataclass
class EnvConfig:
    """Environment configuration.

    Attributes:
        name: Problem type ('vrpp', 'cvrpp', 'ctop', 'wcvrp', 'cwcvrp', 'sdwcvrp', 'scwcvrp').
        num_loc: Number of customer locations (excluding depot).
        min_loc: Minimum coordinate value for node generation.
        max_loc: Maximum coordinate value for node generation.
        capacity: Vehicle capacity (None uses problem default). For ctop this
            is CVRPP's per-trip capacity, not replaced by the time budget.
        shift_hours / avg_speed_kmh / service_time_h: Optional ctop temporal
            overrides; None uses routing temporal defaults (7 h, 35 km/h, 0.025 h).
        graph: Graph connectivity settings.
        reward: Multi-objective reward weights.
        data_distribution: Geographic distribution ('uniform', 'clustered', 'mixed').
        min_fill: Minimum initial bin fill level (0.0 to 1.0).
        max_fill: Maximum initial bin fill level (0.0 to 1.0).
        fill_distribution: Stochastic model for bin levels ('uniform', 'gamma', 'beta').
    """

    name: str = "vrpp"
    num_loc: int = 50
    min_loc: float = 0.0
    max_loc: float = 1.0
    capacity: Optional[float] = None
    shift_hours: Optional[float] = None
    avg_speed_kmh: Optional[float] = None
    service_time_h: Optional[float] = None
    graph: GraphConfig = field(default_factory=GraphConfig)
    reward: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    data_distribution: Optional[str] = None
    min_fill: float = 0.0
    max_fill: float = 1.0
    fill_distribution: str = "uniform"
```

**Usage Examples**

```python
# Standard VRPP with 50 locations
env_config = EnvConfig(name="vrpp", num_loc=50)

# CTOP: CVRPP capacity plus an independent per-trip time budget
env_config = EnvConfig(name="ctop", capacity=100.0, shift_hours=7.0)

# Capacitated WCVRP with custom capacity
env_config = EnvConfig(
    name="cwcvrp",
    num_loc=100,
    capacity=200.0,
    min_fill=0.3,
    max_fill=0.9
)

# Stochastic waste with gamma distribution
env_config = EnvConfig(
    name="scwcvrp",
    fill_distribution="gamma",
    min_fill=0.0,
    max_fill=1.0
)
```

### GraphConfig

**File**: `envs/graph.py`

Defines connectivity and real-world data paths.

```python
@dataclass
class GraphConfig:
    """Graph connectivity and data configuration.

    Attributes:
        area: Geographic area ('riomaior', 'figueiradafoz', 'mixrmbac').
        num_loc: Number of locations (overrides env.num_loc if set).
        waste_type: Type of waste ('plastic', 'glass', 'paper').
        vertex_method: Coordinate transformation method ('mmn', 'standard').
        distance_method: Distance matrix computation ('ogd', 'euclidean').
        dm_filepath: Path to pre-computed distance matrix.
        edge_threshold: Edge density threshold (0 = fully connected).
        edge_method: Edge selection method ('dist', 'knn').
        focus_graph: Paths to specific graph file for clustering.
        focus_size: Number of focus graphs to include in training.
        eval_focus_size: Number of focus graphs for evaluation.
    """

    area: str = "riomaior"
    num_loc: int = 50
    waste_type: str = "plastic"
    vertex_method: str = "mmn"
    distance_method: str = "ogd"
    dm_filepath: Optional[str] = None
    edge_threshold: str = "0"
    edge_method: Optional[str] = None
    focus_graph: Optional[List[str]] = None
    focus_size: int = 0
    eval_focus_size: int = 0
```

**Usage Examples**

```python
# Real-world Rio Maior plastic collection
graph_config = GraphConfig(
    area="riomaior",
    waste_type="plastic",
    vertex_method="mmn",
    distance_method="ogd"
)

# Custom distance matrix
graph_config = GraphConfig(
    dm_filepath="data/custom_distances.csv"
)

# Focus on specific graphs for curriculum learning
graph_config = GraphConfig(
    focus_graph=["data/hard_instances/*.pkl"],
    focus_size=100
)
```

### DataConfig

**File**: `envs/data.py`

Parameters for batch generating synthetic datasets.

```python
@dataclass
class DataConfig:
    """Data generation configuration.

    Attributes:
        name: Dataset identifier (creates .td files for training data).
        filename: Specific filename (overrides data_dir).
        data_dir: Root directory for datasets.
        problem: Problem filter ('vrpp', 'wcvrp', 'all').
        mu: Mean for Gaussian coordinate noise.
        sigma: Std dev for Gaussian coordinate noise.
        data_distributions: List of distributions ('uniform', 'cluster', 'all').
        num_locs: Problem sizes to generate.
        penalty_factor: Penalty weight for uncollected waste (VRPP).
        overwrite: Overwrite existing files.
        seed: Random seed.
        n_epochs: Number of epochs worth of data.
        dataset_type: Category ('train', 'val', 'test_simulator').
        area: Geographic context.
        waste_type: Waste type for realistic scenarios.
        graph: Graph generation settings.
    """

    name: Optional[str] = None
    filename: Optional[str] = None
    data_dir: str = "datasets"
    problem: str = "all"
    mu: Optional[List[float]] = None
    sigma: Any = 0.6
    data_distributions: List[str] = field(default_factory=lambda: ["all"])
    num_locs: List[int] = field(default_factory=lambda: [20, 50, 100])
    penalty_factor: float = 3.0
    overwrite: bool = False
    seed: int = 42
    n_epochs: int = 1
    epoch_start: int = 0
    dataset_type: Optional[str] = None
    graph: GraphConfig = field(default_factory=GraphConfig)
```

**Usage Examples**

```python
# Generate training data for multiple problem sizes
data_config = DataConfig(
    name="train_vrpp",
    problem="vrpp",
    num_locs=[20, 50, 100],
    data_distributions=["uniform", "cluster"]
)

# Generate validation data with fixed seed
data_config = DataConfig(
    name="val_wcvrp",
    dataset_type="val",
    seed=1234,
    overwrite=False
)
```

### ObjectiveConfig

**File**: `envs/objective.py`

Multi-objective reward function weights.

```python
@dataclass
class ObjectiveConfig:
    """Multi-objective reward configuration.

    Attributes:
        w_waste: Priority for collecting waste.
        w_overflows: Penalty for bin overflows.
        w_length: Penalty for total route length.
        w_distance: Penalty for Euclidean distance.
        w_profit: Weight for collection profit (VRPP).
        w_efficiency: Weight for route efficiency (waste/distance).
        w_failure: Penalty for failing required collections.
        w_service_level: Weight for temporal service constraints.
    """

    w_waste: float = 1.0
    w_overflows: float = 10.0
    w_length: float = 0.1
    w_distance: float = 0.0
    w_profit: float = 1.0
    w_efficiency: float = 0.5
    w_failure: float = 100.0
    w_service_level: float = 1.0
```

**Usage Examples**

```python
# Prioritize overflow prevention
objective = ObjectiveConfig(
    w_waste=1.0,
    w_overflows=50.0,  # Heavy penalty
    w_length=0.05
)

# Balance profit and efficiency for VRPP
objective = ObjectiveConfig(
    w_profit=2.0,
    w_efficiency=1.0,
    w_length=0.1
)
```

---

## 5. Model Configurations

**Directory**: `logic/src/configs/models/`
**Purpose**: Neural network architecture, optimization, and inference strategies

### ModelConfig

**File**: `models/model.py`

Top-level model assembly configuration.

```python
@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        name: Architecture type ('am', 'deep_decoder', 'tam', 'ptr').
        encoder: Node feature processing settings.
        decoder: Action generation settings.
        temporal_horizon: Temporal history depth for time-aware models.
        policy_config: Path to policy-specific hyperparameter JSON.
        load_path: Path to pre-trained weights.
    """

    name: str = "am"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    temporal_horizon: int = 0
    policy_config: Optional[str] = None
    load_path: Optional[str] = None
```

**Usage Examples**

```python
# Standard Attention Model
model_config = ModelConfig(name="am")

# Deep Decoder AM with 6 layers
model_config = ModelConfig(
    name="deep_decoder",
    encoder=EncoderConfig(n_layers=6)
)

# Temporal AM for multi-day scenarios
model_config = ModelConfig(
    name="tam",
    temporal_horizon=10
)

# Load pre-trained weights
model_config = ModelConfig(
    name="am",
    load_path="assets/model_weights/best_model.pt"
)
```

### EncoderConfig

**File**: `models/encoder.py`

Configures how input nodes are embedded into latent space.

```python
@dataclass
class EncoderConfig:
    """Encoder architecture configuration.

    Attributes:
        type: Encoder backbone ('gat', 'gnn', 'attention', 'gcn').
        embed_dim: Size of latent embedding vector.
        hidden_dim: Internal feed-forward dimension.
        n_layers: Number of transformer/GNN layers.
        n_heads: Number of attention heads (MHA).
        normalization: Normalization layer settings.
        activation: Non-linear activation settings.
        dropout: Dropout probability.
        connection_type: Skip connection type ('residual', 'dense', 'none').
        aggregation_graph: Graph-level pooling ('avg', 'max', 'sum').
        aggregation_node: Node-level neighbor aggregation.
        spatial_bias: Enable distance-based attention bias.
        spatial_bias_scale: Scaling factor for distance biases.
        hyper_expansion: Feed-forward expansion factor.
    """

    type: str = "gat"
    embed_dim: int = 128
    hidden_dim: int = 512
    n_layers: int = 3
    n_heads: int = 8
    n_sublayers: Optional[int] = None
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    dropout: float = 0.1
    mask_inner: bool = True
    mask_graph: bool = False
    spatial_bias: bool = False
    connection_type: str = "residual"
    aggregation_graph: str = "avg"
    aggregation_node: str = "sum"
    spatial_bias_scale: float = 1.0
    hyper_expansion: int = 4
```

**Usage Examples**

```python
# Deep GAT encoder for large problems
encoder_config = EncoderConfig(
    type="gat",
    embed_dim=256,
    n_layers=6,
    n_heads=16,
    dropout=0.1
)

# Lightweight encoder for fast inference
encoder_config = EncoderConfig(
    embed_dim=64,
    n_layers=2,
    n_heads=4
)

# Spatial-aware encoder with distance biasing
encoder_config = EncoderConfig(
    spatial_bias=True,
    spatial_bias_scale=2.0
)
```

### DecoderConfig

**File**: `models/decoder.py`

Configures the policy head that selects actions.

```python
@dataclass
class DecoderConfig:
    """Decoder architecture configuration.

    Attributes:
        type: Mechanism ('attention', 'fc', 'pointer').
        embed_dim: Input hidden dimension.
        hidden_dim: Internal feed-forward dimension.
        n_layers: Number of decoder layers.
        n_heads: Heads for cross-attention.
        normalization: Normalization settings.
        activation: Activation settings.
        decoding: Inference strategy (greedy, sampling, beam).
        dropout: Dropout rate.
        mask_logits: Enforce validity by masking illegal moves.
        tanh_clipping: Soft-clipping of logits to prevent saturation.
    """

    type: str = "attention"
    embed_dim: int = 128
    hidden_dim: int = 512
    n_layers: int = 3
    n_heads: int = 8
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    dropout: float = 0.1
    mask_logits: bool = True
    tanh_clipping: float = 10.0
```

### DecodingConfig

**File**: `models/decoding.py`

Inference strategy for action selection.

```python
@dataclass
class DecodingConfig:
    """Decoding strategy configuration.

    Attributes:
        strategy: Selection method ('greedy', 'sampling', 'beam_search').
        beam_width: Beam size or number of samples.
        temperature: Softmax temperature (>1 = random, <1 = greedy).
        top_k: Limit selection to top K actions.
        top_p: Nucleus sampling threshold.
        tanh_clipping: Logit clipping value.
        mask_logits: Mask invalid actions.
        multistart: Run from multiple starting nodes.
        num_starts: Number of starting positions.
        select_best: Return best of multiple starts.
    """

    strategy: str = "greedy"
    beam_width: int = 1
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    tanh_clipping: float = 0.0
    mask_logits: bool = True
    multistart: bool = False
    num_starts: int = 1
    select_best: bool = False
```

**Usage Examples**

```python
# Greedy deterministic decoding
decoding_config = DecodingConfig(strategy="greedy")

# Stochastic sampling with temperature
decoding_config = DecodingConfig(
    strategy="sampling",
    temperature=1.2,
    beam_width=16
)

# Beam search with width 5
decoding_config = DecodingConfig(
    strategy="beam_search",
    beam_width=5
)

# Top-p nucleus sampling
decoding_config = DecodingConfig(
    strategy="sampling",
    top_p=0.9,
    temperature=1.0
)

# Multi-start construction
decoding_config = DecodingConfig(
    multistart=True,
    num_starts=20,
    select_best=True
)
```

### OptimConfig

**File**: `models/optim.py`

Optimization and learning rate scheduling.

```python
@dataclass
class OptimConfig:
    """Optimizer configuration.

    Attributes:
        optimizer: Optimizer name ('adam', 'adamw', 'sgd', 'rmsprop').
        lr: Base learning rate for the main network.
        lr_critic: Learning rate for Value network (Critic).
        lr_scheduler: Decay type ('cosine', 'step', 'linear', 'plateau').
        lr_decay: Magnitude of decay step.
        lr_min_value: Floor for learning rate.
        weight_decay: L2 regularization constant.
    """

    optimizer: str = "adam"
    lr: float = 1e-4
    lr_critic: float = 1e-4
    lr_scheduler: Optional[str] = None
    lr_decay: float = 1.0
    lr_min_value: float = 0.0
    weight_decay: float = 0.0
```

**Usage Examples**

```python
# AdamW with cosine annealing
optim_config = OptimConfig(
    optimizer="adamw",
    lr=1e-3,
    lr_scheduler="cosine",
    weight_decay=1e-4
)

# Step decay scheduler
optim_config = OptimConfig(
    lr=1e-4,
    lr_scheduler="step",
    lr_decay=0.1
)
```

---

## 6. Policy Configurations

**Directory**: `logic/src/configs/policies/`
**Purpose**: Classical solvers, metaheuristics, and algorithmic extensions

### ALNSConfig

**File**: `policies/alns.py`

Adaptive Large Neighborhood Search configuration.

```python
@dataclass
class ALNSConfig:
    """ALNS policy configuration.

    Attributes:
        time_limit: Maximum runtime in seconds.
        max_iterations: Maximum destroy/repair loops.
        start_temp: Initial temperature for Simulated Annealing.
        cooling_rate: Rate of temperature decrease.
        reaction_factor: How fast operator weights adapt.
        min_removal: Minimum nodes to remove per destroy.
        max_removal_pct: Maximum % of nodes to remove.
        engine: Solver engine ('custom', 'alns' third-party).
        mandatory_selection: Pre-selection strategies.
        route_improvement: Refinement steps.
    """

    time_limit: float = 60.0
    max_iterations: int = 5000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    reaction_factor: float = 0.1
    min_removal: int = 1
    max_removal_pct: float = 0.3
    engine: str = "custom"
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[route improvementConfig]] = None
```

**Usage Examples**

```python
# Standard ALNS
alns_config = ALNSConfig(
    time_limit=120.0,
    max_iterations=10000
)

# Aggressive perturbation
alns_config = ALNSConfig(
    max_removal_pct=0.5,
    start_temp=200.0,
    cooling_rate=0.99
)

# With pre-selection and route improvement
alns_config = ALNSConfig(
    mandatory_selection=[
        MandatorySelectionConfig(strategy="last_minute", threshold=0.9)
    ],
    route_improvement=[
        route improvementConfig(methods=["2opt", "relocate"])
    ]
)
```

### HGSConfig

**File**: `policies/hgs.py`

Hybrid Genetic Search configuration.

```python
@dataclass
class HGSConfig:
    """HGS policy configuration.

    Attributes:
        time_limit: Maximum runtime.
        population_size: Number of individuals.
        elite_size: Number of top individuals preserved.
        mutation_rate: Probability of genetic mutation.
        crossover_type: Crossover operator ('ox', 'pmx').
        local_search_iterations: LS steps per individual.
    """

    time_limit: float = 60.0
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.2
    crossover_type: str = "ox"
    local_search_iterations: int = 100
```

### ACOConfig

**File**: `policies/aco.py`

Ant Colony Optimization configuration.

```python
@dataclass
class ACOConfig:
    """ACO policy configuration.

    Attributes:
        n_ants: Colony size.
        alpha: Pheromone importance.
        beta: Visibility heuristic importance.
        rho: Evaporation rate.
        tau_0: Initial pheromone intensity.
        q0: Exploitation vs exploration balance.
        k_sparse: Limit search to K neighbors.
    """

    n_ants: int = 20
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    tau_0: float = 1.0
    q0: float = 0.9
    k_sparse: int = 15
```

### ILSConfig

**File**: `policies/ils.py`

Iterated Local Search configuration.

```python
@dataclass
class ILSConfig:
    """ILS policy configuration.

    Attributes:
        n_restarts: Macro iterations.
        ls_iterations: Inner local search iterations.
        perturbation_strength: Jump magnitude.
        op_probs: Operator probabilities (swap, 2opt, relocate).
    """

    n_restarts: int = 5
    ls_iterations: int = 50
    perturbation_strength: float = 0.2
    op_probs: Dict[str, float] = field(default_factory=lambda: {
        "swap": 0.3,
        "2opt": 0.4,
        "relocate": 0.3
    })
```

### MandatorySelectionConfig

**File**: `policies/helpers/mandatory_selection.py`

Pre-selection strategy for mandatory collections.

```python
@dataclass
class MandatorySelectionConfig:
    """mandatory selection configuration.

    Attributes:
        strategy: Selector logic ('last_minute', 'regular', 'manager',
                                   'combined', 'revenue', 'service_level').
        threshold: Fill level threshold for last_minute.
        frequency: Collection frequency in days for regular.
        max_fill: Absolute overflow limit (0.0 to 1.0).
        logic: Boolean logic for combined ('and', 'or').
        combined_strategies: List of strategy dicts for combined.

        # Neural Manager Settings (manager strategy)
        hidden_dim: GAT manager embedding dimension.
        lstm_hidden: Temporal hidden dimension.
        history_length: Days of history to consider.
        critical_threshold: Neural policy activation trigger.
        manager_weights: Path to weights file.
        device: Execution hardware.
    """

    strategy: Optional[str] = None
    threshold: float = 0.7
    frequency: int = 3
    max_fill: float = 1.0
    logic: str = "or"
    combined_strategies: Optional[list] = None

    # Manager-specific
    hidden_dim: int = 128
    lstm_hidden: int = 64
    history_length: int = 10
    critical_threshold: float = 0.9
    manager_weights: Optional[str] = None
    device: str = "cuda"
```

**Usage Examples**

```python
# Last-minute collection at 90% fill
mandatory_selection = MandatorySelectionConfig(
    strategy="last_minute",
    threshold=0.9
)

# Regular collection every 3 days
mandatory_selection = MandatorySelectionConfig(
    strategy="regular",
    frequency=3
)

# Combined strategy
mandatory_selection = MandatorySelectionConfig(
    strategy="combined",
    logic="or",
    combined_strategies=[
        {"strategy": "last_minute", "threshold": 0.9},
        {"strategy": "regular", "frequency": 7}
    ]
)

# Neural manager
mandatory_selection = MandatorySelectionConfig(
    strategy="manager",
    manager_weights="assets/manager_weights.pt",
    critical_threshold=0.85
)
```

### route improvementConfig

**File**: `policies/helpers/route_improvement.py`

Solution refinement pipeline.

```python
@dataclass
class route improvementConfig:
    """route improvement configuration.

    Attributes:
        methods: Sequence of methods ('2opt', 'ils', 'split', 'fast_tsp').
        iterations: Max refinement iterations per method.
        n_restarts: Perturbation counts for ILS stages.
        perturbation_strength: Escaping jump magnitude.
        ls_operator: Neighborhood operator.
        time_limit: Maximum time for all refinements.
        params: Method-specific tuning parameters.
    """

    methods: List[str] = field(default_factory=lambda: ["fast_tsp"])
    iterations: int = 50
    n_restarts: int = 5
    perturbation_strength: float = 0.2
    ls_operator: str = "2opt"
    time_limit: float = 10.0
    params: Dict = field(default_factory=dict)
```

**Usage Examples**

```python
# Standard TSP improvement
post_proc = route improvementConfig(methods=["fast_tsp"])

# Multi-stage refinement
post_proc = route improvementConfig(
    methods=["2opt", "relocate", "ils"],
    iterations=100,
    time_limit=30.0
)
```

---

## 7. Reinforcement Learning Configurations

**Directory**: `logic/src/configs/rl/`
**Purpose**: RL algorithms, baselines, and expert policy wrappers

### RLConfig

**File**: `rl/__init__.py`

Central RL algorithm configuration.

```python
@dataclass
class RLConfig:
    """RL algorithm configuration.

    Attributes:
        algorithm: RL algorithm name ('reinforce', 'ppo', 'sapo', 'grpo',
                                      'pomo', 'symnco', 'imitation').
        baseline: Baseline type ('rollout', 'critic', 'pomo', 'warmup', 'none').
        bl_warmup_epochs: Epochs for baseline warmup.
        entropy_weight: Weight for entropy regularization.
        max_grad_norm: Maximum gradient norm for clipping.
        gamma: Discount factor for future rewards.
        exp_beta: Exponential baseline decay.
        bl_alpha: Baseline learning rate.

        # Algorithm-specific sub-configs
        ppo: PPO configuration.
        sapo: SAPO configuration.
        grpo: GRPO configuration.
        pomo: POMO configuration.
        symnco: SymNCO configuration.
        imitation: Imitation learning configuration.
        gdpo: GDPO configuration.
        adaptive_imitation: Adaptive imitation configuration.
    """

    algorithm: str = "reinforce"
    baseline: str = "rollout"
    bl_warmup_epochs: int = 0
    entropy_weight: float = 0.0
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    exp_beta: float = 0.8
    bl_alpha: float = 0.05

    ppo: PPOConfig = field(default_factory=PPOConfig)
    sapo: SAPOConfig = field(default_factory=SAPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    pomo: POMOConfig = field(default_factory=POMOConfig)
    symnco: SymNCOConfig = field(default_factory=SymNCOConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)
    gdpo: GDPOConfig = field(default_factory=GDPOConfig)
    adaptive_imitation: AdaptiveImitationConfig = field(
        default_factory=AdaptiveImitationConfig
    )
```

**Usage Examples**

```python
# REINFORCE with rollout baseline
rl_config = RLConfig(
    algorithm="reinforce",
    baseline="rollout"
)

# PPO with critic baseline
rl_config = RLConfig(
    algorithm="ppo",
    baseline="critic",
    ppo=PPOConfig(epochs=10, eps_clip=0.2)
)

# POMO with entropy regularization
rl_config = RLConfig(
    algorithm="pomo",
    baseline="pomo",
    entropy_weight=0.01,
    pomo=POMOConfig(num_augment=8)
)
```

### PPOConfig

**File**: `rl/core/ppo.py`

Proximal Policy Optimization parameters.

```python
@dataclass
class PPOConfig:
    """PPO specific configuration.

    Attributes:
        epochs: Optimization loops over same batch.
        eps_clip: Probability ratio clipping epsilon.
        value_loss_weight: Weight for Critic regression loss.
        mini_batch_size: Fraction of rollout used per update.
    """

    epochs: int = 10
    eps_clip: float = 0.2
    value_loss_weight: float = 0.5
    mini_batch_size: float = 0.25
```

### POMOConfig

**File**: `rl/core/pomo.py`

Policy Optimization with Multiple Optima.

```python
@dataclass
class POMOConfig:
    """POMO configuration.

    Attributes:
        num_augment: Spatial augmentations (1 to 8).
        num_starts: Force selection from multiple starting nodes.
        augment_fn: Coordinate rotation/reflection method ('dihedral8').
    """

    num_augment: int = 1
    num_starts: Optional[int] = None
    augment_fn: str = "dihedral8"
```

### SymNCOConfig

**File**: `rl/core/symnco.py`

Symmetry-aware Neural Combinatorial Optimization.

```python
@dataclass
class SymNCOConfig:
    """SymNCO configuration.

    Attributes:
        alpha: Weight for problem symmetricity loss.
        beta: Weight for solution symmetricity loss.
    """

    alpha: float = 0.2
    beta: float = 1.0
```

### ImitationConfig

**File**: `rl/core/imitation.py`

Supervised learning from expert policies.

```python
@dataclass
class ImitationConfig:
    """Imitation learning configuration.

    Attributes:
        policy_config: Expert heuristic configuration.
        loss_fn: Objective function ('nll', 'mse').
    """

    policy_config: Any = field(default_factory=HGSConfig)
    loss_fn: str = "nll"
```

### AdaptiveImitationConfig

**File**: `rl/core/adaptive_imitation.py`

Dynamic IL-to-RL transition.

```python
@dataclass
class AdaptiveImitationConfig:
    """Adaptive imitation configuration.

    Attributes:
        il_weight: Initial weight of supervised loss.
        il_decay: Decay factor per training step.
        patience: Epochs to wait before decaying weight.
        threshold: Improvement delta to reset patience.
    """

    il_weight: float = 1.0
    il_decay: float = 0.95
    patience: int = 5
    threshold: float = 0.05
```

---

## 8. Task Configurations

**Directory**: `logic/src/configs/tasks/`
**Purpose**: Workflow orchestration for training, evaluation, simulation, and HPO

### TrainConfig

**File**: `tasks/train.py`

PyTorch Lightning training loop configuration.

```python
@dataclass
class TrainConfig:
    """Training configuration.

    Attributes:
        n_epochs: Total training cycles.
        batch_size: Instances per gradient update.
        train_data_size: Count of instances per epoch.
        val_data_size: Count of validation instances.
        val_dataset: Path to static validation dataset.
        train_dataset: Path to static training dataset.
        num_workers: Parallel data loading workers.
        precision: Numerical precision ('16-mixed', 'bf16-mixed', '32-true').
        accumulation_steps: Gradient accumulation factor.
        checkpoint_epochs: Frequency of saving weights.

        # Time-based training
        train_time: Enable multi-day temporal training.
        eval_time_days: Days for temporal evaluation.

        # route improvement
        route_improvement_epochs: Epochs with refinement training.
        lr_route_improvement: Learning rate for refinement.
        efficiency_weight: Priority for waste/length efficiency.
        overflow_weight: Priority for overflow prevention.

        # Process control
        devices: Number of GPUs or device string.
        strategy: Distributed strategy ('ddp', 'auto').
        resume: Checkpoint path to resume from.
        eval_only: Skip training, only evaluate.
    """

    n_epochs: int = 100
    batch_size: int = 256
    train_data_size: int = 100000
    val_data_size: int = 10000
    val_dataset: Optional[str] = None
    train_dataset: Optional[str] = None
    load_dataset: Optional[str] = None
    num_workers: int = 4
    data_distribution: Optional[str] = None
    precision: str = "16-mixed"

    train_time: bool = False
    eval_time_days: int = 1
    accumulation_steps: int = 1
    enable_scaler: bool = False
    checkpoint_epochs: int = 1
    shrink_size: Optional[int] = None
    route_improvement_epochs: int = 0
    lr_route_improvement: float = 0.001
    efficiency_weight: float = 0.8
    overflow_weight: float = 0.2
    log_step: int = 50

    epoch_start: int = 0
    eval_only: bool = False
    checkpoint_encoder: bool = False
    resume: Optional[str] = None
    logs_dir: Optional[str] = None
    model_weights_path: Optional[str] = None
    final_model_path: Optional[str] = None
    eval_batch_size: int = 256
    persistent_workers: bool = True
    pin_memory: bool = False
    reload_dataloaders_every_n_epochs: int = 1
    devices: Union[int, str] = "auto"
    strategy: Optional[str] = "auto"

    graph: GraphConfig = field(default_factory=GraphConfig)
    reward: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    policy: NeuralConfig = field(default_factory=NeuralConfig)
    callbacks: Optional[List[Any]] = None
```

**Usage Examples**

```python
# Standard training
train_config = TrainConfig(
    n_epochs=100,
    batch_size=256,
    precision="16-mixed"
)

# Multi-GPU distributed training
train_config = TrainConfig(
    devices=4,
    strategy="ddp",
    batch_size=512,
    accumulation_steps=2
)

# Resume from checkpoint
train_config = TrainConfig(
    resume="assets/model_weights/epoch_50.ckpt",
    epoch_start=50
)

# Evaluation only
train_config = TrainConfig(
    eval_only=True,
    val_dataset="data/test_set.pkl"
)
```

### SimConfig

**File**: `tasks/sim.py`

Multi-day simulation configuration.

```python
@dataclass
class SimConfig:
    """Simulation configuration.

    Attributes:
        days: Total simulation duration.
        warmup_days: Initial days excluded from statistics.
        stochastic: Enable random bin fill-rate noise.
        re_planning: Allow mid-simulation routing updates.
        n_samples: Instances to simulate in parallel.
    """

    days: int = 31
    warmup_days: int = 5
    stochastic: bool = True
    re_planning: bool = True
    n_samples: int = 20
```

**Usage Examples**

```python
# Standard month simulation
sim_config = SimConfig(days=31, warmup_days=5)

# Year-long stress test
sim_config = SimConfig(
    days=365,
    stochastic=True,
    n_samples=100
)

# Deterministic scenario
sim_config = SimConfig(
    days=30,
    stochastic=False,
    re_planning=False
)
```

### MetaRLConfig

**File**: `tasks/meta_rl.py`

Meta-learning and hierarchical RL configuration.

```python
@dataclass
class MetaRLConfig:
    """Meta-RL configuration.

    Attributes:
        use_meta: Enable meta-learning wrapper.
        meta_strategy: Strategy ('rnn', 'bandit', 'morl', 'hrl').
        meta_lr: Learning rate for meta-optimizer.
        hrl_threshold: Activation trigger for High-Level Manager.
        gat_hidden_dim: GAT embedding size.
        shared_encoder: Share weights between Worker and Manager.
    """

    use_meta: bool = False
    meta_strategy: str = "rnn"
    meta_lr: float = 1e-3
    hrl_threshold: float = 0.9
    gat_hidden_dim: int = 128
    shared_encoder: bool = True
```

### EvalConfig

**File**: `tasks/eval.py`

Evaluation configuration.

```python
@dataclass
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        dataset: Path to evaluation dataset.
        metrics: Metrics to compute ('gap', 'cost', 'time').
        batch_size: Evaluation batch size.
        num_samples: Number of stochastic samples per instance.
        baseline_policy: Reference policy for gap calculation.
    """

    dataset: Optional[str] = None
    metrics: List[str] = field(default_factory=lambda: ["gap", "cost"])
    batch_size: int = 256
    num_samples: int = 1
    baseline_policy: Optional[str] = None
```

### HPOConfig

**File**: `tasks/hpo.py`

Hyperparameter optimization configuration.

```python
@dataclass
class HPOConfig:
    """HPO configuration.

    Attributes:
        method: HPO method ('optuna', 'ray', 'dehb').
        n_trials: Number of trials to run.
        timeout: Maximum time for HPO.
        pruning: Enable trial pruning.
        search_space: Parameter search space definition.
    """

    method: str = "optuna"
    n_trials: int = 100
    timeout: Optional[float] = None
    pruning: bool = True
    search_space: Dict = field(default_factory=dict)
```

---

## 9. Integration Examples

### Complete Training Pipeline

```python
from logic.src.configs import (
    Config, EnvConfig, ModelConfig, TrainConfig,
    RLConfig, PPOConfig, EncoderConfig
)

# Create comprehensive config for AM training on WCVRP
config = Config(
    task="train",
    seed=42,
    device="cuda:0",
    wandb_mode="online",
    experiment_name="am_wcvrp_100_ppo",

    # Environment
    env=EnvConfig(
        name="cwcvrp",
        num_loc=100,
        capacity=200.0,
        min_fill=0.3,
        max_fill=0.9
    ),

    # Model
    model=ModelConfig(
        name="am",
        encoder=EncoderConfig(
            type="gat",
            embed_dim=128,
            n_layers=3,
            n_heads=8
        )
    ),

    # Training
    train=TrainConfig(
        n_epochs=100,
        batch_size=256,
        precision="16-mixed",
        devices=2,
        strategy="ddp"
    ),

    # RL Algorithm
    rl=RLConfig(
        algorithm="ppo",
        baseline="critic",
        entropy_weight=0.01,
        ppo=PPOConfig(
            epochs=10,
            eps_clip=0.2
        )
    )
)

# Use config in training
from logic.src.pipeline.features.train import train_model
train_model(config)
```

### Evaluation Pipeline

```python
from logic.src.configs import Config, EvalConfig, DecodingConfig

config = Config(
    task="eval",

    eval=EvalConfig(
        dataset="data/test_wcvrp_100.pkl",
        metrics=["gap", "cost", "time"],
        batch_size=512
    ),

    model=ModelConfig(
        load_path="assets/model_weights/best_model.pt",
        decoder=DecoderConfig(
            decoding=DecodingConfig(
                strategy="beam_search",
                beam_width=5
            )
        )
    )
)

from logic.src.pipeline.features.eval import evaluate_model
results = evaluate_model(config)
```

### Multi-Day Simulation

```python
from logic.src.configs import Config, SimConfig, GraphConfig

config = Config(
    task="test_sim",

    env=EnvConfig(
        name="cwcvrp",
        num_loc=150,
        graph=GraphConfig(
            area="riomaior",
            waste_type="plastic"
        )
    ),

    sim=SimConfig(
        days=31,
        warmup_days=5,
        stochastic=True,
        n_samples=50
    )
)

from logic.src.pipeline.features.test import run_simulation
results = run_simulation(config, policies=["gurobi", "hgs", "am"])
```

### ALNS with Pre/route improvement

```python
from logic.src.configs import (
    Config, ALNSConfig, MandatorySelectionConfig, route improvementConfig
)

config = Config(
    env=EnvConfig(name="cwcvrp", num_loc=100),

    policy=ALNSConfig(
        time_limit=120.0,
        max_iterations=10000,
        max_removal_pct=0.4,

        mandatory_selection=[
            MandatorySelectionConfig(
                strategy="combined",
                logic="or",
                combined_strategies=[
                    {"strategy": "last_minute", "threshold": 0.9},
                    {"strategy": "regular", "frequency": 7}
                ]
            )
        ],

        route_improvement=[
            route improvementConfig(
                methods=["2opt", "relocate", "ils"],
                iterations=100,
                time_limit=30.0
            )
        ]
    )
)
```

### Meta-RL Training

```python
from logic.src.configs import Config, MetaRLConfig, TrainConfig

config = Config(
    task="train",

    meta_rl=MetaRLConfig(
        use_meta=True,
        meta_strategy="hrl",
        meta_lr=1e-3,
        hrl_threshold=0.85,
        gat_hidden_dim=128,
        shared_encoder=True
    ),

    train=TrainConfig(
        n_epochs=200,
        train_time=True,
        eval_time_days=10
    )
)
```

### Imitation to RL Transition

```python
from logic.src.configs import (
    Config, RLConfig, AdaptiveImitationConfig,
    ImitationConfig, HGSConfig
)

config = Config(
    rl=RLConfig(
        algorithm="adaptive_imitation",

        imitation=ImitationConfig(
            policy_config=HGSConfig(
                time_limit=60.0,
                population_size=100
            ),
            loss_fn="nll"
        ),

        adaptive_imitation=AdaptiveImitationConfig(
            il_weight=1.0,
            il_decay=0.95,
            patience=5,
            threshold=0.05
        )
    ),

    train=TrainConfig(n_epochs=100)
)
```

---

## 10. Best Practices

### ✅ Good Practices

**Use Factory Defaults**

```python
# ✅ GOOD: Let sub-configs use defaults
config = Config(
    env=EnvConfig(name="wcvrp", num_loc=100)
)
# encoder, decoder, etc. use sensible defaults
```

**Explicit Critical Parameters**

```python
# ✅ GOOD: Override only what matters
config = Config(
    seed=42,  # Reproducibility
    device="cuda:0",  # Hardware
    env=EnvConfig(name="wcvrp", num_loc=100),
    train=TrainConfig(n_epochs=50)
)
```

**Validation Before Use**

```python
# ✅ GOOD: Validate config before expensive operations
def validate_config(config: Config):
    assert config.env.num_loc > 0
    assert config.train.batch_size > 0
    assert config.device in ["cpu", "cuda"] or config.device.startswith("cuda:")

validate_config(config)
train_model(config)
```

**Type Hints for Safety**

```python
# ✅ GOOD: Use type hints
def train_model(config: Config) -> Dict[str, float]:
    """Train model with given config."""
    ...
```

**Nested Config Composition**

```python
# ✅ GOOD: Build configs hierarchically
encoder_config = EncoderConfig(embed_dim=256, n_layers=6)
model_config = ModelConfig(name="am", encoder=encoder_config)
config = Config(model=model_config)
```

### ❌ Anti-Patterns

**Avoid Hardcoding Nested Values**

```python
# ❌ BAD: Hardcoding deep in code
def train():
    embed_dim = 128  # Should be in config
    n_layers = 3
    ...

# ✅ GOOD: Use config
def train(config: Config):
    embed_dim = config.model.encoder.embed_dim
    n_layers = config.model.encoder.n_layers
    ...
```

**Don't Mutate Configs After Creation**

```python
# ❌ BAD: Mutating config is error-prone
config = Config()
config.env.num_loc = 100  # Avoid
config.model.encoder.embed_dim = 256

# ✅ GOOD: Create with all values
config = Config(
    env=EnvConfig(num_loc=100),
    model=ModelConfig(encoder=EncoderConfig(embed_dim=256))
)
```

**Avoid Partial Config Passing**

```python
# ❌ BAD: Passing partial dicts
def setup_encoder(params: dict):
    embed_dim = params.get("embed_dim", 128)
    ...

# ✅ GOOD: Use typed configs
def setup_encoder(config: EncoderConfig):
    embed_dim = config.embed_dim
    ...
```

**Don't Bypass Validation**

```python
# ❌ BAD: Skipping validation
config.env.num_loc = -50  # Invalid!

# ✅ GOOD: Use dataclass __post_init__ for validation
@dataclass
class EnvConfig:
    num_loc: int = 50

    def __post_init__(self):
        assert self.num_loc > 0, "num_loc must be positive"
```

### Configuration Serialization

**Save/Load Configs**

```python
from dataclasses import asdict
import json
import yaml

# Save as JSON
config_dict = asdict(config)
with open("config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

# Load from JSON
with open("config.json", "r") as f:
    config_dict = json.load(f)
config = Config(**config_dict)

# Save as YAML
with open("config.yaml", "w") as f:
    yaml.dump(config_dict, f)
```

**Hydra Integration**

```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # Convert Hydra config to dataclass
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config = Config(**config_dict)

    # Use config
    train_model(config)
```

### Common Pitfalls

**Problem**: Config explosion from too many parameters

```python
# ❌ BAD: 100+ parameters at top level
config = Config(
    seed=42,
    device="cuda",
    env_name="wcvrp",
    env_num_loc=100,
    encoder_type="gat",
    encoder_embed_dim=128,
    encoder_n_layers=3,
    # ... 90 more parameters
)

# ✅ GOOD: Use hierarchical structure
config = Config(
    seed=42,
    device="cuda",
    env=EnvConfig(name="wcvrp", num_loc=100),
    model=ModelConfig(
        encoder=EncoderConfig(
            type="gat",
            embed_dim=128,
            n_layers=3
        )
    )
)
```

**Problem**: Type safety loss from dicts

```python
# ❌ BAD: Using raw dicts
config = {
    "env": {"name": "wcvrp", "num_loc": 100},
    "model": {"encoder": {"embed_dim": 128}}
}
# No IDE autocomplete, no type checking

# ✅ GOOD: Use dataclasses
config = Config(
    env=EnvConfig(name="wcvrp", num_loc=100),
    model=ModelConfig(encoder=EncoderConfig(embed_dim=128))
)
# Full IDE support and type safety
```

---

## 11. Quick Reference

### Common Imports

```python
# Root config
from logic.src.configs import Config

# Environment configs
from logic.src.configs.envs import EnvConfig, GraphConfig, DataConfig, ObjectiveConfig

# Model configs
from logic.src.configs.models import (
    ModelConfig, EncoderConfig, DecoderConfig,
    DecodingConfig, OptimConfig
)

# Policy configs
from logic.src.configs.policies import (
    ALNSConfig, HGSConfig, ACOConfig, ILSConfig, BPCConfig,
    MandatorySelectionConfig, route improvementConfig
)

# RL configs
from logic.src.configs.rl import RLConfig
from logic.src.configs.rl.core import (
    PPOConfig, POMOConfig, SymNCOConfig,
    ImitationConfig, AdaptiveImitationConfig
)

# Task configs
from logic.src.configs.tasks import (
    TrainConfig, EvalConfig, SimConfig,
    MetaRLConfig, HPOConfig
)
```

### Config Hierarchy Reference

```
Config
├── env: EnvConfig
│   ├── graph: GraphConfig
│   └── reward: ObjectiveConfig
├── model: ModelConfig
│   ├── encoder: EncoderConfig
│   │   ├── normalization: NormalizationConfig
│   │   └── activation: ActivationConfig
│   └── decoder: DecoderConfig
│       ├── decoding: DecodingConfig
│       ├── normalization: NormalizationConfig
│       └── activation: ActivationConfig
├── optim: OptimConfig
├── rl: RLConfig
│   ├── ppo: PPOConfig
│   ├── pomo: POMOConfig
│   ├── symnco: SymNCOConfig
│   ├── imitation: ImitationConfig
│   └── adaptive_imitation: AdaptiveImitationConfig
├── train: TrainConfig
│   ├── graph: GraphConfig
│   ├── reward: ObjectiveConfig
│   └── decoding: DecodingConfig
├── eval: EvalConfig
├── sim: SimConfig
├── meta_rl: MetaRLConfig
├── hpo: HPOConfig
├── data: DataConfig
│   └── graph: GraphConfig
├── mandatory_selection: MandatorySelectionConfig
└── route_improvement: route improvementConfig
```

### Default Values Reference

| Config Class             | Key Defaults                                                |
| ------------------------ | ----------------------------------------------------------- |
| **Config**               | task="train", device="cuda", seed=42                        |
| **EnvConfig**            | name="vrpp", num_loc=50, capacity=None                      |
| **GraphConfig**          | area="riomaior", waste_type="plastic", num_loc=50           |
| **ModelConfig**          | name="am", temporal_horizon=0                               |
| **EncoderConfig**        | type="gat", embed_dim=128, n_layers=3, n_heads=8            |
| **DecoderConfig**        | type="attention", embed_dim=128, hidden_dim=512, n_layers=3 |
| **DecodingConfig**       | strategy="greedy", temperature=1.0, beam_width=1            |
| **OptimConfig**          | optimizer="adam", lr=1e-4, lr_scheduler=None                |
| **RLConfig**             | algorithm="reinforce", baseline="rollout"                   |
| **PPOConfig**            | epochs=10, eps_clip=0.2, mini_batch_size=0.25               |
| **POMOConfig**           | num_augment=1, augment_fn="dihedral8"                       |
| **TrainConfig**          | n_epochs=100, batch_size=256, precision="16-mixed"          |
| **SimConfig**            | days=31, warmup_days=5, stochastic=True                     |
| **ALNSConfig**           | time_limit=60.0, max_iterations=5000, cooling_rate=0.995    |
| **HGSConfig**            | population_size=50, elite_size=10, mutation_rate=0.2        |
| **MandatorySelectionConfig**         | strategy=None, threshold=0.7, frequency=3                   |
| **route improvementConfig** | methods=["fast_tsp"], iterations=50                         |

### File Size Reference

| File                        | Lines | Description       |
| --------------------------- | ----- | ----------------- |
| `__init__.py`               | 85    | Root Config class |
| `envs/env.py`               | 36    | EnvConfig         |
| `envs/graph.py`             | 39    | GraphConfig       |
| `envs/data.py`              | 54    | DataConfig        |
| `envs/objective.py`         | ~30   | ObjectiveConfig   |
| `models/model.py`           | 30    | ModelConfig       |
| `models/encoder.py`         | 37    | EncoderConfig     |
| `models/decoder.py`         | ~40   | DecoderConfig     |
| `models/decoding.py`        | 23    | DecodingConfig    |
| `models/optim.py`           | ~25   | OptimConfig       |
| `rl/__init__.py`            | 44    | RLConfig          |
| `rl/core/ppo.py`            | 14    | PPOConfig         |
| `rl/core/pomo.py`           | ~20   | POMOConfig        |
| `policies/alns.py`          | 39    | ALNSConfig        |
| `policies/helpers/mandatory_selection.py` | ~60   | MandatorySelectionConfig      |
| `tasks/train.py`            | 75    | TrainConfig       |
| `tasks/sim.py`              | ~20   | SimConfig         |

### Related Documentation

- [CONSTANTS_MODULE.md](CONSTANTS_MODULE.md) - System constants and mappings
- [UTILS_MODULE.md](UTILS_MODULE.md) - Utility functions and helpers
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [CLAUDE.md](../CLAUDE.md) - Agent instructions and coding standards

---

**Last Updated**: January 2026
**Maintainer**: WSmart+ Route Development Team
**Status**: ✅ Active - Comprehensive configuration system documentation
