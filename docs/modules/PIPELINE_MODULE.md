# Operational Pipelines & Orchestration

**Module**: `logic/src/pipeline`
**Purpose**: Comprehensive technical specification of the WSmart-Route execution engine—handling training, evaluation, and large-scale simulation.
**Version**: 3.0
**Last Updated**: February 2026

## Table of Contents

1.  [**Overview**](#1-overview)
2.  [**Module Architecture**](#2-module-architecture)
3.  [**Callbacks**](#3-callbacks)
4.  [**Features**](#4-features)
5.  [**Reinforcement Learning (RL)**](#5-reinforcement-learning-rl)
6.  [**Simulations**](#6-simulations)
7.  [**Design Patterns**](#7-design-patterns)
8.  [**Usage Examples**](#8-usage-examples)
9.  [**Best Practices**](#9-best-practices)
10. [**Quick Reference**](#10-quick-reference)
11. [**Conclusion**](#11-conclusion)

---

## 1. Overview

The **Pipeline Module** orchestrates the primary operational workflows of the WSmart-Route framework, serving as the execution engine for training, evaluation, simulation, and visualization.

### Core Responsibilities

| Component       | Description                                     |
| --------------- | ----------------------------------------------- |
| **Features**    | Training, evaluation, and testing entry points  |
| **RL**          | Reinforcement learning algorithms and utilities |
| **Simulations** | Physics engine and environment management       |
| **Callbacks**   | PyTorch Lightning monitoring and hooks          |
| **UI**          | Streamlit dashboard for visualization           |

### File Statistics

- **Total Python Files**: 184
- **Lines of Code**: ~25,000+
- **Major Subdirectories**: 4 (callbacks, features, rl, simulations)

---

## 2. Module Architecture

```
logic/src/pipeline/
├── callbacks/              # PyTorch Lightning callbacks (5 files)
│   ├── model_summary.py    # Architecture summary display
│   ├── training_display.py # Rich/Plotext live visualization
│   ├── speed_monitor.py    # Training speed metrics
│   └── reptile.py          # Meta-learning callback
│
├── features/               # Operational workflows (~20 files)
│   ├── train/              # Training engine & model factory
│   ├── eval/               # Evaluation engine & evaluators
│   └── test/               # Simulation testing orchestration
│
├── rl/                     # Reinforcement learning (~40 files)
│   ├── common/             # Base modules, baselines, utilities
│   ├── core/               # 13 RL algorithms
│   ├── meta/               # Meta-learning strategies
│   └── hpo/                # Hyperparameter optimization
│
├── simulations/            # Simulation engine (~80 files)
│   ├── simulator.py        # Top-level orchestration
│   ├── bins/               # Bin state management
│   ├── network/            # Distance calculation strategies
│   ├── actions/            # Command pattern (daily steps)
│   ├── states/             # State machine pattern
│   ├── processor/          # Data formatting
│   ├── repository/         # Data loading
│   └── checkpoints/        # Save/resume simulation
│
    ├── app.py              # Main entry point
    ├── pages/              # Training, simulation, benchmark
    ├── services/           # Data loading services
    ├── components/         # Charts, maps, sidebar
    └── styles/             # Custom CSS
```

### Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Application                        │
│                    (main.py / Studio)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐   ┌───▼────┐   ┌───▼─────┐
    │ Features│   │  RL    │   │ Simulations│
    │ (Train) │   │(Algos) │   │  (Engine)  │
    └────┬────┘   └───┬────┘   └───┬─────┘
         │            │             │
         └────────────┼─────────────┘
                      │
              ┌───────▼────────┐
              │   Callbacks    │
              │ (Monitoring)   │
              └────────────────┘
```

---

## 3. Callbacks

PyTorch Lightning callbacks for training monitoring and visualization.

### Callback Registry

```python
"""
Logic: logic/src/pipeline/callbacks/__init__.py
"""
from .model_summary import ModelSummaryCallback
from .training_display import TrainingDisplayCallback
from .speed_monitor import SpeedMonitor
from .reptile import ReptileCallback

__all__ = [
    "TrainingDisplayCallback",
    "ReptileCallback",
    "SpeedMonitor",
    "ModelSummaryCallback",
]
```

### 1. ModelSummaryCallback

**File**: `callbacks/model_summary.py`

Prints detailed model architecture summary using Rich tables.

**Features**:

- Environment display
- Algorithm identification
- Policy architecture breakdown (encoder/decoder)
- Baseline summary
- mandatory selector details
- Expert policy (for imitation learning)
- Trainable parameter counts

**Usage**:

```python
from logic.src.pipeline.callbacks import ModelSummaryCallback

# Add to trainer callbacks
callbacks = [
    ModelSummaryCallback(),
    # ... other callbacks
]

trainer = WSTrainer(callbacks=callbacks, ...)
```

**Output Example**:

```
┏━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━┓
┃Idx ┃   Name   ┃      Type       ┃ Params ┃ Mode ┃ FLOPs ┃
┡━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━┩
│ 0  │ env      │ VRPP            │ 0      │ train│ 0     │
│ 1  │ algo     │ PPO             │ 0      │ N/A  │ 0     │
│ 2  │ policy   │ AttentionModel  │ 2.1 M  │ train│ 0     │
│ 2a │ encoder  │ GATEncoder (gat)│ 1.4 M  │ train│ 0     │
│ 2b │ decoder  │ AttentionDecoder│ 685 K  │ train│ 0     │
│ 3  │ baseline │ CriticBaseline  │ 142 K  │ train│ 0     │
└────┴──────────┴─────────────────┴────────┴──────┴───────┘
Total Trainable Params: 2.2 M
```

### 2. TrainingDisplayCallback

**File**: `callbacks/training_display.py`

Unified terminal-based training visualization using Rich and Plotext.

**Features**:

- Real-time metric charts (line plots)
- Progress bars for epochs and steps
- Metrics table with current values
- Status header with elapsed time
- Configurable refresh rate and history length
- Multi-metric support with color coding

**Constructor Parameters**:

```python
TrainingDisplayCallback(
    metric_keys: str | List[str] = "train/reward",
    chart_title: str = "Training Progress",
    refresh_rate: int = 4,              # Updates per second
    history_length: Optional[int] = None,  # Max history points
    theme: str = "dark",                # Plotext theme
)
```

**Usage**:

```yaml
# In Hydra config: assets/configs/train.yaml
train:
  callbacks:
    - _target_: logic.src.pipeline.callbacks.TrainingDisplayCallback
      metric_keys: "train/reward,train/loss"
      chart_title: "PPO Training"
      refresh_rate: 4
      history_length: 500
```

**Terminal Output**:

```
 WSmart-Route Training • 01:23:45
┌──────────────────────────────────┬───────────────────┐
│ 📈 Training Chart                │ 📊 Metrics        │
│                                  │ ┏━━━━━━━━━┳━━━━━┓ │
│  20.0 ┤                  ╭╮      │ ┃ Metric  ┃Value┃ │
│  15.0 ┤             ╭────╯╰╮     │ ┡━━━━━━━━━╇━━━━━┩ │
│  10.0 ┤        ╭────╯      ╰─    │ │train/los│0.023│ │
│   5.0 ┤   ╭────╯                 │ │train/rew│18.45│ │
│   0.0 ┤───╯                      │ └─────────┴─────┘ │
│       └─────────────────────────►│                   │
│         0   100   200   300      │                   │
└──────────────────────────────────┴───────────────────┘
⠋ Epoch 5/50 ███████░░░ 5/50 • 00:01:23 • 00:10:52
⠋ Steps 128/512 ██████░░░ 128/512 • 00:00:12 • 00:00:36
```

### 3. SpeedMonitor

**File**: `callbacks/speed_monitor.py`

Tracks training speed metrics (steps/sec, samples/sec, epoch time).

**Usage**:

```python
from logic.src.pipeline.callbacks import SpeedMonitor

callbacks = [SpeedMonitor(epoch_time=True)]
```

### 4. ReptileCallback

**File**: `callbacks/reptile.py`

Implements Reptile meta-learning algorithm for model-agnostic meta-learning.

**Usage**: Specialized for meta-RL training scenarios.

---

## 4. Features

Orchestrates primary operational workflows: training, evaluation, and testing.

### Architecture

```
features/
├── __init__.py
├── train/
│   ├── engine.py              # Training orchestration
│   ├── hpo.py                 # Hyperparameter optimization
│   └── model_factory/         # Model creation registry
│       ├── builder.py         # Main creation logic
│       ├── registry.py        # Algorithm registry
│       ├── ppo.py            # PPO family factory
│       ├── imitation.py      # Imitation learning
│       ├── constructive.py   # POMO/SymNCO
│       └── hrl.py            # Hierarchical RL
│
├── eval/
│   ├── engine.py              # Evaluation orchestration
│   ├── evaluate.py            # Core evaluation logic
│   ├── validation.py          # Validation utilities
│   └── evaluators/            # Decoding strategies
│       ├── greedy.py
│       ├── sampling.py
│       ├── augmentation.py
│       ├── multi_start.py
│       └── combined.py
│
└── test/
    ├── engine.py              # Test runner entry point
    ├── config.py              # Policy config expansion
    └── orchestrator.py        # Multi-seed orchestration
```

### 1. Training Engine

**File**: `features/train/engine.py`

**Key Function**: `run_training(cfg: Config) -> float`

**Workflow**:

1. Set random seed
2. Enable Tensor Core acceleration (if available)
3. Create model via factory
4. Setup callbacks (Hydra instantiation)
5. Configure WSTrainer (PyTorch Lightning wrapper)
6. Run training loop
7. Save final weights
8. Return validation reward

**Example**:

```python
from logic.src.configs import Config
from logic.src.pipeline.features.train.engine import run_training

# Load config via Hydra
cfg = Config()
cfg.seed = 42
cfg.train.n_epochs = 50
cfg.model.type = "attention_model"
cfg.rl.algorithm = "ppo"

# Run training
final_reward = run_training(cfg)
print(f"Final validation reward: {final_reward:.4f}")
```

**Callback Integration**:

```python
# Callbacks are instantiated via Hydra
callbacks = []
for cb_cfg in cfg.train.callbacks:
    if "_target_" in cb_cfg:
        callbacks.append(hydra.utils.instantiate(cb_cfg))

# Link progress bar and chart if both exist
progress_bar = next((c for c in callbacks if c.__class__.__name__ == "CleanProgressBar"), None)
terminal_chart = next((c for c in callbacks if c.__class__.__name__ == "TerminalChartCallback"), None)

if progress_bar and terminal_chart:
    progress_bar.set_chart_callback(terminal_chart)
```

### 2. Model Factory

**File**: `features/train/model_factory/registry.py`

**Algorithm Registry**:

```python
_ALGO_REGISTRY = {
    "ppo": _create_ppo_family,
    "sapo": _create_ppo_family,
    "gspo": _create_ppo_family,
    "dr_grpo": _create_ppo_family,
    "gdpo": _create_gdpo,
    "pomo": _create_pomo,
    "symnco": _create_symnco,
    "hrl": _create_hrl,
    "imitation": _create_imitation,
    "adaptive_imitation": _create_adaptive_imitation,
    "reinforce": _create_reinforce,
}
```

**Factory Usage**:

```python
from logic.src.pipeline.features.train.model_factory.builder import create_model
from logic.src.configs import Config

cfg = Config()
cfg.rl.algorithm = "ppo"
cfg.model.type = "attention_model"
cfg.env.name = "vrpp"

# Factory creates appropriate Lightning module
model = create_model(cfg)
# Returns: PPO(policy=AttentionModel, env=VRPP, ...)
```

### 3. Evaluation Engine

**File**: `features/eval/engine.py`

**Key Functions**:

```python
def eval_dataset(
    dataset_path: str,
    beam_width: int,
    softmax_temp: float,
    opts: Dict[str, Any],
    method: Optional[str] = None,
) -> Tuple[List[float], List[Optional[List[int]]], List[float]]:
    """
    Evaluates a model on a given dataset.

    Returns:
        costs: List of solution costs
        tours: List of solution sequences
        durations: List of inference times
    """
```

**Evaluation Strategies**:

| Strategy         | File                         | Description                          |
| ---------------- | ---------------------------- | ------------------------------------ |
| **Greedy**       | `evaluators/greedy.py`       | Deterministic decoding (argmax)      |
| **Sampling**     | `evaluators/sampling.py`     | Stochastic decoding with temperature |
| **Augmentation** | `evaluators/augmentation.py` | 8-fold symmetry augmentation         |
| **Multi-Start**  | `evaluators/multi_start.py`  | Multiple starting positions (POMO)   |
| **Combined**     | `evaluators/combined.py`     | Multiple strategies aggregation      |

**Usage Example**:

```python
from logic.src.pipeline.features.eval.engine import eval_dataset

opts = {
    "model": "weights/best.pt",
    "val_size": 10000,
    "eval_batch_size": 1024,
    "strategy": "greedy",
    "multiprocessing": False,
    "graph_size": 50,
    "area": "riomaior",
    # ... other options
}

costs, tours, durations = eval_dataset(
    dataset_path="data/vrpp/test.pkl",
    beam_width=1,
    softmax_temp=1.0,
    opts=opts,
)

print(f"Average cost: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
print(f"Average time: {np.mean(durations):.4f}s")
```

**Multiprocessing Support**:

```python
# Enable GPU multiprocessing for faster evaluation
opts["multiprocessing"] = True
opts["val_size"] = 10000  # Must be divisible by num_gpus

# Automatically distributes across all available GPUs
costs, tours, durations = eval_dataset(...)
```

### 4. Test Engine

**File**: `features/test/engine.py`

**Key Function**: `run_simulator_test(opts)`

**Workflow**:

1. Set random seeds
2. Load simulator data (bins, coordinates)
3. Expand policy configurations
4. Create output directories
5. Initialize device
6. Execute multi-seed simulation testing

**Usage**:

```python
from logic.src.pipeline.features.test.engine import run_simulator_test

opts = {
    "seed": 42,
    "size": 50,
    "area": "riomaior",
    "waste_type": "plastic",
    "days": 31,
    "policies": ["gurobi", "hgs", "am_greedy"],
    "n_samples": 10,
    "output_dir": "simulation_results",
    # ... other options
}

run_simulator_test(opts)
```

---

## 5. Reinforcement Learning (RL)

Comprehensive RL infrastructure with 13 algorithms, 8 baselines, meta-learning, and HPO.

### Architecture

```
rl/
├── __init__.py                 # Algorithm registry
├── common/                     # Shared utilities
│   ├── base/                   # RL4COLitModule base class
│   │   ├── model.py            # Main Lightning module
│   │   ├── data.py             # DataMixin (data loading)
│   │   ├── optimization.py     # OptimizationMixin (optimizers)
│   │   └── steps.py            # StepMixin (train/val/test steps)
│   ├── baselines/              # Variance reduction baselines
│   │   ├── base.py             # Abstract baseline
│   │   ├── none.py             # No baseline
│   │   ├── exponential.py      # Moving average
│   │   ├── rollout.py          # Greedy rollout
│   │   ├── critic.py           # Learned value network
│   │   ├── warmup.py           # Gradual transition
│   │   ├── pomo.py             # Multi-start baseline
│   │   ├── mean.py             # Batch mean
│   │   └── shared_critic.py    # Shared critic
│   ├── epoch.py                # Epoch utilities
│   ├── time_training.py        # Temporal training
│   ├── route_improvement.py      # Route refinement
│   └── reward_scaler.py        # Reward normalization
│
├── core/                       # RL algorithms
│   ├── reinforce.py            # REINFORCE (policy gradient)
│   ├── a2c.py                  # Advantage Actor-Critic
│   ├── ppo.py                  # Proximal Policy Optimization
│   ├── sapo.py                 # Self-Adaptive Policy Optimization
│   ├── gspo.py                 # Gradient-Scaled Proxy Optimization
│   ├── gdpo.py                 # Generalized Deviation Policy Optimization
│   ├── dr_grpo.py              # Divergence-Regularized GRPO
│   ├── pomo.py                 # Policy Optimization with Multiple Optima
│   ├── symnco.py               # Symmetry-aware NCO
│   ├── imitation.py            # Imitation Learning
│   ├── adaptive_imitation.py   # IL → RL transition
│   ├── mvmoe_pomo.py           # Multi-Vehicle MoE POMO
│   ├── mvmoe_am.py             # Multi-Vehicle MoE AM
│   ├── stepwise_ppo.py         # Step-wise PPO variant
│   └── losses/                 # Loss functions
│       ├── nll_loss.py
│       ├── weighted_nll_loss.py
│       ├── kl_divergence_loss.py
│       └── js_divergence_loss.py
│
├── meta/                       # Meta-learning
│   ├── module.py               # MetaRLModule
│   ├── hrl.py                  # Hierarchical RL
│   ├── registry.py             # Meta-strategy registry
│   ├── weight_strategy.py      # Weight adjustment strategies
│   ├── weight_optimizer.py     # Gradient-based weight opt
│   ├── td_learning.py          # TD-based cost weights
│   ├── contextual_bandits.py   # UCB/Thompson sampling
│   ├── hypernet_strategy.py    # Hypernetwork for meta-learning
│   └── multi_objective/        # Multi-objective optimization
│       ├── weight_optimizer.py
│       ├── pareto_front.py
│       └── pareto_solution.py
│
└── hpo/                        # Hyperparameter optimization
    ├── optuna_hpo.py           # Optuna integration
    └── dehb.py                 # Differential Evolution Hyperband
```

### RL Algorithm Registry

**File**: `rl/__init__.py`

```python
RL_ALGORITHM_REGISTRY = {
    "reinforce": REINFORCE,
    "ppo": PPO,
    "a2c": A2C,
    "sapo": SAPO,
    "gspo": GSPO,
    "gdpo": GDPO,
    "dr_grpo": DRGRPO,
    "pomo": POMO,
    "symnco": SymNCO,
    "imitation": ImitationLearning,
    "adaptive_imitation": AdaptiveImitation,
    "hrl": HRLModule,
    "meta_rl": MetaRLModule,
    "mvmoe_pomo": MVMoE_POMO,
    "mvmoe_am": MVMoE_AM,
}

def get_rl_algorithm(name: str) -> type:
    """Look up RL algorithm class by name."""
    if name not in RL_ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown RL algorithm: {name}")
    return RL_ALGORITHM_REGISTRY[name]
```

### Base Lightning Module

**File**: `rl/common/base/model.py`

**Class**: `RL4COLitModule`

Inherits from:

- `DataMixin` - Data loading utilities
- `OptimizationMixin` - Optimizer configuration
- `StepMixin` - Training/validation/test step logic
- `pl.LightningModule` - PyTorch Lightning base

**Constructor**:

```python
class RL4COLitModule(DataMixin, OptimizationMixin, StepMixin, pl.LightningModule):
    def __init__(
        self,
        env: IEnv,                          # Problem environment
        policy: IPolicy,                    # Neural policy
        baseline: str = "rollout",          # Baseline type
        optimizer: str = "adam",            # Optimizer name
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
        train_data_size: int = 100000,
        val_data_size: int = 10000,
        val_dataset_path: Optional[str] = None,
        batch_size: int = 256,
        num_workers: int = 4,
        mandatory_selection_selector: Optional[VectorizedSelector] = None,
        **kwargs,
    ):
        # Initialization logic
```

**Key Methods**:

```python
def save_weights(self, path: str):
    """Save model weights and hyperparameters."""
    torch.save({
        "state_dict": self.state_dict(),
        "hparams": self.hparams,
    }, path)
```

### Baselines

**File**: `rl/common/baselines/__init__.py`

**Registry**:

```python
BASELINE_REGISTRY = {
    "none": NoBaseline,
    "exponential": ExponentialBaseline,
    "rollout": RolloutBaseline,
    "critic": CriticBaseline,
    "warmup": WarmupBaseline,
    "pomo": POMOBaseline,
    "mean": MeanBaseline,
    "shared": SharedBaseline,
}

def get_baseline(name: str, policy: Optional[nn.Module] = None, **kwargs) -> Baseline:
    """Factory function for baselines."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}")
    baseline = BASELINE_REGISTRY[name](**kwargs)
    if policy is not None:
        baseline.setup(policy)
    return baseline
```

**Baseline Comparison**:

| Baseline        | Complexity | Variance    | Description                       |
| --------------- | ---------- | ----------- | --------------------------------- |
| **None**        | O(1)       | Highest     | No baseline (REINFORCE)           |
| **Mean**        | O(1)       | High        | Batch mean baseline               |
| **Exponential** | O(1)       | Medium-High | Moving average (β=0.8)            |
| **Rollout**     | O(N²)      | Medium      | Greedy rollout of policy          |
| **Critic**      | O(N)       | Low         | Learned value network V(s)        |
| **Warmup**      | Variable   | Medium      | Transition from rollout to critic |
| **POMO**        | O(N³)      | Lowest      | Multi-start best-of-N             |
| **Shared**      | O(N)       | Low         | Shared critic across tasks        |

**Usage Example**:

```python
from logic.src.pipeline.rl.common.baselines import get_baseline

# Create critic baseline with learned value network
baseline = get_baseline(
    "critic",
    policy=policy,
    embedding_dim=128,
    hidden_dim=128,
)

# Or use exponential moving average
baseline = get_baseline("exponential", beta=0.8)
```

### Core Algorithms

#### 1. REINFORCE

**File**: `rl/core/reinforce.py`

Classic policy gradient with baselines.

```python
from logic.src.pipeline.rl import REINFORCE

model = REINFORCE(
    env=env,
    policy=policy,
    baseline="exponential",
    optimizer="adam",
    optimizer_kwargs={"lr": 1e-4},
)
```

**Algorithm**:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) (R(\tau) - b) \right]
$$

Where:

- $R(\tau)$ = Total reward for trajectory $\tau$
- $b$ = Baseline (e.g., moving average)

#### 2. PPO (Proximal Policy Optimization)

**File**: `rl/core/ppo.py`

Trust-region policy optimization with clipping.

```python
from logic.src.pipeline.rl import PPO

model = PPO(
    env=env,
    policy=policy,
    baseline="critic",
    clip_range=0.2,           # ε for clipping
    entropy_coef=0.01,        # Entropy bonus
    value_coef=0.5,           # Value loss weight
    ppo_epochs=4,             # K epochs per batch
    mini_batch_size=64,       # Sub-batch size
)
```

**Clipped Surrogate Objective**:

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

Where:

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ (probability ratio)
- $\hat{A}_t$ = Advantage estimate
- $\epsilon$ = Clip range (typically 0.2)

#### 3. POMO (Policy Optimization with Multiple Optima)

**File**: `rl/core/pomo.py`

Multi-start decoding for combinatorial optimization.

```python
from logic.src.pipeline.rl import POMO

model = POMO(
    env=env,
    policy=policy,
    baseline="pomo",          # Uses min cost from N starts
    num_starts=50,            # Number of starting positions
    augment=True,             # Use 8-fold augmentation
)
```

**Key Idea**: Start decoding from each node, use best solution as baseline.

$$
b_{\text{POMO}} = \min_{i=1}^N C(\text{decode from node } i)
$$

#### 4. Imitation Learning

**File**: `rl/core/imitation.py`

Supervised learning from expert demonstrations.

```python
from logic.src.pipeline.rl import ImitationLearning

model = ImitationLearning(
    env=env,
    policy=policy,
    expert_policy="hgs",      # Expert algorithm (HGS, Gurobi, etc.)
    expert_kwargs={},         # Expert hyperparameters
    use_nll_loss=True,        # Negative log-likelihood loss
)
```

**Loss Function**:

$$
\mathcal{L}_{IL} = -\mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{expert}}} \left[ \log \pi_\theta(a|s) \right]
$$

#### 5. Adaptive Imitation

**File**: `rl/core/adaptive_imitation.py`

Gradual transition from imitation to RL.

```python
from logic.src.pipeline.rl import AdaptiveImitation

model = AdaptiveImitation(
    env=env,
    policy=policy,
    expert_policy="hgs",
    transition_epochs=10,     # Epochs to transition from IL to RL
    il_weight_start=1.0,      # Initial IL weight
    il_weight_end=0.0,        # Final IL weight (pure RL)
    rl_algorithm="ppo",       # RL algorithm after transition
)
```

**Combined Loss**:

$$
\mathcal{L} = \alpha(t) \cdot \mathcal{L}_{IL} + (1 - \alpha(t)) \cdot \mathcal{L}_{RL}
$$

Where $\alpha(t)$ linearly decays from 1 → 0 over `transition_epochs`.

### Meta-Learning

**File**: `rl/meta/module.py`

**MetaRLModule**: Adapts to distribution shifts via meta-learning strategies.

**Strategies**:

| Strategy               | File                    | Description                                |
| ---------------------- | ----------------------- | ------------------------------------------ |
| **Contextual Bandits** | `contextual_bandits.py` | UCB/Thompson sampling for weight selection |
| **TD Learning**        | `td_learning.py`        | Temporal-difference for cost weights       |
| **Weight Optimizer**   | `weight_optimizer.py`   | Gradient-based weight optimization         |
| **HyperNet**           | `hypernet_strategy.py`  | Context-dependent weight generation        |
| **Multi-Objective**    | `multi_objective/`      | Pareto-optimal weight optimization         |

**Usage**:

```python
from logic.src.pipeline.rl.meta import MetaRLModule

model = MetaRLModule(
    env=env,
    policy=policy,
    meta_strategy="contextual_bandit",
    cost_components=["distance", "waste", "overflows"],
    base_algorithm="ppo",
)
```

### Hyperparameter Optimization

#### Optuna HPO

**File**: `rl/hpo/optuna_hpo.py`

```python
from logic.src.pipeline.rl.hpo import OptunaHPO

hpo = OptunaHPO(
    study_name="vrpp_hpo",
    n_trials=100,
    sampler="tpe",            # TPE, CMA-ES, Random
    pruner="median",          # MedianPruner for early stopping
    direction="maximize",     # Maximize validation reward
)

# Define search space
search_space = {
    "lr": ("float", 1e-5, 1e-3, True),  # (type, low, high, log)
    "clip_range": ("float", 0.1, 0.3),
    "entropy_coef": ("float", 0.0, 0.1),
}

best_params = hpo.optimize(cfg, search_space)
```

#### DEHB (Differential Evolution Hyperband)

**File**: `rl/hpo/dehb.py`

Efficient HPO combining differential evolution with successive halving.

```python
from logic.src.pipeline.rl.hpo import DifferentialEvolutionHyperband

dehb = DifferentialEvolutionHyperband(
    min_budget=1,             # Minimum epochs
    max_budget=50,            # Maximum epochs
    eta=3,                    # Halving parameter
)

best_config = dehb.run(cfg, search_space, n_iterations=50)
```

---

## 6. Simulations

Physics engine for waste collection simulation with realistic bin dynamics.

### Architecture

```
simulations/
├── simulator.py                # Top-level orchestration
├── bins/
│   ├── base.py                 # Bins class (state management)
│   └── prediction.py           # Fill level prediction
├── network/
│   ├── __init__.py             # Distance matrix computation
│   ├── base/                   # Abstract strategy
│   ├── haversine.py            # Great-circle distance
│   ├── geodesic.py             # Vincenty formula
│   ├── euclidean.py            # Planar distance
│   ├── osm.py                  # OpenStreetMap routing
│   ├── google.py               # Google Maps API
│   ├── geopandas.py            # GeoPandas routing
│   └── file.py                 # Load cached matrix
├── actions/
│   ├── base.py                 # SimulationAction abstract
│   ├── fill.py                 # FillAction (waste accumulation)
│   ├── selection.py            # MandatorySelectionSelectionAction
│   ├── policy.py               # PolicyExecutionAction
│   ├── post_process.py         # PostProcessAction
│   ├── collection.py           # CollectAction (emptying bins)
│   └── logging.py              # LogAction (metrics)
├── states/
│   ├── base/
│   │   ├── base.py             # SimState abstract
│   │   └── context.py          # SimulationContext
│   ├── initializing.py         # InitializingState
│   ├── running.py              # RunningState
│   └── finishing.py            # FinishingState
├── processor/
│   ├── formatting.py           # Data formatting
│   └── mapper.py               # Coordinate mapping
├── repository/
│   ├── base.py                 # DataRepository interface
│   └── filesystem.py           # FileSystemRepository
├── checkpoints/
│   └── checkpoint.py           # Save/resume simulation
└── wsmart_bin_analysis/        # Real-world data analysis
    └── export/                 # Production-ready tools
```

### Simulator Orchestration

**File**: `simulations/simulator.py`

**Key Functions**:

```python
def single_simulation(
    opts: Dict[str, Any],
    device: torch.device,
    indices: List[int],
    sample_id: int,
    pol_id: int,
    model_weights_path: str,
    n_cores: int,
) -> Dict[str, Any]:
    """
    Execute single simulation run for one (policy, sample) pair.

    Returns:
        result: Dict with metrics (kg, km, cost, profit, overflows)
    """

def sequential_simulations(
    opts: Dict[str, Any],
    data_size: int,
    device: torch.device,
    pol_id: int,
    n_cores: int,
) -> None:
    """
    Run multiple simulation samples sequentially.

    Aggregates metrics across samples and logs mean/std.
    """
```

**Multiprocessing Support**:

```python
import multiprocessing as mp

def init_single_sim_worker(lock, counter):
    """Initialize shared resources for parallel workers."""
    global _lock, _counter
    _lock = lock
    _counter = counter

# Parallel execution
with mp.Manager() as manager:
    lock = manager.Lock()
    counter = manager.Value('i', 0)

    with mp.Pool(n_cores, initializer=init_single_sim_worker, initargs=(lock, counter)) as pool:
        results = pool.starmap(single_simulation, args_list)
```

### State Machine Pattern

**File**: `simulations/states/base/context.py`

**SimulationContext**: Manages simulation lifecycle through states.

**States**:

| State            | File              | Responsibilities                      |
| ---------------- | ----------------- | ------------------------------------- |
| **Initializing** | `initializing.py` | Load data, create bins, setup network |
| **Running**      | `running.py`      | Execute daily simulation loop         |
| **Finishing**    | `finishing.py`    | Aggregate metrics, log results        |

**State Transitions**:

```
   ┌─────────────────┐
   │  Initializing   │
   │  - Load data    │
   │  - Create bins  │
   │  - Setup network│
   └────────┬────────┘
            │ handle()
            ▼
   ┌─────────────────┐
   │    Running      │
   │  - Daily loop   │◄──┐
   │  - Fill bins    │   │ next_day()
   │  - Select bins  │   │
   │  - Route        │───┘
   │  - Collect      │
   │  - Log          │
   └────────┬────────┘
            │ handle()
            ▼
   ┌─────────────────┐
   │   Finishing     │
   │  - Aggregate    │
   │  - Save results │
   │  - Cleanup      │
   └─────────────────┘
```

**Usage**:

```python
from logic.src.pipeline.simulations.states import SimulationContext

context = SimulationContext(
    opts=opts,
    device=device,
    indices=indices,
    sample_id=0,
    pol_id=0,
    model_weights_path="weights/best.pt",
    variables_dict={"lock": lock, "counter": counter},
)

# State machine executes automatically
result = context.result  # Final metrics dict
```

### Bins Management

**File**: `simulations/bins/base.py`

**Bins Class**: Manages bin state and waste accumulation.

**Attributes**:

```python
class Bins:
    c: np.ndarray          # Current fill levels [0, 1]
    h: np.ndarray          # Historical fill levels (T × N)
    rates: np.ndarray      # Daily accumulation rates
    overflows: int         # Total overflow count
    kg_lost: float         # Waste lost from overflows
```

**Key Methods**:

```python
def fill(self, day: int, distribution: str = "empirical"):
    """
    Simulate waste accumulation for one day.

    Args:
        day: Current simulation day
        distribution: "empirical" or "stochastic"
    """
    if distribution == "empirical":
        # Use historical data
        self.c += self.h[day]
    else:
        # Sample from learned distribution
        self.c += np.random.gamma(shape=self.rates, scale=1.0)

    # Detect overflows
    overflow_mask = self.c > 1.0
    self.overflows += overflow_mask.sum()
    self.kg_lost += (self.c[overflow_mask] - 1.0).sum() * bin_capacity
    self.c = np.clip(self.c, 0, 1.0)

def collect(self, indices: List[int]):
    """
    Empty bins at specified indices.

    Args:
        indices: List of bin IDs to collect (0-indexed)
    """
    self.c[indices] = 0.0
```

### Network Distance Calculation

**File**: `simulations/network/__init__.py`

**Function**: `compute_distance_matrix(coords: pd.DataFrame, method: str, **kwargs) -> np.ndarray`

**Available Strategies**:

| Method    | Strategy             | Description           | Use Case                     |
| --------- | -------------------- | --------------------- | ---------------------------- |
| **hsd**   | `HaversineStrategy`  | Great-circle distance | Fast, small errors (<1%)     |
| **gdsc**  | `GeodesicStrategy`   | Vincenty formula      | Accurate, medium speed       |
| **ogd**   | `EuclideanStrategy`  | Planar distance       | Very fast, virtual instances |
| **osm**   | `OSMStrategy`        | OpenStreetMap routing | Real road networks           |
| **gmaps** | `GoogleMapsStrategy` | Google Maps API       | Production-quality           |
| **gpd**   | `GeoPandasStrategy`  | GeoPandas routing     | Offline road networks        |
| **file**  | `FileStrategy`       | Load cached matrix    | Pre-computed distances       |

**Usage Example**:

```python
import pandas as pd
from logic.src.pipeline.simulations.network import compute_distance_matrix

# Load bin coordinates
coords = pd.DataFrame({
    "ID": [0, 1, 2, 3],
    "latitude": [39.2, 39.3, 39.25, 39.35],
    "longitude": [-8.9, -8.85, -8.92, -8.88],
})

# Compute distance matrix
distance_matrix = compute_distance_matrix(
    coords,
    method="hsd",  # Haversine
    dm_filepath="riomaior_50_hsd.csv",  # Cache to file
)

# Result: 4×4 symmetric matrix in kilometers
print(distance_matrix)
# [[0.0, 12.3, 5.6, 16.8],
#  [12.3, 0.0, 14.2, 8.9],
#  [5.6, 14.2, 0.0, 18.1],
#  [16.8, 8.9, 18.1, 0.0]]
```

**Caching**:

```python
# First call: compute and save
dm = compute_distance_matrix(coords, "osm", dm_filepath="cache.csv")

# Subsequent calls: load from cache (fast!)
dm = compute_distance_matrix(coords, "file", dm_filepath="cache.csv")
```

### Action Pattern

**File**: `simulations/actions/base.py`

**Abstract Base**:

```python
class SimulationAction(ABC):
    """Base class for simulation actions."""

    @abstractmethod
    def execute(self, context: SimulationContext, **kwargs) -> Any:
        """Execute the action."""
        pass
```

**Concrete Actions**:

#### 1. FillAction

**File**: `actions/fill.py`

```python
class FillAction(SimulationAction):
    def execute(self, context: SimulationContext, day: int):
        """Simulate waste accumulation for current day."""
        context.bins.fill(day, context.opts["data_distribution"])
```

#### 2. MandatorySelectionSelectionAction

**File**: `actions/selection.py`

```python
class MandatorySelectionSelectionAction(SimulationAction):
    def execute(self, context: SimulationContext, **kwargs):
        """Select bins that must be collected."""
        selector = context.mandatory_selection_selector
        mandatory_selection_indices = selector.select(context.bins, context.opts)
        return mandatory_selection_indices
```

#### 3. PolicyExecutionAction

**File**: `actions/policy.py`

```python
class PolicyExecutionAction(SimulationAction):
    def execute(self, context: SimulationContext, mandatory_selection: List[int], **kwargs):
        """Execute routing policy to generate tour."""
        policy = context.policy_adapter
        tour, cost, metadata = policy.execute(
            mandatory_selection=mandatory_selection,
            bins=context.bins,
            distance_matrix=context.distance_matrix,
            area=context.opts["area"],
            waste_type=context.opts["waste_type"],
            config=context.opts.get("policy_config", {}),
        )
        return tour, cost
```

#### 4. CollectAction

**File**: `actions/collection.py`

```python
class CollectAction(SimulationAction):
    def execute(self, context: SimulationContext, tour: List[int]):
        """Empty bins along the tour."""
        # Extract unique bin IDs from tour (excluding depot 0)
        unique_bins = list(set(tour) - {0})

        # Calculate waste collected
        kg_collected = sum(context.bins.c[i] * BIN_CAPACITY for i in unique_bins)

        # Empty bins
        context.bins.collect(unique_bins)

        return kg_collected
```

#### 5. LogAction

**File**: `actions/logging.py`

```python
class LogAction(SimulationAction):
    def execute(self, context: SimulationContext, metrics: Dict[str, float]):
        """Log daily metrics to file."""
        log_entry = {
            "day": context.current_day,
            "policy": context.pol_name,
            **metrics,
        }

        with context.lock:
            with open(context.log_path, "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
```

**Daily Execution Chain**:

```python
# In RunningState.handle()
actions = [
    FillAction(),
    MandatorySelectionSelectionAction(),
    PolicyExecutionAction(),
    CollectAction(),
    LogAction(),
]

for day in range(num_days):
    for action in actions:
        result = action.execute(context, ...)
```

---

## 7. Design Patterns

### 1. Factory Pattern

**Location**: `features/train/model_factory/`

**Purpose**: Create appropriate Lightning modules based on algorithm name.

**Example**:

```python
# Registry-based factory
_ALGO_REGISTRY = {
    "ppo": _create_ppo_family,
    "pomo": _create_pomo,
    "imitation": _create_imitation,
}

def create_model(cfg: Config) -> RL4COLitModule:
    """Factory method for model creation."""
    algo_name = cfg.rl.algorithm
    if algo_name not in _ALGO_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    factory_fn = _ALGO_REGISTRY[algo_name]
    return factory_fn(cfg, policy, env, kwargs)
```

### 2. Strategy Pattern

**Location**: `simulations/network/base/`

**Purpose**: Pluggable distance calculation strategies.

**Example**:

```python
class DistanceStrategy(ABC):
    @abstractmethod
    def calculate(self, coords: pd.DataFrame, **kwargs) -> np.ndarray:
        pass

class HaversineStrategy(DistanceStrategy):
    def calculate(self, coords, **kwargs):
        # Great-circle distance implementation
        pass

class OSMStrategy(DistanceStrategy):
    def calculate(self, coords, **kwargs):
        # OpenStreetMap routing implementation
        pass

# Usage
STRATEGIES = {
    "hsd": HaversineStrategy,
    "osm": OSMStrategy,
}

strategy = STRATEGIES[method]()
distance_matrix = strategy.calculate(coords)
```

### 3. State Pattern

**Location**: `simulations/states/`

**Purpose**: Manage simulation lifecycle through distinct states.

**Example**:

```python
class SimState(ABC):
    @abstractmethod
    def handle(self, context: SimulationContext) -> None:
        pass

class InitializingState(SimState):
    def handle(self, context):
        # Load data, create bins, setup network
        context.current_state = RunningState()

class RunningState(SimState):
    def handle(self, context):
        # Execute daily simulation loop
        if context.current_day >= context.opts["days"]:
            context.current_state = FinishingState()

class FinishingState(SimState):
    def handle(self, context):
        # Aggregate metrics, save results
        pass

# Usage
context = SimulationContext(...)
while context.current_state:
    context.current_state.handle(context)
```

### 4. Command Pattern

**Location**: `simulations/actions/`

**Purpose**: Encapsulate daily simulation steps as objects.

**Example**:

```python
class SimulationAction(ABC):
    @abstractmethod
    def execute(self, context: SimulationContext, **kwargs) -> Any:
        pass

# Concrete commands
fill_action = FillAction()
selection_action = MandatorySelectionSelectionAction()
policy_action = PolicyExecutionAction()
collect_action = CollectAction()
log_action = LogAction()

# Execute sequence
actions = [fill_action, selection_action, policy_action, collect_action, log_action]
for action in actions:
    result = action.execute(context, ...)
```

### 5. Mixin Pattern

**Location**: `rl/common/base/`

**Purpose**: Compose Lightning module from reusable components.

**Example**:

```python
class DataMixin:
    """Data loading utilities."""
    def setup_data(self, stage):
        pass

class OptimizationMixin:
    """Optimizer configuration."""
    def configure_optimizers(self):
        pass

class StepMixin:
    """Training/validation/test step logic."""
    def training_step(self, batch, batch_idx):
        pass

class RL4COLitModule(DataMixin, OptimizationMixin, StepMixin, pl.LightningModule):
    """Composed Lightning module with all capabilities."""
    pass
```

### 6. Registry Pattern

**Location**: `rl/__init__.py`, `rl/common/baselines/__init__.py`

**Purpose**: Dynamic algorithm/baseline discovery and instantiation.

**Example**:

```python
RL_ALGORITHM_REGISTRY = {
    "reinforce": REINFORCE,
    "ppo": PPO,
    "pomo": POMO,
}

def get_rl_algorithm(name: str) -> type:
    if name not in RL_ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}")
    return RL_ALGORITHM_REGISTRY[name]

# Usage
algo_cls = get_rl_algorithm("ppo")
model = algo_cls(env=env, policy=policy, ...)
```

---

## 8. Usage Examples

### Example 1: Train PPO Model

```python
from logic.src.configs import Config
from logic.src.pipeline.features.train.engine import run_training

# Create configuration
cfg = Config()
cfg.seed = 42
cfg.device = "cuda"

# Environment
cfg.env.name = "vrpp"
cfg.env.num_loc = 50

# Model
cfg.model.type = "attention_model"
cfg.model.encoder.type = "gat"
cfg.model.decoder.type = "attention"

# RL Algorithm
cfg.rl.algorithm = "ppo"
cfg.rl.clip_range = 0.2
cfg.rl.entropy_coef = 0.01
cfg.rl.ppo_epochs = 4

# Training
cfg.train.n_epochs = 50
cfg.train.batch_size = 256
cfg.train.train_data_size = 100000
cfg.train.val_data_size = 10000

# Baseline
cfg.rl.baseline = "critic"

# Run training
final_reward = run_training(cfg)
print(f"Final validation reward: {final_reward:.4f}")
```

### Example 2: Evaluate Model

```python
from logic.src.pipeline.features.eval.engine import eval_dataset

opts = {
    "model": "weights/vrpp_ppo_best.pt",
    "val_size": 10000,
    "eval_batch_size": 1024,
    "strategy": "greedy",
    "graph_size": 50,
    "area": "riomaior",
    "multiprocessing": False,
    "offset": 0,
    "results_dir": "results",
    "output_filename": None,
    "overwrite": False,
    # ... other options
}

costs, tours, durations = eval_dataset(
    dataset_path="data/vrpp/test.pkl",
    beam_width=1,
    softmax_temp=1.0,
    opts=opts,
)

import numpy as np
print(f"Average cost: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
print(f"Average inference time: {np.mean(durations):.4f}s")
```

### Example 3: Run Simulation

```python
from logic.src.pipeline.features.test.engine import run_simulator_test

opts = {
    "seed": 42,
    "size": 50,
    "area": "riomaior",
    "waste_type": "plastic",
    "days": 31,
    "n_samples": 10,
    "policies": [
        "gurobi_empirical",
        "hgs_empirical",
        "am_greedy_empirical",
    ],
    "data_distribution": "empirical",
    "output_dir": "simulation_results",
    "checkpoint_dir": "checkpoints",
    "distance_method": "hsd",
    "focus_graph": None,
    # ... other options
}

run_simulator_test(opts)

# Results saved to:
# - assets/simulation_results/31_days/riomaior_50/log_mean_10N.json
# - assets/simulation_results/31_days/riomaior_50/log_std_10N.json
```

### Example 4: Imitation Learning

```python
from logic.src.configs import Config
from logic.src.pipeline.features.train.engine import run_training

cfg = Config()
cfg.rl.algorithm = "imitation"
cfg.rl.expert_policy = "hgs"  # Learn from HGS expert
cfg.rl.expert_kwargs = {"time_limit": 10}
cfg.train.n_epochs = 20

# Run imitation learning
run_training(cfg)
```

### Example 5: Adaptive Imitation → RL

```python
cfg = Config()
cfg.rl.algorithm = "adaptive_imitation"
cfg.rl.expert_policy = "hgs"
cfg.rl.transition_epochs = 10  # Transition from IL to RL
cfg.rl.il_weight_start = 1.0
cfg.rl.il_weight_end = 0.0
cfg.rl.rl_algorithm = "ppo"  # Use PPO after transition
cfg.train.n_epochs = 50

run_training(cfg)
```

### Example 6: Meta-RL Training

```python
cfg = Config()
cfg.rl.algorithm = "meta_rl"
cfg.rl.meta_strategy = "contextual_bandit"
cfg.rl.cost_components = ["distance", "waste", "overflows"]
cfg.rl.base_algorithm = "ppo"
cfg.train.n_epochs = 100

run_training(cfg)
```

### Example 7: Hyperparameter Optimization

```python
from logic.src.pipeline.rl.hpo import OptunaHPO
from logic.src.configs import Config

cfg = Config()

hpo = OptunaHPO(
    study_name="vrpp_ppo_hpo",
    n_trials=100,
    sampler="tpe",
    pruner="median",
)

search_space = {
    "rl.lr": ("float", 1e-5, 1e-3, True),  # log scale
    "rl.clip_range": ("float", 0.1, 0.3),
    "rl.entropy_coef": ("float", 0.0, 0.1),
    "train.batch_size": ("categorical", [128, 256, 512]),
}

best_params = hpo.optimize(cfg, search_space)
print(f"Best hyperparameters: {best_params}")
```

### Example 8: POMO Multi-Start Training

```python
cfg = Config()
cfg.rl.algorithm = "pomo"
cfg.rl.num_starts = 50  # Decode from 50 different start positions
cfg.rl.baseline = "pomo"  # Use min cost as baseline
cfg.rl.augment = True  # 8-fold augmentation
cfg.train.n_epochs = 100

run_training(cfg)
```

### Example 9: Custom Distance Matrix

```python
from logic.src.pipeline.simulations.network import compute_distance_matrix
import pandas as pd

# Load bin coordinates
coords = pd.read_csv("data/riomaior_bins.csv")

# Compute using OpenStreetMap
distance_matrix = compute_distance_matrix(
    coords,
    method="osm",
    dm_filepath="riomaior_osm.csv",  # Cache result
    verbose=True,
)

# Use in simulation
opts["distance_matrix"] = distance_matrix
run_simulator_test(opts)
```

### Example 10: WSmart-Route Studio

```bash
# Launch the Studio desktop app (dev mode)
just studio

# Build a release bundle
just studio-build
```

---

## 9. Best Practices

### Training

1. **Use Callbacks for Monitoring**

   ```python
   callbacks = [
       ModelSummaryCallback(),
       TrainingDisplayCallback(metric_keys="train/reward,val/reward"),
       SpeedMonitor(epoch_time=True),
   ]
   ```

2. **Enable Mixed Precision for Speed**

   ```python
   cfg.train.precision = "16-mixed"  # or "bf16-mixed" for Ampere+
   torch.set_float32_matmul_precision("medium")
   ```

3. **Use Warmup Baseline for Stability**

   ```python
   cfg.rl.baseline = "warmup"  # Start with rollout, transition to critic
   cfg.rl.warmup_epochs = 5
   ```

4. **Save Checkpoints Regularly**
   ```python
   cfg.train.model_weights_path = "weights/experiment_name"
   cfg.train.checkpoint_every_n_epochs = 5
   ```

### Evaluation

1. **Use Augmentation for Better Solutions**

   ```python
   opts["strategy"] = "augmentation"  # 8-fold symmetry
   opts["augment_samples"] = 8
   ```

2. **Enable Multiprocessing for Speed**

   ```python
   opts["multiprocessing"] = True
   opts["eval_batch_size"] = 1024
   # Automatically uses all GPUs
   ```

3. **Auto-Tune Batch Size**
   ```python
   opts["auto_batch_size"] = True
   # Automatically finds maximum batch size
   ```

### Simulation

1. **Cache Distance Matrices**

   ```python
   opts["distance_method"] = "hsd"
   opts["dm_filepath"] = "area_size_hsd.csv"
   # Second run loads from cache (10x faster)
   ```

2. **Use Checkpoints for Long Runs**

   ```python
   opts["checkpoint_enabled"] = True
   opts["checkpoint_dir"] = "checkpoints"
   # Resume from checkpoint if interrupted
   ```

3. **Run Multiple Seeds for Statistical Significance**

   ```python
   opts["n_samples"] = 30  # 30 seeds
   # Reports mean ± std for all metrics
   ```

4. **Use Empirical Distribution for Realism**
   ```python
   opts["data_distribution"] = "empirical"
   # Uses real historical bin fill patterns
   ```

### Hyperparameter Optimization

1. **Start with Coarse Search**

   ```python
   # Phase 1: Coarse
   search_space = {
       "lr": ("float", 1e-5, 1e-3, True),
       "batch_size": ("categorical", [128, 256, 512]),
   }

   # Phase 2: Fine
   search_space = {
       "lr": ("float", best_lr * 0.5, best_lr * 2.0, True),
       "clip_range": ("float", 0.15, 0.25),
   }
   ```

2. **Use Pruning for Efficiency**

   ```python
   hpo = OptunaHPO(
       n_trials=100,
       pruner="median",  # Stop bad trials early
   )
   ```

3. **Parallelize Trials**
   ```python
   # Run 4 trials in parallel
   hpo.optimize(cfg, search_space, n_jobs=4)
   ```

### Performance Optimization

1. **Batch Size Guidelines**
   | GPU VRAM | Batch Size | Graph Size |
   |----------|------------|------------|
   | 8 GB | 128 | 50 |
   | 12 GB | 256 | 50 |
   | 16 GB | 512 | 50 |
   | 24 GB | 1024 | 100 |

2. **Enable Persistent Workers**

   ```python
   cfg.train.num_workers = 4
   cfg.train.persistent_workers = True  # Faster data loading
   ```

3. **Use Pin Memory on GPU**
   ```python
   cfg.train.pin_memory = True  # Faster CPU → GPU transfer
   ```

### Debugging

1. **Check Model Summary**

   ```python
   callbacks = [ModelSummaryCallback()]
   # Prints architecture, parameter counts
   ```

2. **Monitor Speed**

   ```python
   callbacks = [SpeedMonitor(epoch_time=True)]
   # Logs steps/sec, samples/sec
   ```

3. **Validate Gradients**
   ```python
   cfg.train.gradient_clip_val = 1.0  # Clip gradients
   cfg.train.detect_anomaly = True    # NaN detection
   ```

---

## 10. Quick Reference

### Algorithm Selection Guide

| Problem Characteristics   | Recommended Algorithm            |
| ------------------------- | -------------------------------- |
| Small graphs (<50 nodes)  | **POMO** (multi-start)           |
| Large graphs (>100 nodes) | **PPO** (scalable)               |
| Need fast training        | **Imitation** (from HGS)         |
| Need best quality         | **Adaptive Imitation** → **PPO** |
| Multiple distributions    | **Meta-RL**                      |
| Hierarchical decisions    | **HRL**                          |

### Baseline Selection Guide

| Training Stage              | Recommended Baseline               |
| --------------------------- | ---------------------------------- |
| Early training (0-5 epochs) | **Rollout** or **Exponential**     |
| Mid training (5-20 epochs)  | **Warmup** (rollout → critic)      |
| Late training (20+ epochs)  | **Critic**                         |
| POMO algorithm              | **POMO** (multi-start min)         |
| Very large graphs           | **Mean** or **Exponential** (fast) |

### Distance Method Selection

| Scenario            | Method           | Accuracy | Speed     |
| ------------------- | ---------------- | -------- | --------- |
| Virtual instances   | `ogd`            | N/A      | Fastest   |
| Small graphs (<100) | `hsd`            | ~99%     | Very Fast |
| Large graphs        | `hsd`            | ~99%     | Fast      |
| Production quality  | `osm` or `gmaps` | 100%     | Slow      |
| Offline routing     | `gpd`            | ~99%     | Medium    |

### Simulation Policy Naming Convention

```
<policy_name>_[<engine>]_[<threshold>]_<distribution>

Examples:
- gurobi_empirical
- hgs_empirical
- ms_last_minute_cf70_empirical  (mandatory last-minute, 70% capacity)
- am_greedy_empirical
- tsp_empirical
```

### Key Configuration Files

| Config Type       | Path                                       |
| ----------------- | ------------------------------------------ |
| **Training**      | `assets/configs/train.yaml`                |
| **RL Algorithms** | `assets/configs/rl/core/*.yaml`            |
| **Models**        | `assets/configs/models/*.yaml`             |
| **Environments**  | `assets/configs/envs/*.yaml`               |
| **Policies**      | `assets/configs/policies/*.yaml`           |
| **Callbacks**     | Defined in `train.yaml` under `callbacks:` |

### Logging and Outputs

| Output Type            | Location                                         |
| ---------------------- | ------------------------------------------------ |
| **Training Logs**      | `logs/<experiment_name>/`                        |
| **Model Weights**      | `weights/<experiment_name>/`                     |
| **Simulation Results** | `assets/<output_dir>/<days>_days/<area>_<size>/` |
| **Distance Matrices**  | `data/simulator/distance_matrix/`            |
| **Checkpoints**        | `<checkpoint_dir>/`                              |

### Important Metrics

#### Training Metrics

| Metric            | Description                      | Good Value |
| ----------------- | -------------------------------- | ---------- |
| `train/reward`    | Negative cost (higher is better) | Increasing |
| `train/loss`      | Policy loss                      | Decreasing |
| `val/reward`      | Validation performance           | Increasing |
| `train/entropy`   | Policy entropy                   | 0.01-0.1   |
| `train/grad_norm` | Gradient norm                    | <10.0      |

#### Simulation Metrics

| Metric      | Description               | Unit  |
| ----------- | ------------------------- | ----- |
| `kg`        | Waste collected           | kg    |
| `km`        | Distance traveled         | km    |
| `cost`      | Total cost                | €     |
| `profit`    | Revenue - Cost            | €     |
| `overflows` | Number of overflows       | count |
| `kg_lost`   | Waste lost from overflows | kg    |
| `ncol`      | Number of collections     | count |
| `kg/km`     | Collection efficiency     | kg/km |

### CLI Commands

```bash
# Training
python main.py train_lightning model=am env.name=vrpp rl.algorithm=ppo

# Evaluation
python main.py eval data/vrpp/test.pkl --model weights/best.pt --strategy greedy

# Simulation
python main.py test_sim --policies gurobi hgs am_greedy --days 31 --n_samples 10

# Hyperparameter Optimization
python main.py train_lightning experiment=hpo env.name=vrpp

# Launch the Studio desktop app
just studio
```

### Common Issues and Solutions

| Issue                    | Solution                                                     |
| ------------------------ | ------------------------------------------------------------ |
| **CUDA OOM**             | Reduce `batch_size` or `eval_batch_size`                     |
| **Slow training**        | Enable `persistent_workers=True`, use `precision="16-mixed"` |
| **NaN loss**             | Reduce learning rate, enable gradient clipping               |
| **Poor convergence**     | Use warmup baseline, increase training epochs                |
| **Simulation errors**    | Check policy string format, verify data paths                |
| **Distance matrix slow** | Cache to file with `dm_filepath`, use `hsd` instead of `osm` |

### Performance Benchmarks

**Training Speed** (50-node VRPP, RTX 3090 Ti):

| Configuration   | Samples/sec | Time/Epoch |
| --------------- | ----------- | ---------- |
| Batch 128, FP32 | 2,500       | 40s        |
| Batch 256, FP32 | 4,200       | 24s        |
| Batch 512, FP16 | 8,500       | 12s        |

**Evaluation Speed** (10K instances, RTX 3090 Ti):

| Strategy           | Time  | Quality Gap   |
| ------------------ | ----- | ------------- |
| Greedy             | 45s   | 0% (baseline) |
| Sampling (N=10)    | 6min  | -2%           |
| Augmentation (8×)  | 5min  | -3%           |
| Multi-Start (N=50) | 40min | -5%           |

**Simulation Speed** (50 bins, 31 days, 10 samples):

| Policy      | Time/Sample | Total Time |
| ----------- | ----------- | ---------- |
| TSP         | 2s          | 20s        |
| HGS         | 5s          | 50s        |
| Gurobi      | 15s         | 150s       |
| Neural (AM) | 0.5s        | 5s         |

---

## 11. Conclusion

The Pipeline Module is the execution engine of WSmart-Route, orchestrating training, evaluation, simulation, and visualization workflows. With 184 files spanning callbacks, features, RL algorithms, simulation physics, and UI components, it provides a comprehensive MLOps infrastructure for routing optimization research and deployment.

**Key Highlights**:

- **13 RL Algorithms**: From REINFORCE to Meta-RL
- **8 Baselines**: Variance reduction strategies
- **State Machine**: Robust simulation lifecycle management
- **8 Distance Strategies**: From Haversine to Google Maps
- **Streamlit Dashboard**: Real-time monitoring and visualization
- **Rich Callbacks**: Terminal-based training visualization

For detailed information on specific components:

- **Configs**: See `docs/CONFIGS_MODULE.md`
- **Policies**: See `docs/POLICIES_MODULE.md`
- **Models**: See `docs/MODELS_MODULE.md`
- **Environments**: See `docs/ENVS_MODULE.md`

---

**Last Updated**: February 2026
**Maintainer**: WSmart-Route Team
**License**: Proprietary
