# Abstract Protocols & System Interfaces

> **Status (Q-22):** narrative guide. The API reference is generated into
> `docs/api/` by `tools/gen_api_docs.py`; this file is not the source of
> truth for signatures and module inventories.


**Module**: `logic/src/interfaces`
**Purpose**: Comprehensive technical specification of abstract protocols and interfaces ensuring modularity, decoupling, and type safety.
**Version**: 3.0
**Last Updated**: February 2026

---

## Table of Contents

1.  [**Overview**](#1-overview)
2.  [**Module Organization**](#2-module-organization)
3.  [**Environment Interface**](#3-environment-interface)
4.  [**Model Interface**](#4-model-interface)
5.  [**Policy Interface**](#5-policy-interface)
6.  [**Policy Adapter Interface**](#6-policy-adapter-interface)
7.  [**packing Interface**](#7-packing-interface)
8.  [**route improvement Interface**](#8-route improvement-interface)
9.  [**Infrastructure & Utility Protocols**](#9-infrastructure--utility-protocols)
10. [**Integration Examples**](#10-integration-examples)
11. [**Best Practices**](#11-best-practices)
12. [**Quick Reference**](#12-quick-reference)

---

## 1. Overview

The `logic/src/interfaces` module defines abstract protocols and interfaces that enforce clear contracts between components. By defining these boundaries, the system remains agnostic to specific implementations, enabling seamless swapping of components (e.g., replacing a neural policy with a classical heuristic without changing the calling code).

### Key Features

- **Protocol-Based Design**: Uses `typing.Protocol` for structural subtyping
- **Type Safety**: Static type checking with mypy compatibility
- **Decoupling**: Clear boundaries between components
- **Testability**: Easy mocking and unit testing
- **Extensibility**: Simple to add new implementations
- **Interoperability**: Consistent interfaces across neural and classical methods

### Design Principles

1. **Liskov Substitution**: Any implementation can replace another
2. **Interface Segregation**: Focused, minimal interfaces
3. **Dependency Inversion**: Depend on abstractions, not concretions
4. **Single Responsibility**: Each interface has one clear purpose

### Import Strategy

```python
# Import specific interfaces
from logic.src.interfaces import IEnv, IModel, IPolicy

# Import all interfaces
from logic.src.interfaces import (
    IEnv,
    IModel,
    IPolicy,
    IPolicyAdapter,
    IPackingAlgorithm,
    Iroute improvement
)
```

---

## 2. Module Organization

```
logic/src/interfaces/
├── __init__.py              # Module exports
├── env.py                   # IEnv protocol
├── model.py                 # IModel protocol
├── policy.py                # IPolicy abstract base class
├── adapter.py               # IPolicyAdapter protocol
├── packing.py               # IPackingAlgorithm protocol
└── route_improvement.py       # Iroute improvement protocol
```

### Interface Hierarchy

```
┌─────────────────────────────────────────────────────┐
│                   Core Interfaces                    │
├─────────────────────────────────────────────────────┤
│ IEnv            │ IModel          │ IPolicy          │
│ (Environment)   │ (Neural Net)    │ (Solver)         │
└─────────────────┴─────────────────┴──────────────────┘
         │                │                │
         ▼                ▼                ▼
┌─────────────────────────────────────────────────────┐
│               Specialized Interfaces                 │
├─────────────────────────────────────────────────────┤
│ IPolicyAdapter  │ IPackingAlgorithm │ Iroute improvement │
│ (Bridge)        │ (Pre-selection)  │ (Refinement)    │
└─────────────────────────────────────────────────────┘
```

---

## 3. Environment Interface

**File**: `logic/src/interfaces/env.py`

The `IEnv` protocol standardizes how Reinforcement Learning (RL) environments interact with trainers, evaluators, and simulators. It is designed for compatibility with the **RL4CO** standard.

### Protocol Definition

```python
from typing import Protocol, Optional, Any
import torch
from tensordict import TensorDict

class IEnv(Protocol):
    """Protocol for RL environments."""

    @property
    def name(self) -> str:
        """Unique environment identifier."""
        ...

    @property
    def device(self) -> torch.device:
        """Target hardware for tensor operations."""
        ...

    @property
    def generator(self) -> Optional[Any]:
        """Associated data generator instance."""
        ...

    def reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[int] = None
    ) -> TensorDict:
        """Initialize or reset environment state."""
        ...

    def step(self, td: TensorDict) -> TensorDict:
        """Transition state based on selected action."""
        ...

    def get_reward(
        self,
        td: TensorDict,
        actions: torch.Tensor
    ) -> TensorDict:
        """Calculate numerical reward for actions."""
        ...

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Return boolean mask of invalid moves."""
        ...
```

### Key Properties

| Property    | Type            | Description                                   |
| ----------- | --------------- | --------------------------------------------- |
| `name`      | `str`           | Unique identifier (e.g., `vrpp`, `wcvrp`)     |
| `device`    | `torch.device`  | Target hardware (`cpu`, `cuda`, `cuda:0`)     |
| `generator` | `Optional[Any]` | Data generator for creating problem instances |

### Core Methods

| Method            | Signature                        | Description                        |
| ----------------- | -------------------------------- | ---------------------------------- |
| `reset`           | `(td, batch_size) -> TensorDict` | Initialize/reset environment state |
| `step`            | `(td) -> TensorDict`             | Apply action and transition state  |
| `get_reward`      | `(td, actions) -> TensorDict`    | Calculate reward for actions       |
| `get_action_mask` | `(td) -> torch.Tensor`           | Return mask of invalid actions     |

### Usage Example

```python
from logic.src.interfaces import IEnv
from logic.src.envs import VRPPEnv

# Environment implements IEnv protocol
env: IEnv = VRPPEnv(num_loc=50)

# Standard workflow
td = env.reset(batch_size=128)  # Initialize
while not td["done"].all():
    # Get valid actions
    mask = env.get_action_mask(td)

    # Select action (policy logic here)
    actions = select_action(td, mask)

    # Transition
    td = env.step(td)

# Calculate final reward
td = env.get_reward(td, actions)
```

---

## 4. Model Interface

**File**: `logic/src/interfaces/model.py`

The `IModel` protocol defines common behavior for all neural components (encoders, decoders, critics). It is marked as `@runtime_checkable` for validation at runtime.

### Protocol Definition

```python
from typing import Protocol, Any, runtime_checkable
import torch
import torch.nn as nn
from tensordict import TensorDict

@runtime_checkable
class IModel(Protocol):
    """Protocol for neural network models."""

    def forward(self, td: TensorDict, **kwargs: Any) -> Any:
        """Core computation pass."""
        ...

    def to(self, device: torch.device) -> "IModel":
        """Move model parameters to device."""
        ...

    def train(self, mode: bool = True) -> "IModel":
        """Set model to training mode."""
        ...

    def eval(self) -> "IModel":
        """Set model to evaluation mode."""
        ...
```

### Core Methods

| Method    | Signature               | Description                              |
| --------- | ----------------------- | ---------------------------------------- |
| `forward` | `(td, **kwargs) -> Any` | Core computation pass                    |
| `to`      | `(device) -> IModel`    | Move parameters to hardware (fluent API) |
| `train`   | `(mode=True) -> IModel` | Switch to training mode                  |
| `eval`    | `() -> IModel`          | Switch to evaluation/inference mode      |

### Usage Example

```python
from logic.src.interfaces import IModel
from logic.src.models import AttentionModel

# Model implements IModel protocol
model: IModel = AttentionModel(
    embed_dim=128,
    num_layers=3,
    num_heads=8
)

# Standard workflow
model = model.to(device)
model = model.train()

# Forward pass
output = model.forward(td, decode_type="greedy")

# Inference mode
model = model.eval()
with torch.no_grad():
    output = model.forward(td)
```

---

## 5. Policy Interface

**File**: `logic/src/interfaces/policy.py`

The `IPolicy` abstract base class defines how a solver outputs a valid route from given context, regardless of internal logic (neural, classical, metaheuristic).

### Abstract Base Class

```python
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

class IPolicy(ABC):
    """Abstract base class for routing policies."""

    @abstractmethod
    def run(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute policy and return tour, cost, and metadata.

        Args:
            **kwargs: Context-specific parameters (distance_matrix,
                     packing_bins, state, etc.)

        Returns:
            Tuple of (tour, cost, metadata):
                - tour: List[int] - Sequence of node IDs
                - cost: float - Evaluation metric
                - metadata: Any - Policy-specific info
        """
        pass
```

### Method Specification

#### `run(**kwargs: Any) -> Tuple[List[int], float, Any]`

The primary execution entry point for all policies.

**Common Inputs** (via `kwargs`):

- `distance_matrix`: `np.ndarray` - Pairwise distances
- `packing_bins`: `List[int]` - Mandatory nodes
- `state`: `Dict` - Current environment state
- `capacity`: `float` - Vehicle capacity
- `depot`: `int` - Depot node index (usually 0)

**Outputs**:

1. `tour`: `List[int]` - Sequence of node IDs (starts/ends with depot)
2. `cost`: `float` - Objective value (distance, penalty, etc.)
3. `metadata`: `Any` - Policy-specific debug info

### Usage Example

```python
from logic.src.interfaces import IPolicy
from logic.src.policies import HGSPolicy

# Policy implements IPolicy interface
policy: IPolicy = HGSPolicy(
    time_limit=60.0,
    population_size=100
)

# Execute policy
tour, cost, metadata = policy.run(
    distance_matrix=dist_matrix,
    packing_bins=[1, 5, 10],
    capacity=100.0,
    depot=0
)

print(f"Tour: {tour}")
print(f"Cost: {cost:.2f}")
print(f"Iterations: {metadata['iterations']}")
```

---

## 6. Policy Adapter Interface

**File**: `logic/src/interfaces/adapter.py`

The `IPolicyAdapter` protocol bridges the gap between neural models and functional policies, enabling unified access to both paradigms.

### Protocol Definition

```python
from typing import Protocol
from logic.src.interfaces import IModel, IPolicy

class IPolicyAdapter(Protocol):
    """Protocol for policy adapters."""

    def get_policy(self, **kwargs) -> IPolicy:
        """Build usable policy instance from model."""
        ...

    def get_model(self) -> IModel:
        """Provide access to underlying neural network."""
        ...
```

### Core Methods

| Method       | Signature               | Description                                  |
| ------------ | ----------------------- | -------------------------------------------- |
| `get_policy` | `(**kwargs) -> IPolicy` | Builds policy instance with given parameters |
| `get_model`  | `() -> IModel`          | Returns the underlying neural model          |

### Usage Example

```python
from logic.src.interfaces import IPolicyAdapter
from logic.src.policies.adapters import NeuralAdapter

# Adapter implements IPolicyAdapter protocol
adapter: IPolicyAdapter = NeuralAdapter(
    model_path="weights/am_best.pt",
    device="cuda"
)

# Get neural model
model = adapter.get_model()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Get executable policy
policy = adapter.get_policy(decode_type="greedy", temperature=1.0)

# Use policy
tour, cost, _ = policy.run(state=environment_state)
```

---

## 7. packing Interface

**File**: `logic/src/interfaces/packing.py`

The `IPackingAlgorithm` protocol standardizes logic for pre-selecting mandatory collection targets before routing begins.

### Protocol Definition

```python
from typing import Protocol, List
from dataclasses import dataclass

@dataclass
class PackingContext:
    """Context for packing."""
    fill_rates: List[float]
    days_since_last_collection: List[int]
    revenue: List[float]
    capacity: float
    threshold: float

class IPackingAlgorithm(Protocol):
    """Protocol for packing algorithms."""

    def pack(self, context: PackingContext) -> List[int]:
        """
        Determine mandatory nodes for collection.

        Args:
            context: Selection context with fill rates, history, etc.

        Returns:
            List of bin IDs that must be visited
        """
        ...
```

### Usage Example

```python
from logic.src.interfaces import IPackingAlgorithm, PackingContext
from logic.src.policies.helpers.packing import LastMinuteSelection

# Selection strategy implements IPackingAlgorithm
selector: IPackingAlgorithm = LastMinuteSelection(threshold=0.9)

# Create context
context = PackingContext(
    fill_rates=[0.5, 0.95, 0.7, 0.88, 0.3],
    days_since_last_collection=[2, 5, 3, 4, 1],
    revenue=[10.0, 15.0, 12.0, 14.0, 8.0],
    capacity=100.0,
    threshold=0.9
)

# Select mandatory bins
packing_bins = selector.pack(context)
# Result: [1, 3] (bins with fill_rate >= 0.9)
```

---

## 8. route improvement Interface

**File**: `logic/src/interfaces/route_improvement.py`

The `Iroute improvement` protocol standardizes route refinement algorithms applied to complete tours.

### Protocol Definition

```python
from typing import Protocol, List, Any

class Iroute improvement(Protocol):
    """Protocol for route improvement refinement."""

    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Refine an existing tour.

        Args:
            tour: Original tour as list of node IDs
            **kwargs: Additional context (distance_matrix, etc.)

        Returns:
            Refined tour (original if no improvement found)
        """
        ...
```

### Common Refinement Methods

| Method  | Description                | Complexity |
| ------- | -------------------------- | ---------- |
| `2-opt` | Edge exchange local search | O(n²)      |
| `3-opt` | Three-edge exchange        | O(n³)      |
| `ILS`   | Iterated local search      | O(k·n²)    |
| `Split` | Route splitting algorithm  | O(n²)      |

### Usage Example

```python
from logic.src.interfaces import Iroute improvement
from logic.src.policies.helpers.route_improvement import TwoOptProcessor

# Post-processor implements Iroute improvement
refiner: Iroute improvement = TwoOptProcessor(max_iterations=100)

# Original tour
original_tour = [0, 5, 3, 8, 2, 7, 1, 4, 6, 0]

# Refine tour
refined_tour = refiner.process(
    tour=original_tour,
    distance_matrix=dist_matrix
)

# Calculate improvement
original_cost = calculate_cost(original_tour, dist_matrix)
refined_cost = calculate_cost(refined_tour, dist_matrix)
improvement = ((original_cost - refined_cost) / original_cost) * 100
print(f"Improvement: {improvement:.2f}%")
```

---

## 9. Infrastructure & Utility Protocols

These protocols provide abstract interfaces for common data structures and utility objects used across the system, ensuring that logic remains decoupled from specific storage implementations (like dictionaries vs. OmegaConf objects).

### 9.1 ITraversable Protocol

**Location**: `logic/src/interfaces/traversable.py`

Standardizes access to configuration-like objects. It allows functions to work seamlessly with `dict`, `DictConfig` (OmegaConf), and `ListConfig` without explicit type checking.

| Method                    | Description                       |
| :------------------------ | :-------------------------------- |
| `__getitem__`             | Access values via `obj[key]`      |
| `get`                     | Safely access values with default |
| `keys`, `items`, `values` | Standard dictionary views         |
| `__contains__`            | Membership check via `key in obj` |

**Usage Pattern**:

```python
def parse_params(cfg: ITraversable):
    # Works for dict OR DictConfig
    threshold = cfg.get("threshold", 0.5)
    return threshold
```

### 9.2 ITensorDictLike Protocol

**Location**: `logic/src/interfaces/tensor_dict_like.py`

Standardizes access to objects storing batches of tensors. Primarily used for RL states, observation dictionaries, and model inputs.

| Method/Property           | Description                      |
| :------------------------ | :------------------------------- |
| `batch_size`              | Dimensions of stored tensors     |
| `device`                  | Current storage device (CPU/GPU) |
| `get` / `set`             | Manage tensor storage            |
| `keys`, `items`, `values` | Dictionary-style iteration       |

**Usage Pattern**:

```python
def move_state(state: ITensorDictLike, device: torch.device):
    # Logic can operate on any tensor-dict-like object
    for k, v in state.items():
        state.set(k, v.to(device))
```

### 9.3 IBinContainer Protocol

**Location**: `logic/src/interfaces/bin_container.py`

Standardizes access to objects managing waste bin states in simulations and environments. It unifies the interface for fill level monitoring and state transitions.

| Property             | Description                                    |
| :------------------- | :--------------------------------------------- |
| `fill_levels`        | Current normalized fill levels (batch, n_bins) |
| `wastes`             | Current bin wastes/capacity usage              |
| `update_fill_levels` | Transition state after collection events       |

**Usage Pattern**:

```python
def check_policy_selection(bins: IBinContainer, mask: torch.Tensor):
    # Agnostic to whether bins is an Environment or a simple DataClass
    overflow_risk = bins.fill_levels > 0.95
    return overflow_risk & mask
```

## 10. Integration Examples

### Example 1: Building a Complete Routing Pipeline

```python
from logic.src.interfaces import (
    IEnv, IPolicy, IPackingAlgorithm, Iroute improvement
)
from logic.src.envs import WCVRPEnv
from logic.src.policies import HGSPolicy
from logic.src.policies.helpers.packing import LastMinuteSelection
from logic.src.policies.helpers.route_improvement import TwoOptProcessor

def run_routing_pipeline(
    env: IEnv,
    policy: IPolicy,
    selector: IPackingAlgorithm,
    refiner: Iroute improvement
):
    """Complete routing pipeline using interfaces."""

    # 1. Initialize environment
    state = env.reset(batch_size=1)

    # 2. Select mandatory bins
    context = create_packing_context(state)
    packing_bins = selector.pack(context)

    # 3. Solve routing problem
    tour, cost, metadata = policy.run(
        state=state,
        packing_bins=packing_bins,
        distance_matrix=state["distance_matrix"]
    )

    # 4. Refine solution
    refined_tour = refiner.process(
        tour=tour,
        distance_matrix=state["distance_matrix"]
    )

    return refined_tour, cost

# Usage
env = WCVRPEnv(num_loc=50)
policy = HGSPolicy(time_limit=60.0)
selector = LastMinuteSelection(threshold=0.9)
refiner = TwoOptProcessor()

tour, cost = run_routing_pipeline(env, policy, selector, refiner)
```

### Example 2: Swapping Implementations

```python
# Can easily swap implementations without changing pipeline code

# Use neural policy instead of HGS
from logic.src.policies.adapters import NeuralAdapter

neural_policy = NeuralAdapter(
    model_path="weights/am_best.pt"
).get_policy(decode_type="greedy")

# Use different selection strategy
from logic.src.policies.helpers.packing import RegularSelection

regular_selector = RegularSelection(frequency=3)

# Run same pipeline with different implementations
tour, cost = run_routing_pipeline(
    env, neural_policy, regular_selector, refiner
)
```

### Example 3: Type-Safe Function Signatures

```python
from typing import List, Tuple
from logic.src.interfaces import IPolicy, IEnv

def evaluate_policy(
    policy: IPolicy,
    env: IEnv,
    num_episodes: int = 100
) -> Tuple[float, float]:
    """Evaluate policy on environment."""

    total_cost = 0.0
    total_time = 0.0

    for _ in range(num_episodes):
        state = env.reset()

        start_time = time.time()
        tour, cost, _ = policy.run(state=state)
        elapsed_time = time.time() - start_time

        total_cost += cost
        total_time += elapsed_time

    avg_cost = total_cost / num_episodes
    avg_time = total_time / num_episodes

    return avg_cost, avg_time

# Type checker ensures policy implements IPolicy
result = evaluate_policy(policy, env)  # ✅ Type-safe
```

---

## 11. Best Practices

### ✅ Good Practices

**Use Interfaces for Function Signatures**

```python
# ✅ GOOD: Type-safe and flexible
def train_agent(env: IEnv, model: IModel, epochs: int):
    for epoch in range(epochs):
        state = env.reset()
        output = model.forward(state)
        # Training logic...

# Can accept any IEnv and IModel implementation
```

**Protocol-Based Design**

```python
# ✅ GOOD: Use Protocol for duck typing
from typing import Protocol

class IOptimizer(Protocol):
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...

# Any class with these methods is compatible
```

**Interface Segregation**

```python
# ✅ GOOD: Focused, minimal interfaces
class IReader(Protocol):
    def read(self) -> str: ...

class IWriter(Protocol):
    def write(self, data: str) -> None: ...

# Use specific interfaces
def process_file(reader: IReader, writer: IWriter):
    data = reader.read()
    writer.write(data)
```

### ❌ Anti-Patterns

**Avoid Concrete Types in Signatures**

```python
# ❌ BAD: Tightly coupled to specific implementation
def train_agent(env: WCVRPEnv, model: AttentionModel):
    # Can only accept these specific types
    pass

# ✅ GOOD: Use interfaces
def train_agent(env: IEnv, model: IModel):
    # Accepts any implementation
    pass
```

**Don't Implement Unnecessary Methods**

```python
# ❌ BAD: Implementing methods that aren't needed
class SimplePolicy(IPolicy):
    def run(self, **kwargs):
        return self.solve(**kwargs)

    def solve(self, **kwargs):  # Unnecessary indirection
        # Actual logic
        pass

# ✅ GOOD: Implement interface directly
class SimplePolicy(IPolicy):
    def run(self, **kwargs):
        # Actual logic
        pass
```

**Avoid Breaking Liskov Substitution Principle**

```python
# ❌ BAD: Subclass changes contract
class BrokenPolicy(IPolicy):
    def run(self, **kwargs):
        # Returns only 2 values instead of 3
        return tour, cost  # Missing metadata!

# ✅ GOOD: Respect the contract
class GoodPolicy(IPolicy):
    def run(self, **kwargs):
        return tour, cost, {}  # Include all return values
```

### Common Pitfalls

**Pitfall 1: Forgetting Runtime Type Checks**

```python
# Problem
def use_model(model):  # No type hint
    model.forward(data)  # May fail at runtime

# Solution
from logic.src.interfaces import IModel

def use_model(model: IModel):  # Type-safe
    if not isinstance(model, IModel):
        raise TypeError("model must implement IModel")
    model.forward(data)
```

**Pitfall 2: Over-Engineering Interfaces**

```python
# ❌ BAD: Too many methods in one interface
class IGodPolicy(Protocol):
    def run(self): ...
    def train(self): ...
    def evaluate(self): ...
    def save(self): ...
    def load(self): ...
    # ... 10 more methods

# ✅ GOOD: Split into focused interfaces
class IPolicy(Protocol):
    def run(self): ...

class ITrainable(Protocol):
    def train(self): ...

class ISerializable(Protocol):
    def save(self): ...
    def load(self): ...
```

---

## 12. Quick Reference

### Common Imports

```python
# Core interfaces
from logic.src.interfaces import IEnv, IModel, IPolicy

# Specialized interfaces
from logic.src.interfaces import (
    IPolicyAdapter,
    IPackingAlgorithm,
    Iroute improvement,
    ITraversable,
    ITensorDictLike,
    IBinContainer
)

# Context classes
from logic.src.interfaces.packing import PackingContext
```

### Interface Summary

| Interface          | Purpose                        | Key Methods                      |
| ------------------ | ------------------------------ | -------------------------------- |
| `IEnv`             | Environment standardization    | `reset`, `step`, `get_reward`    |
| `IModel`           | Neural network standardization | `forward`, `to`, `train`, `eval` |
| `IPolicy`          | Solver standardization         | `run`                            |
| `IPolicyAdapter`   | Bridge neural/classical        | `get_policy`, `get_model`        |
| `IPackingAlgorithm` | Pre-selection strategy         | `pack`                    |
| `Iroute improvement`  | Tour refinement                | `process`                        |
| `ITraversable`     | Configuration standardization  | `get`, `items`, `keys`           |
| `ITensorDictLike`  | Tensor storage standardization | `batch_size`, `device`, `get`    |
| `IBinContainer`    | Bin state standardization      | `fill_levels`, `wastes`          |

### File Locations

| File                  | Lines | Description                 |
| --------------------- | ----- | --------------------------- |
| `env.py`              | ~40   | IEnv protocol definition    |
| `model.py`            | ~30   | IModel protocol definition  |
| `policy.py`           | ~25   | IPolicy abstract base class |
| `adapter.py`          | ~20   | IPolicyAdapter protocol     |
| `packing.py`          | ~35   | IPackingAlgorithm protocol   |
| `route_improvement.py`  | ~20   | Iroute improvement protocol    |
| `traversable.py`      | ~50   | ITraversable protocol       |
| `tensor_dict_like.py` | ~60   | ITensorDictLike protocol    |
| `bin_container.py`    | ~45   | IBinContainer protocol      |

### Related Documentation

- [POLICIES_MODULE.md](POLICIES_MODULE.md) - Policy implementations
- [MODELS_MODULE.md](MODELS_MODULE.md) - Neural model architectures
- [ENVS_MODULE.md](ENVS_MODULE.md) - Environment implementations
- [CLAUDE.md](../CLAUDE.md) - Agent instructions and coding standards

---

**Last Updated**: February 2026
**Maintainer**: WSmart+ Route Development Team
**Status**: ✅ Active - Comprehensive interface documentation
