# Simulation Environments & RL Physics

> **Status (Q-22):** narrative guide. The API reference is generated into
> `docs/api/` by `tools/gen_api_docs.py`; this file is not the source of
> truth for signatures and module inventories.


**Module**: `logic/src/envs`
**Purpose**: Defines the mathematical "physics" of routing problems, governing state transitions, action validity, and reward signals.
**Version**: 3.0
**Last Updated**: February 2026

---

## 1. Overview

The `envs` module implements the core Reinforcement Learning environments for the WSmart-Route system. These classes provide a vectorized, type-safe interface that abstracts away the complexities of batch management and tensor operations, allowing agents to interact with optimization problems as dynamic state machines.

This document serves as a comprehensive technical guide for the following components:

1.  **Base Architecture** (`logic/src/envs/base`)
2.  **Data Generators** (`logic/src/envs/generators`)
3.  **Environment Implementations** (Root `logic/src/envs/*.py`)
4.  **Task Wrappers** (`logic/src/envs/tasks`)

---

## 2. Base Environment Architecture

**Module**: `logic/src/envs/base`

This submodule defines the foundational building blocks for all Reinforcement Learning environments. It provides a robust, vectorized, and type-safe interface that facilitates high-performance batch processing and state management.

### 2.1 Core Component: `RL4COEnvBase`

**File**: `base/base.py`

The `RL4COEnvBase` is the abstract parent class for all constructive environments. It inherits from `torchrl.envs.EnvBase` and integrates specialized logic for combinatorial optimization.

### Inheritance MRO (Method Resolution Order)

```python
class RL4COEnvBase(BatchMixin, OpsMixin, EnvBase): ...
```

1. **`BatchMixin`**: Intercepts `batch_size` attribute access to ensure compatibility with TorchRL's strict typing.
2. **`OpsMixin`**: Provides common mathematical operations and safety wrappers.
3. **`EnvBase`**: The underlying TorchRL primitive.

### Key Methods

#### `__init__(generator, generator_params, device, batch_size, **kwargs)`

Initializes the environment state.

- **`generator`**: `Generator` instance. If `None`, use `generator_params` to instantiate the default generator for this env type.
- **`batch_size`**: `torch.Size` or `int`. Defines the vectorization throughput for parallel simulation.

#### `reset(batch_size=None) -> TensorDict`

Resets the environment to $t=0$.

1. **Data Generation**: Calls `generator.generate(batch_size)` to spawn a new batch of graphs.
2. **Instance Setup**: Invokes `_reset_instance(td)` to add dynamic fields (e.g., `visited` mask, `current_node`).
3. **Output**: Returns the initial observation `TensorDict`.

#### `step(td: TensorDict) -> TensorDict`

Advances the environment by one timestep.

1. **Action Application**: Updates the current node and tour history using the action in `td`.
2. **Mask Computation**: Calls `_get_action_mask(td)` to determine valid next moves.
3. **Termination Check**: Calls `_check_done(td)` to see if the episode ends.
4. **Reward**: Computes reward if done (or dense reward if supported).

#### `_set_seed(seed: Optional[int])`

Sets the random seed for reproducible stochastic transitions.

---

### 2.2 Functional Mixins

### `BatchMixin` (`base/batch.py`)

A critical utility for managing the `batch_size` property, which can be fickle in TorchRL.

- **Purpose**: TorchRL expects `batch_size` to be a `torch.Size` object. This mixin intercepts setters to automatically convert `int` or `list` inputs into `torch.Size`.
- **Fail-Safe Logic**: If strict setting fails (due to TorchRL version mismatches), it falls back to a "soft" set on `self._batch_size` and attempts to resize the specs manually safely.

### `OpsMixin` (`base/ops.py`)

Encapsulates common tensor operations to prevent code duplication across environments.

#### `_make_spec(td: TensorDict)`

Automatically generates `CompositeSpec` (observation specs) and `UnboundedContinuousTensorSpec` (reward specs) based on the shape of the data in the `TensorDict`. This allows environments to be "self-describing" without manually defining every input/output shape.

#### `get_action_mask(td)`

A public wrapper for `_get_action_mask`. It ensures the returned mask is boolean and explicitly on the correct device.

---

### 2.3 Improvement Environments

**File**: [`base/improvement.py`](base/improvement.py)

While `RL4COEnvBase` is for _constructive_ approaches (Node 1 -> Node 2 -> ...), `ImprovementEnvBase` is for _iterative_ approaches description (Initial Solution -> Improved Solution).

### Key Abstraction: `ImprovementEnvBase`

- **Use Case**: Local Search, Simulated Annealing, SISR, k-opt.
- **State**: The state is not a partial tour, but a _complete solution_.
- **Action**: A perturbation operator (e.g., "Swap index $i$ and $j$").
- **Constraint**: Must implement `_get_initial_solution(td)` to provide the starting point for the search.

---

### 2.4 Abstract Methods (The "Contract")

Any concrete environment inheriting from `RL4COEnvBase` **MUST** implement:

1.  **`_reset_instance(td) -> td`**
    - **Responsibility**: Initialize dynamic state variables.
    - **Example**: `td["visited"] = torch.zeros_like(td["locs"])`

2.  **`_get_action_mask(td) -> torch.Tensor`**
    - **Responsibility**: Return a boolean mask where `True` = Invalid / Masked.
    - **Example**: `mask = td["visited"].clone()` (Don't revisit nodes).

3.  **`_get_reward(td, actions) -> torch.Tensor`**
    - **Responsibility**: Compute the objective value (usually negative cost).
    - **Example**: `return -td["tour_length"]`

---

---

## 3. Procedural Data Generators

**Module**: `logic/src/envs/generators`

This submodule implements the **dynamic instance generation** system. Instead of iterating over a fixed dataset, these classes procedurally generate new problem instances on the fly, allowing for infinite training data and robust generalization.

### 3.1 Base Architecture: `Generator`

**File**: [`generators/base.py`](generators/base.py)

The abstract base class for all generators. It standardizes the interface for creating batches of `TensorDict` instances.

### Key Attributes

- **`num_loc`**: Number of customer nodes (excluding depot).
- **`min_loc` / `max_loc`**: Coordinate bounds (typically `0.0` to `1.0`).
- **`loc_distribution`**: Spatial distribution logic (e.g., "uniform", "clustered").
- **`device`**: The target device for generated tensors.

### Core Method: `generate(batch_size)`

Produces a batch of problem instances.

1. Calls internal helpers like `_generate_locations()` and `_generate_depot()`.
2. Samples problem-specific attributes (waste, profits).
3. Returns a `TensorDict` ready for the environment's `reset()`.

---

### 3.2 Generator Implementations

### 1. `VRPPGenerator` (`generators/vrpp.py`)

Generates instances for the **Vehicle Routing Problem with Profits**.

#### Features

- **Profits**: Sampled from a distribution (default: Uniform `[1, 100]`).
- **Time Windows**: (Optional) Can generate `start_time`, `end_time`, `service_time`.
- **Method**:
  ```python
  td = generator.generate(batch_size=128)
  # keys: locs, depot, waste, capacity, time_windows (opt)
  ```

### 2. `WCVRPGenerator` (`generators/wcvrp.py`)

Generates instances for the **Waste Collection VRP**.

#### Features

- **Bin Logic**: Tracks `current_waste` vs `max_waste`.
- **Maps**: Supports generating nodes based on real-world city topologies via the `map_name` parameter (e.g., "Rio Maior").
- **mandatory**: Can flag certain nodes as mandatory vs optional.
- **Config**:
  - `fill_distribution`: How full are the bins? (e.g., Gamma distribution).
  - `depot_type`: "center", "corner", or "random".

### 3. `TSPGenerator` (`generators/tsp.py`)

Generates instances for the **Traveling Salesperson Problem**.

#### Features

- **Simplicity**: Only generates `locs`. No waste or capacities.
- **Output**: `TensorDict(locs=(B, N, 2))` (start node is index 0).

### 4. `SCWCVRPGenerator` (`generators/scwcvrp.py`)

Generates **Stochastic** Waste Collection instances.

#### Features

- **Uncertainty**: Generates two sets of values:
  - **`expected_waste`**: What the agent _thinks_ is there.
  - **`real_waste`**: What is _actually_ there (used for failure simulation).
- **Distribution**: Uses `mu` (mean) and `sigma` (std) to sample log-normal or truncated normal waste levels.

---

### 3.3 Generator Registry

### `GENERATOR_REGISTRY` (`generators/__init__.py`)

A global dictionary mapping string keys to generator classes.

| Key         | Class              | Usage                                                  |
| :---------- | :----------------- | :----------------------------------------------------- |
| `"vrpp"`    | `VRPPGenerator`    | Standard VRPP                                          |
| `"cvrpp"`   | `VRPPGenerator`    | Capacitated VRPP (same generator, different env logic) |
| `"wcvrp"`   | `WCVRPGenerator`   | Waste Collection                                       |
| `"tsp"`     | `TSPGenerator`     | TSP                                                    |
| `"scwcvrp"` | `SCWCVRPGenerator` | Stochastic WCVRP                                       |

### Factory Pattern: `get_generator`

The standard way to instantiate generators.

```python
from logic.src.envs.generators import get_generator

# Create a generator for 50-node WCVRP instances on GPU
gen = get_generator("wcvrp", num_loc=50, device="cuda", fill_distribution="gamma")

# Generate a batch of 128 instances
data = gen.generate(128)
```

---

---

## 4. Environment Implementations

**Module**: `logic/src/envs`

This section details the concrete environment classes found in the root of the `envs` directory. These classes implement the rigorous "physics" of routing problems—defining state spaces, transitions, validity masks, and reward functions.

### 4.1 Vehicle Routing Problem with Profits (VRPP)

**File**: [`vrpp.py`](vrpp.py)
**Class**: `VRPPEnv`

The flagship environment. The agent must select a subset of nodes to visit to maximize total collected profit while minimizing travel cost, subject to a tour length constraint (implicit or explicit).

### State Representation (`reset`)

The `TensorDict` includes:

- **`locs`**: `(B, N, 2)` - Coordinates of all nodes (Index 0 is Depot).
- **`waste`**: `(B, N)` - Waste at each node.
- **`visited`**: `(B, N)` - Binary mask of visited nodes.
- **`current_node`**: `(B, 1)` - Current location of the agent.
- **`tour_length`**: `(B)` - Accumulated distance traveled.

### Transition Logic (`step`)

1. **Move**: Updates `current_node` -> `action`.
2. **Visit**: Sets `visited[action] = 1`.
3. **Collect**: Adds `waste[action]` to `collected_profit`.
4. **Distance**: Adds Euclidean distance from previous node to `tour_length`.

### Validity Mask (`_get_action_mask`)

- **Already Visited**: Nodes with `visited=1` are masked.
- **Depot Rule**:
  - The depot (`0`) is usually masked _unless_ the agent decides to end the tour.
  - In some variants, returning to depot replenishes capacity (for multi-trip).

### Reward Function

$$ R = \sum\_{i \in Tour} P_i - \beta \times \text{Length}(Tour) $$

- **$P_i$**: Profit at node $i$.
- **$\beta$**: Cost weight (configurable).

---

### 4.2 Waste Collection VRP (WCVRP)

**File**: [`wcvrp.py`](wcvrp.py)
**Class**: `WCVRPEnv`

A specialized waste collection VRP where bin waste overflows are part of the cost function.

### Key Features

- **Capacity**: Vehicles have a `vehicle_cap` (e.g., 100.0).
- **Overflow Penalty**: If a node is _not_ visited but has high waste (`> max_waste`), a heavy penalty is applied.
- **mandatory Nodes**: Some nodes may be flagged as mandatory.

### Logic Differences from VRPP

- **Load Tracking**: The state tracks `current_load`.
- **Masking**: Nodes are masked if `current_load + node_waste > capacity` (unless partial collection is allowed, which usually isn't).
- **Episode End**: Often fixed horizon (visit all required) or until capacity full.

### Reward Function

$$ Reward = \text{Collection} \times W*{waste} - \text{Cost} \times W*{dist} - \text{Overflows} \times W\_{overflow} $$

---

### 4.3 Stochastic Environments (SWCVRP / SCWCVRP)

**Files**: [`swcvrp.py`](swcvrp.py), [`scwcvrp.py`](scwcvrp.py)
**Classes**: `SCWCVRPEnv`

These environments introduce uncertainty.

### `SCWCVRPEnv` (Stochastic Capacitated WCVRP)

The agent deals with probabilistic waste values.

- **State**: Includes `expected_waste` and `real_waste`. The agent sees `expected`, but transitions use `real`.
- **Failure**: If the agent attempts to service a node and the truck overflows (`real_waste > capacity`), a "failure" occurs.
- **Recourse**: A mandatory recourse action (typically return to depot) is triggered, incurring penalty.

---

### 4.4 Traveling Salesperson (TSP)

**File**: [`tsp.py`](tsp.py)
**Class**: `TSPEnv`

The classic combinatorial problem.

- **Goal**: Visit **all** nodes exactly once and return to start.
- **Simplify**: No capacity, no profits (all nodes must be visited).
- **Mask**: Mask is simple `~visited`. Node `0` (depot) is masked until `visited.sum() == N`.
- **Reward**: Negative tour length.

### K-Opt Variant (`tsp_kopt.py`)

An environment designed for `Improvement` policies using k-opt local search moves (2-opt, 3-opt) rather than constructive steps.

- **State**: A complete tour `[0, 5, 2, ..., 0]`.
- **Action**: A pair of indices `(i, j)` to perform the swap/reverse.

---

### 4.5 Capacitated Variants (CVRPP / CWCVRP)

**Files**: [`cvrpp.py`](cvrpp.py), [`cwcvrp.py`](cwcvrp.py)

Subclasses that mix constraints.

- **CVRPP**: Profit maximization, but you also have a capacity limit (can't pick up infinite waste).
- **CWCVRP**: Capacitated Waste Collection (redundant naming sometimes, but ensures specific constraints like "No Return to Depot" vs "Multi-Trip" are handled).

---

---

## 5. Task Wrappers

**Module**: `logic/src/envs/tasks`

This submodule defines **static problem definitions** and **cost evaluators**. Unlike the dynamic environment classes, these "Task" classes are largely stateless utility collections used to validate full solutions and compute standardized benchmarks.

### 5.1 Base Architecture: `BaseProblem`

**File**: [`tasks/base.py`](tasks/base.py)

The legacy parent class for all task definitions. It provides static methods for tour validation and length calculation.

### Key Static Methods

#### `validate_tours(pi: torch.Tensor) -> bool`

Checks if a tour `pi` (tensor of node indices) is valid.

- **Constraint**: No duplicate node visits (except depot `0`).
- **Logic**: Sorts the tour and checks for monotonicity/uniqueness.

#### `get_tour_length(dataset, pi, dist_matrix=None) -> Tensor`

Computes the total Euclidean length of a tour.

- **Inputs**:
  - `dataset`: Dictionary containing `locs` or `dist_matrix`.
  - `pi`: Shape `(Batch, Sequence_Length)` containing visited node indices.
- **Logic**: Sums $L2(Node_i, Node_{i+1})$ for the sequence.

---

### 5.2 Task Implementations

### 1. `VRPP` (`tasks/vrpp.py`)

**Vehicle Routing Problem with Profits**

#### Objective Function

Minimizing **Negative Net Profit**:
$$ Cost = -\left(\sum \text{Collected Profit} - \beta \times \text{Tour Length}\right) $$

#### `get_costs(dataset, pi, cw_dict)`

- **`cw_dict`**: Cost Weight dictionary (e.g., `{"length": 0.1}`).
- **Returns**: Tuple `(Total_Cost, Breakdown_Dict, None)`.
  - Breakdown includes `profit` and `length`.

---

### 2. `WCVRP` (`tasks/wcvrp.py`)

**Waste Collection VRP**

#### Objective Function

Minimizing **Weighted Cost**:
$$ Cost = \alpha \times \text{Overflows} + \beta \times \text{Tour Length} - \gamma \times \text{Collected Waste} $$

#### Features

- **Overflow Penalty**: Counts nodes where `waste >= max_waste` that were _not_ visited.
- **Waste Collection**: Sum of waste at visited nodes.

---

### 3. `CVRPP` (`tasks/cvrpp.py`)

**Capacitated VRPP**

Inherits `VRPP` logic but enforces a capacity constraint.

- **Use Case**: When vehicle capacity limits the total waste collection.

---

### 4. `SCWCVRP` (`tasks/scwcvrp.py`)

**Stochastic Dynamic WCVRP**

Defines the evaluation metrics for multi-day scenarios.

- **Long-term Cost**: Sum of costs over $T$ days.
- **Dynamic**: The dataset here includes temporal evolution parameters needed for offline evaluation.

---

## ⚠️ Distinction from `logic.src.envs`

| Component    | `logic/src/envs/*.py`                                  | `logic/src/envs/tasks/*.py`                          |
| :----------- | :----------------------------------------------------- | :--------------------------------------------------- |
| **Purpose**  | **RL Environment** (State machine, transitions, masks) | **Problem Definition** (Cost evaluation, validation) |
| **State**    | Dynamic `TensorDict` (updates every step)              | Static `TensorDict` (dataset) + Complete Tour        |
| **Use Case** | Training Loop (Agent interaction)                      | Evaluation / Baselines / route improvement             |
