# Routing Policies & Global Optimization

**Module**: `logic/src/policies`
**Purpose**: Comprehensive collection of classical, metaheuristic, exact, and neural routing policies for Combinatorial Optimization.
**Files**: 157 Python modules across 14 major components
**Version**: 3.0
**Last Updated**: February 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture & Design Patterns](#architecture--design-patterns)
3. [Module Organization](#module-organization)
4. [Policy Adapters](#policy-adapters)
5. [Classical & Metaheuristic Policies](#classical--metaheuristic-policies)
   - [Rigorous Meta-Heuristic Implementations](#rigorous-meta-heuristic-implementations)
6. [Exact Optimization](#exact-optimization)
7. [Neural Policies](#neural-policies)
8. [Operators Library](#operators-library)
9. [packing algorithms](#packing-algorithms)
10. [route improvement & Refinement](#route improvement--refinement)
11. [Local Search](#local-search)
12. [Integration Examples](#integration-examples)
13. [Best Practices](#best-practices)
14. [Quick Reference](#quick-reference)

---

## 1. Overview

The `policies` module provides a comprehensive technical ecosystem of routing optimization algorithms designed to solve diverse variants of the Vehicle Routing Problem (VRP). It integrates **13 distinct policy implementations**, **26 specialized operators**, **6 packing algorithms**, and **5 route improvement methods** into a unified operational framework.

### Key Features

- **Unified Adapter Pattern**: Ensures a consistent execution interface across heterogeneous routing policies.
- **Modular Framework**: Facilitates the seamless integration of new solvers and neighborhood search operators.
- **Dynamic Policy Registry**: Automated discovery and instantiation of optimization engines by name.
- **Multi-Engine Integration**: Out-of-the-box support for Gurobi, OR-Tools, Hexaly, and PyVRP.
- **Operator Composability**: Mix and match atomic operators to construct complex metaheuristics.
- **Workflow Standardization**: Template-based execution logic that handles common preprocessing and mapping steps.
- **Integrated packing**: Advanced strategies for identifying and prioritizing mandatory collection points.
- **route improvement Pipeline**: Automated route refinement and local search improvement after main solver execution.

### Supported Problems

| Problem     | Description                         | Policies                       |
| ----------- | ----------------------------------- | ------------------------------ |
| **TSP**     | Traveling Salesman Problem          | LKH-3, fast_tsp, exact solvers |
| **CVRP**    | Capacitated Vehicle Routing Problem | LKH-3, HGS, OR-Tools, Gurobi   |
| **VRPP**    | VRP with Profits                    | LKH-3, Gurobi, Hexaly          |
| **WCVRP**   | Waste Collection VRP                | ALNS, HGS, SANS, SISR, Neural  |
| **SCWCVRP** | Stochastic Capacitated WCVRP        | Neural, Adaptive Heuristics    |

### Policy Categories

1. **Exact Methods** (BPC): Branch-and-Price-and-Cut via Gurobi/OR-Tools/VRPy
2. **Metaheuristics**:
   - ALNS (Adaptive Large Neighborhood Search)
   - HGS (Hybrid Genetic Search)
   - HGS-ALNS (Hybrid of HGS and ALNS)
   - ACO (Ant Colony Optimization - K-Sparse and Hyper-Heuristic variants)
   - SANS (Simulated Annealing Neighborhood Search)
   - SISR (Slack Induction by String Removal)
3. **Neural Policies**: Deep RL models (AM, TAM, DDAM)
4. **Classical & Matheuristic Policies**: TSP, CVRP, VRPP solvers (LKH-3 LNS, OR-Tools)
5. **Specialized**: VRPP exact optimizers

---

## 2. Architecture & Design Patterns

The `policies` module utilizes established design patterns to maintain scalability and decouple high-level optimization logic from specific solver implementations.

### Strategic Design Patterns

1.  **Adapter Pattern** (`adapters/`): Provides a unified interface that abstracts the complexities of disparate routing libraries (Gurobi, OR-Tools, etc.).
2.  **Factory Pattern** (`adapters/factory.py`): Centralizes the instantiation logic for all policy adapters.
3.  **Registry Pattern** (`adapters/registry.py`): Facilitates dynamic discovery and decoupled registration of new optimization strategies.
4.  **Template Method Pattern** (`adapters/base_routing_policy.py`): Defines a standardized skeleton for routing execution, ensuring consistent data preparation and result mapping.
5.  **Strategy Pattern** (`other/packing/`): Enables pluggable logic for customer selection and prioritization.
6.  **Command Pattern** (`operators/`): Encapsulates atomic neighborhood search actions to support composable heuristics.

### Core Interface Definitions

The system relies on the `IPolicyAdapter` interface to ensure type safety and consistent interaction.

```python
from logic.src.interfaces.adapter import IPolicyAdapter

class IPolicyAdapter(ABC):
    """Unified interface for all routing optimization policies."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the optimization policy.

        Returns:
            Tuple containing (optimized_tour, total_cost, extra_metadata).
        """
        pass
```

### Standard Execution Flow

The internal workflow for a policy execution follow a strictly defined sequence to maintain data integrity:

1.  **Discovery**: The `PolicyFactory` identifies the requested adapter by key.
2.  **Initialization**: The adapter is instantiated and its specific hyperparameters are loaded.
3.  **Selection**: If configured, a "mandatory" strategy filters the target nodes.
4.  **Transformation**: The global distance matrix is subsetted and normalized for the local solver.
5.  **Optimization**: The core engine (e.g., SISR solver, Gurobi) computes the solution.
6.  **Mapping**: Local solver indices are projected back to global system IDs.
7.  **Refinement**: Optional route improvement steps (e.g., path improvement) are applied.
8.  **Output**: The final tour, accurate cost, and diagnostic metadata are returned.

---

## 3. Module Organization

### Core Directory Structure

```
logic/src/policies/
├── __init__.py                              # Main exports
│
├── adapters/                                # Policy Adapter Pattern (13 adapters)
│   ├── __init__.py
│   ├── base_routing_policy.py              # Template base class
│   ├── factory.py                           # PolicyFactory
│   ├── registry.py                          # RoutingAlgorithmRegistry
│   ├── policy_alns.py                       # ALNS adapter
│   ├── policy_bpc.py                        # BPC adapter
│   ├── policy_cvrp.py                       # CVRP adapter
│   ├── policy_hgs.py                        # HGS adapter
│   ├── policy_hgs_alns.py                   # HGS-ALNS adapter
│   ├── policy_hh_aco.py                     # Hyper-Heuristic ACO adapter
│   ├── policy_ks_aco.py                     # K-Sparse ACO adapter
│   ├── policy_lkh.py                        # Lin-Kernighan-Helsgaun adapter
│   ├── policy_neural.py                     # Neural agent adapter
│   ├── policy_sans.py                       # SANS adapter
│   ├── policy_sisr.py                       # SISR adapter
│   ├── policy_tsp.py                        # TSP adapter
│   └── policy_swc_tcf.py                    # SWC-TCF adapter
│
├── adaptive_large_neighborhood_search/      # ALNS Implementation
│   ├── __init__.py
│   ├── alns.py                              # Custom ALNS implementation
│   ├── alns_package.py                      # Third-party ALNS wrapper
│   ├── ortools_wrapper.py                   # OR-Tools ALNS wrapper
│   └── params.py                            # ALNS parameters
│
├── ant_colony_optimization/                 # ACO Implementations
│   ├── k_sparse_aco/                        # K-Sparse ACO variant
│   │   ├── construction.py                  # Solution construction
│   │   ├── pheromones.py                    # Pheromone management
│   │   ├── solver.py                        # Main solver
│   │   ├── runner.py                        # Execution wrapper
│   │   └── params.py
│   └── hyper_heuristic_aco/                 # Hyper-Heuristic ACO
│       ├── hyper_aco.py                     # Main algorithm
│       ├── hyper_operators.py               # Operator selection
│       ├── runner.py
│       └── params.py
│
├── branch_cut_and_price/                    # Exact Optimization
│   ├── dispatcher.py                        # Engine selection
│   ├── gurobi_engine.py                     # Gurobi solver
│   ├── ortools_engine.py                    # OR-Tools solver
│   └── vrpy_engine.py                       # VRPy solver
│
├── hybrid_genetic_search/                   # HGS Implementation
│   ├── hgs.py                               # Main HGS algorithm
│   ├── pyvrp_wrapper.py                     # PyVRP HGS wrapper
│   ├── evolution.py                         # Evolutionary operators
│   ├── individual.py                        # Individual representation
│   ├── split.py                             # Split algorithm
│   └── params.py
│
├── simulated_annealing_neighborhood_search/ # SANS Implementation
│   ├── heuristics/
│   │   ├── sans.py                          # Main SANS algorithm
│   │   ├── sans_state.py                    # State management
│   │   ├── sans_opt.py                      # Optimization logic
│   │   ├── sans_operators.py                # Operator application
│   │   ├── sans_neighborhoods.py            # Neighborhood structures
│   │   ├── sans_perturbations.py            # Perturbation mechanisms
│   │   └── anneal.py                        # Annealing schedule
│   ├── operators/                           # SANS-specific operators
│   ├── select/                              # Node selection strategies
│   ├── search/                              # Search strategies
│   ├── refinement/                          # Solution refinement
│   └── common/                              # Utilities (routes, distance, etc.)
│
├── slack_induction_by_string_removal/       # SISR Implementation
│   ├── sisr.py                              # Main SISR algorithm
│   ├── solver.py                            # SISR solver
│   └── params.py
│
├── smart_waste_collection_two_commodity_flow/ # SWC-TCF Solvers
│   ├── dispatcher.py                        # SWC-TCF dispatcher
│   ├── gurobi.py                            # Gurobi SWC-TCF solver
│   └── hexaly.py                            # Hexaly SWC-TCF solver
│
├── neural_agent/                            # Neural Policy Wrapper
│   ├── agent.py                             # NeuralAgent class
│   ├── batch.py                             # Batch processing
│   └── simulation.py                        # Simulation interface
│
├── operators/                               # Operator Library (26 operators)
│   ├── move/                                # Move operators
│   │   ├── relocate.py                      # Relocate single node
│   │   └── swap.py                          # Swap two nodes
│   ├── route/                               # Intra-route operators
│   │   ├── two_opt_intra.py                 # 2-opt within route
│   │   ├── two_opt_star.py                  # 2-opt* between routes
│   │   ├── three_opt_intra.py               # 3-opt within route
│   │   └── swap_star.py                     # Swap* between routes
│   ├── destroy/                             # Destroy operators (ALNS)
│   │   ├── random.py                        # Random removal
│   │   ├── worst.py                         # Worst removal
│   │   ├── cluster.py                       # Cluster removal
│   │   ├── shaw.py                          # Shaw removal
│   │   └── string.py                        # String removal (SISR)
│   ├── repair/                              # Repair operators (ALNS)
│   │   ├── greedy.py                        # Greedy insertion
│   │   ├── regret.py                        # Regret-k insertion
│   │   └── greedy_blink.py                  # Greedy with blink
│   ├── perturbation/                        # Perturbation operators
│   │   ├── perturb.py                       # General perturbation
│   │   └── kick.py                          # Kick operator
│   └── exchange/                            # Exchange operators
│       ├── or_opt.py                        # Or-opt
│       ├── cross.py                         # Cross-exchange
│       ├── ejection.py                      # Ejection chain
│       └── lambda_interchange.py            # λ-interchange
│
├── other/                                   # Auxiliary Components
│   ├── packing/                             # packing (6 strategies)
│   │   ├── base/                            # Selection framework
│   │   │   ├── selection_context.py         # Context object
│   │   │   ├── selection_factory.py         # Factory
│   │   │   └── selection_registry.py        # Registry
│   │   ├── selection_last_minute.py         # Threshold-based selection
│   │   ├── selection_regular.py             # Fixed-frequency selection
│   │   ├── selection_lookahead.py           # Predictive selection
│   │   ├── selection_revenue.py             # Profit-based selection
│   │   ├── selection_service_level.py       # Statistical selection
│   │   └── selection_combined.py            # Combined strategies
│   └── route_improvement/                     # route improvement (5 methods)
│       ├── factory.py                       # Factory
│       ├── registry.py                      # Registry
│       ├── fast_tsp.py                      # TSP improvement
│       ├── local_search.py                  # Classical LS
│       ├── random_ls.py                     # Randomized LS
│       ├── ils.py                           # Iterated Local Search
│       └── path.py                          # Path improvement
│
├── local_search/                            # Local Search Base
│   ├── local_search_base.py                 # Base class
│   ├── local_search_hgs.py                  # HGS-specific LS
│   └── local_search_aco.py                  # ACO-specific LS
│
├── cvrp.py                                  # CVRP solver entry
├── tsp.py                                   # TSP solver entry
├── hgs_alns.py                              # HGS-ALNS entry
├── lin_kernighan_helsgaun.py                # LKH-3 entry
├── lin_kernighan_helsgaun_three/            # LKH-3 LNS Matheuristic
│   ├── large_neighborhood_search.py         # LNS orchestrator
│   ├── candidate_set.py                     # Alpha-measure / 1-tree logic
│   ├── subgradient.py                       # Held-Karp pi-penalties
│   ├── tour_improvement.py                  # k-opt (2-5) kernels
│   ├── tour_construction.py                 # Initialization & merging
│   ├── load_tracker.py                      # O(L) penalty updates
│   ├── graph_augmentation.py                # Multi-vehicle dummy depots
│   ├── objective.py                         # Lexicographic (P, C) scoring
│   ├── popmusic.py                          # POPMUSIC subset selection
│   ├── tour_adapter.py                      # Global/Local mapping
│   └── policy_lkh3.py                       # Adapter implementation
```

### Statistics

- **Total Files**: 157 Python modules
- **Policy Adapters**: 13 implementations
- **Operators**: 26 specialized operators
- **packing algorithms**: 6 selection methods
- **Post-Processors**: 5 refinement methods
- **Local Search Variants**: 3 implementations

---

## 4. Policy Adapters

**Directory**: `adapters/`
**Purpose**: Unified interface for all routing policies using Adapter pattern

### 4.1 PolicyFactory

**File**: `adapters/factory.py`

The `PolicyFactory` serves as the centralized entry point for dynamically instantiating policy adapters. It abstracts the underlying engine selection from the high-level routing logic.

```python
from logic.src.policies.adapters import PolicyFactory

# Create policy by name
policy = PolicyFactory.get_adapter("hgs")

# Execute policy
tour, cost, metadata = policy.execute(
    packing=[1, 5, 10, 15],
    bins=bins_state,
    distance_matrix=dist_matrix,
    area="riomaior",
    waste_type="plastic",
    config=config_dict
)
```

**Supported Policy Names**:

- `"alns"`, `"hgs"`, `"hgs_alns"`, `"bpc"`, `"gurobi"`, `"hexaly"`
- `"ks_aco"`, `"hh_aco"`, `"sisr"`, `"sans"`, `"lkh"`
- `"neural"`, `"am"`, `"ddam"`, `"tam"`, `"transgcn"`
- `"tsp"`, `"cvrp"`, `"swc_tcf"`

### 4.2 RoutingAlgorithmRegistry

**File**: `adapters/registry.py`

The `RoutingAlgorithmRegistry` maintains a decentralized map of all available optimization engines, enabling automated discovery and type-safe access through the registration decorator.

```python
from logic.src.policies.adapters import RoutingAlgorithmRegistry

# Register custom policy
@RoutingAlgorithmRegistry.register("my_custom_policy")
class MyCustomPolicy(IPolicyAdapter):
    def execute(self, **kwargs):
        # Implementation
        return tour, cost, metadata

# Get registered policy
policy_cls = RoutingAlgorithmRegistry.get("my_custom_policy")
policy = policy_cls()

# List all policies
all_policies = RoutingAlgorithmRegistry.list_policies()
print(all_policies)
# ['alns', 'hgs', 'bpc', 'neural', 'my_custom_policy', ...]
```

### 4.3 BaseRoutingPolicy

**File**: `adapters/base_routing_policy.py`

The `BaseRoutingPolicy` provides a standardized template method for the routing workflow. It encapsulates common logistical operations (validation, data subsetting, result projection) while delegating the core optimization to subclass-specific solver implementations.

```python
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy

class MyPolicy(BaseRoutingPolicy):
    """Custom policy implementation."""

    def _get_config_key(self) -> str:
        """Return config key for parameter loading."""
        return "my_policy"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        **kwargs
    ) -> Tuple[List[List[int]], float]:
        """Run the solver (subclass-specific)."""
        # 1. Initialize solution
        # 2. Run optimization algorithm
        # 3. Return routes and cost
        routes = [[1, 2, 3], [4, 5]]
        solver_cost = 123.45
        return routes, solver_cost
```

**Template Method Steps**:

1. **Validate** mandatory inputs
2. **Extract** context (bins, distance matrix, config)
3. **Load** area-specific parameters (Q, R, C)
4. **Create** subset problem (distance matrix for mandatory nodes)
5. **Run** solver (subclass-specific implementation)
6. **Map** local indices back to global bin IDs
7. **Compute** total tour cost

---

## 5. Classical & Metaheuristic Policies

### 5.1 ALNS (Adaptive Large Neighborhood Search)

**Location**: `logic/src/policies/adaptive_large_neighborhood_search/`
**Adapters**: `policy_alns.py`

Destroy-and-repair metaheuristic with adaptive operator weights.

#### Algorithm

```text
Algorithm: Adaptive Large Neighborhood Search
1. Generate an initial solution x
2. x_best ← x
3. Initialize weights w_i for all destroy and repair heuristics
4. While stopping criterion is not met:
   a. Select a destroy heuristic d_i and a repair heuristic r_j
      using roulette wheel selection based on weights w_i, w_j
   b. Generate neighborhood solution: x' ← r_j(d_i(x))
   c. If Accept(x', x) criteria met (e.g., Simulated Annealing):
        x ← x'
   d. If f(x) < f(x_best):
        x_best ← x
   e. Update scores for d_i and r_j based on the performance of x'
   f. Periodically update weights w_i and w_j using accumulated scores
5. Return x_best
```

#### Key Features

- **Adaptive Weighting Mechanism**: Maintains a pool of distinct "destroy" (removal) and "repair" (insertion) operators. The algorithm continuously evaluates operator performance and adaptively shifts the selection probability towards heuristics that have historically yielded better solutions.

- **Competing Sub-Heuristics**: Rather than relying on a single neighborhood structure, ALNS explores the search space using multiple, competing strategies (e.g., Shaw removal, worst removal, regret insertion), allowing it to effectively navigate diverse and heavily constrained landscapes.

- **Simulated Annealing Acceptance**: Utilizes a simulated annealing metaheuristic framework at the master level to decide whether to accept worse solutions, enabling the algorithm to escape local optima.

#### Mathematical Formulation

**Operator Selection Probability:**

$$
P(heuristic_i) = \frac{w_i}{\sum_{k=1}^{n} w_k}
$$

where:

- $w_i$ is the weight of heuristic $i$
- $n$ is the total number of heuristics

**Weight Update Rule:**

$$
w_i \leftarrow (1 - r) * w_i + r * \left(\frac{π_i}{θ_i}\right)
$$

where:

- $r$ is the reaction factor
- $π_i$ is the score accumulated during the segment of heuristic $i$
- $θ_i$ is the number of times the heuristic $i$ was used

**Scoring System ($\pi$ increments):**

- $\sigma_1$: New global best solution found.
- $\sigma_2$: Better solution than current found.
- $\sigma_3$: Worse solution accepted.

**Key Parameters:**

- `time_limit`: Maximum runtime in seconds.
- `max_iterations`: Maximum destroy/repair cycles.
- `start_temp`: Initial temperature for SA acceptance.
- `cooling_rate`: The decay rate for the simulated annealing temperature parameter.
- `reaction_factor`(r): Controls how quickly the weight adjustment reacts to recent operator performance.
- `min_removal`: Minimum nodes to destroy.
- `max_removal_pct`: Maximum % of nodes to destroy.

**Complexity:**

- Time: Heavily dependent on the complexity of the selected insertion heuristic. Generally $O(T \times q \times n^2)$ where $T$ is iterations, $q$ is removed nodes, and $n$ is total nodes.
- Space: $O(n)$ to store current and best solutions.

#### Usage Example

```python
from logic.src.policies.adaptive_large_neighborhood_search import ALNSSolver, ALNSParams

params = ALNSParams(
    max_iterations=25000,
    reaction_factor=0.1,
    cooling_rate=0.9995,
    sigma_1=33, sigma_2=9, sigma_3=13,
    min_removal_pct=0.1,
    max_removal_pct=0.4,
    time_limit=120.0
)

solver = ALNSSolver(
    dist_matrix=distance_matrix,
    demands=node_demands,
    capacity=vehicle_capacity,
    params=params
)

best_routes, best_cost = solver.solve()
```

#### References

1. Ropke, S., & Pisinger, D. (2006). "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows." Transportation Science, 40(4), 455-472.

### 5.2 HGS (Hybrid Genetic Search)

**Location**: `logic/src/policies/hybrid_genetic_search/`
**Adapters**: `policy_hgs.py`, `policy_hgs_alns.py`

State-of-the-art genetic algorithm with local search and Split procedure.

#### Algorithm

```text
Algorithm: Hybrid Genetic Search
1. Initialize feasible and infeasible subpopulations (P_feas, P_infeas)
2. While stopping criteria not met:
   a. Select two parent individuals P1, P2 using binary tournament
   b. Generate offspring C via Order Crossover (OX) or similar operator
   c. Educate(C): Apply local search (including SWAP* neighborhood)
   d. If C violates capacity/duration constraints:
        Add C to P_infeas
      Else:
        Add C to P_feas
   e. Update penalty parameters based on proportion of feasible individuals
   f. If |P_feas| > max_size OR |P_infeas| > max_size:
        SurvivorSelection(Subpopulation) → Remove worst individuals based on Biased Fitness
   g. Update global best solution
3. Return best feasible individual
```

#### Key Features

- **Biased Fitness & Diversity Management**: HGS prevents premature convergence by scoring individuals not just on their objective value (cost/distance), but also on their contribution to the population's genetic diversity.
- **Relaxed Constraints (Infeasible Subpopulation)**: Dynamically maintains an infeasible subpopulation, allowing the search to temporarily violate capacity or time constraints to cross barriers in the objective landscape, adjusting penalty weights dynamically based on recent feasibility ratios.
- **Advanced Local Search (SWAP\*)**: Integrates intense local search optimization (education) on every generated offspring. Specifically leverages the SWAP\* neighborhood which evaluates the exchange of nodes between routes without requiring optimal insertion positions a priori.
- **Split Algorithm Evaluation**: Uses a highly efficient Split algorithm with $O(n)$ complexity to evaluate individuals represented as giant tours without trip delimiters.

#### Mathematical Formulation

**Biased Fitness Function:**

$$
biased_fitness(C) = fit(C) + (1 - (\frac{N_{elite}}{|P|})) * dc(C)
$$

where:

- $fit(C)$ is the rank of the individual based on penalized cost
- $dc(C)$ is the diversity contribution rank of the individual (measured via average distance to $n_{close}$ nearest neighbors)
- $N_{elite}$ is the number of elite individuals
- $|P|$ is the population size

**Penalized Cost Function:**

$$
φ(C) = distance(C) + ω_Q * overcapacity(C) + ω_T * overtime(C)
$$

where:

- $distance(C)$ is the total distance of the individual
- $ω_Q$ is the penalty factor for overcapacity
- $overcapacity(C)$ is the total overcapacity of the individual
- $ω_T$ is the penalty factor for overtime
- $overtime(C)$ is the total overtime of the individual

**Key Parameters:**

- `mu (μ)`: Minimum population size.
- `lambda (λ)`: Generation size (number of offspring created before survivor selection).
- `n_elite`: Number of elite individuals strictly protected from survivor deletion.
- `n_close`: Number of nearest neighbors used to calculate diversity contribution.

**Complexity:**

- Time: Dominated by the local search step $O(T \times n^2)$. The SWAP\* operator drastically reduces the constant factor of standard inter-route swaps.
- Space: $O(\mu \times n)$ to maintain the subpopulations.

#### Usage Example

```python
from logic.src.policies.hybrid_genetic_search import HGSSolver, HGSParams

params = HGSParams(
    mu=25,
    lambda_=40,
    n_elite=5,
    n_close=4,
    penalty_update_interval=100,
    max_iterations=10000,
    time_limit=300.0
)

solver = HGSSolver(
    dist_matrix=distance_matrix,
    demands=demands,
    capacity=capacity,
    params=params
)

best_routes, best_cost = solver.solve()
```

#### References

1. Vidal, T. (2022). "Hybrid genetic search for the CVRP: Open-source implementation and SWAP\* neighborhood." Computers & Operations Research, 140, 105643.

### 5.3 ACO (Ant Colony Optimization)

Bio-inspired algorithm with pheromone-based learning, adapted for the WCVRP. The repository splits this into two primary variants: K-Sparse ACO (for efficient routing over large graphs) and Hyper-Heuristic ACO (for dynamic operator selection).

#### 5.3.1 K-Sparse ACO (KS-ACO)

**Location**: `logic/src/policies/ant_colony_optimization_k_sparse/`
**Adapter**: `policy_ks_aco.py`

Efficient ACO variant using a K-nearest neighbors graph to restrict the search space, significantly reducing algorithmic complexity and speeding up convergence on large-scale VRP variants.

##### Algorithm

```text
Algorithm: K-Sparse Ant Colony Optimization (KS-ACO)
1. Initialize pheromone matrix τ(i, j) = τ_0 for all edges (i, j)
2. Compute K-nearest neighbors graph N_i^K for all nodes i
3. While not termination condition:
   a. For each ant k = 1 to n_ants:
      i.   Construct route R_k iteratively:
           - From current node i, select next node j ∈ N_i^K with probability
             p(i,j) ∝ [τ(i,j)]^α * [η(i,j)]^β
           - If N_i^K contains no unvisited nodes, select from all remaining unvisited nodes
      ii.  Calculate fitness f(R_k)
   b. Global Pheromone Update:
      i.   Evaporate: τ(i, j) = (1 - ρ) * τ(i, j)
      ii.  Deposit: Δτ(i, j) on edges of the best ant's route
   c. Update global best route S_best
4. Return S_best
```

##### Key Features

- **K-Sparse Neighborhoods**: Restricts the next-node exploration to the $K$ nearest neighbors, changing the node selection complexity from $\mathcal{O}(n)$ to $\mathcal{O}(K)$.
- **Pheromone-Based Learning**: Uses distributed learning where ants deposit pheromones on high-quality routes.
- **Heuristic Guidance**: Balances learned pheromone values with greedy heuristic information (profitability / distance).

##### Mathematical Formulation

**Transition Probability:**

$$
p_{ij}^k = \frac{[\tau_{ij}]^\alpha [\eta_{ij}]^\beta}{\sum_{l \in N_i^K} [\tau_{il}]^\alpha [\eta_{il}]^\beta}
$$

**Pheromone Update:**

$$
\tau_{ij} \leftarrow (1 - \rho)\tau_{ij} + \sum_{k=1}^{n_{ants}} \Delta\tau_{ij}^k
$$

where $\Delta\tau_{ij}^k$ is inversely proportional to the route cost $f(R_k)$.

**Key Parameters**

- `n_ants`: The number of ants traversing the graph per iteration.
- `alpha` ($\alpha$): Pheromone importance factor.
- `beta` ($\beta$): Heuristic information importance factor.
- `rho` ($\rho$): Evaporation rate.
- `k_sparse` ($K$): Limits allowed moves to the $K$ closest nodes.

**Complexity**

- Time: $\mathcal{O}(I \times M \times n \times K)$, where $I$ is iterations, $M$ is ants, $n$ is nodes, and $K$ is the sparsity factor.
- Space: $\mathcal{O}(n^2)$ for pheromone matrix, though effectively $\mathcal{O}(nK)$ if stored sparsely.

##### Usage Example

```python
ks_aco_policy = PolicyFactory.get_adapter("ks_aco")
tour, cost, _ = ks_aco_policy.execute(
    distance_matrix=dist_matrix,
    bins=bins_state,
    config={
        "ks_aco": {
            "n_ants": 20,
            "alpha": 1.0,
            "beta": 2.0,
            "rho": 0.1,
            "k_sparse": 15
        }
    }
)
```

#### 5.3.2 Hyper-Heuristic ACO (HH-ACO)

**Location** `logic/src/policies/ant_colony_optimization_hyper_heuristic/`
**Adapter**: `policy_hh_aco.py`

ACO variant that integrates an adaptive operator selection mechanism to apply different local search heuristics dynamically to constructed ant paths.

##### Algorithm

```text
Algorithm: Hyper-Heuristic Ant Colony Optimization (HH-ACO)
1. Initialize pheromone matrix τ(i, j) = τ_0
2. Initialize operator weights W(o) = 1.0 for o ∈ {2-opt, relocate, swap, ...}
3. While not termination condition:
   a. For each ant k = 1 to n_ants:
      i.   Construct route R_k using standard ACO probability rules
      ii.  Select a sequence of local search operators S = {o_1, o_2, ..., o_L}
           using roulette wheel selection over weights W(o)
      iii. Apply sequence S to R_k to get improved route R'_k
      iv.  Calculate fitness f(R'_k)
   b. Reward operators in sequence S that produced improvements by increasing their weight W(o)
   c. Global Pheromone Update:
      i.   Evaporate: τ(i, j) = (1 - ρ) * τ(i, j)
      ii.  Deposit: Δτ(i, j) on edges of the best ant's route
   d. Update global best route S_best
4. Return S_best
```

##### Key Features

- **Adaptive Operator Selection**: Learns which local search operators are most effective during the search process and applies them more frequently.
- **Operator Sequences**: Applies a predefined number (`sequence_length`) of operators back-to-back on each constructed ant route.
- **Exploration vs. Exploitation**: Balances the global exploration of the ACO construction phase with the intensive local exploitation of the hyper-heuristic layer.

##### Mathematical Formulation

**Operator Selection Probability:**

$$
p(o) = \frac{W(o)}{\sum_{o' \in \mathcal{O}} W(o')}
$$

**Operator Weight Update:**

$$
W(o) \leftarrow W(o) + \Delta W(o)
$$

where $\Delta W(o)$ is the reward given if application of operator $o$ led to an improvement.

**Key Parameters**

- `n_ants`: Number of ants per iteration.
- `sequence_length` ($L$): The number of heuristic operators applied in sequence to each ant's route.

**Complexity**

- Time: $\mathcal{O}(I \times M \times [n^2 + L \cdot T_{LS}])$, where $T_{LS}$ is the complexity of the applied local search operators (often $\mathcal{O}(n^2)$).
- Space: $\mathcal{O}(n^2)$ for pheromone/heuristic matrices.

##### Usage Example

```python
hh_aco_policy = PolicyFactory.get_adapter("hh_aco")
tour, cost, _ = hh_aco_policy.execute(
    distance_matrix=dist_matrix,
    bins=bins_state,
    config={
        "hh_aco": {
            "n_ants": 20,
            "sequence_length": 3
        }
    }
)
```

### 5.4 SANS (Simulated Annealing Neighborhood Search)

**Location**: `logic/src/policies/simulated_annealing_neighborhood_search/`
**Adapters**: `policy_sans.py`

Comprehensive SA-based search with multiple neighborhoods.

#### Algorithm

```text
Algorithm: Simulated Annealing Neighborhood Search (SANS)
1. Initialize route and bin selection S ← GenerateInitialSolution(packing_bins)
2. S_best ← S
3. T ← T_initial
4. While T > T_final:
   a. For iter = 1 to max_iterations (per temperature level):
      i.   Select a neighborhood operator randomly from active pool (e.g., Swap, Relocate, 2-Opt)
      ii.  Generate candidate solution S' ∈ N(S)
      iii. Δ ← f(S') - f(S)
      iv.  If Δ < 0 (improvement):
             S ← S'
             If f(S') < f(S_best):
                 S_best ← S'
      v.   Else (worse solution):
             Draw random r ∈ (0, 1)
             If r < exp(-Δ / T):
                 S ← S'  (Probabilistic acceptance)
   b. T ← T * α (Geometric cooling schedule)
5. Return S_best
```

#### Key Features

- **Metropolis Acceptance Criterion**: Leverages thermodynamic-inspired probability to accept worse routing configurations early in the search (when the "temperature" is high) to escape local optima. As the system cools, it gradually transforms into a strict, greedy descent algorithm.

- **Profit-Oriented Node Selection**: Specifically tailored for "smart waste collection" or Inventory Routing Problems, SANS actively decides which non-mandatory (optional) nodes are profitable to visit on the current day by balancing the routing cost penalty against the revenue or urgency of the node.

- **Multi-Neighborhood Exploration**: Fuses Simulated Annealing with Variable Neighborhood Search principles. It dynamically applies a suite of inter-route and intra-route operators (e.g., Relocate, Swap, 2-Opt) to thoroughly explore the structural landscape without getting trapped in a single operator's local minimum.

- **Look-Ahead Integration Readiness**: Designed to naturally pair with look-ahead heuristics that flag packing nodes, isolating the combinatorial routing logic from the predictive inventory management logic.

#### Mathematical Formulation

**Acceptabce Probability:**

$$
p_{accept}(S') = \begin{cases}
    1 & \text{if } f(S') < f(S) \\
    \exp\left(-\frac{f(S') - f(S)}{T}\right) & \text{otherwise}
\end{cases}
$$

where:

- $f(S)$ is the objective function value of solution $S$
- $T$ is the current temperature

**Geometric Cooling Schedule:**

$$
T_{k+1} = \alpha \cdot T_k
$$

where:

- $T_{k}$ is the temperature at iteration $k$
- $\alpha$ is the cooling rate

**Key Parameters:**

- `initial_temp` ($T_0$): Starting temperature, typically set high enough to accept a large percentage of worse moves initially.
- `cooling_rate` ($\alpha$): The decay factor applied to the temperature at the end of each epoch (e.g., 0.95 to 0.99).
- `max_iterations`: The number of neighborhood moves evaluated at each temperature plateau.
- `neighborhoods`: Array denoting the specific structural operators allowed to generate $S'$ (e.g., ["swap", "relocate", "2opt"]).

**Complexity:**

- Time: $\mathcal{O}(K \times I \times n^2)$ where $K$ is the number of temperature plateaus (calculated via $\log_{\alpha}(T_{final}/T_{initial})$), $I$ is max_iterations, and $n^2$ bounds the neighborhood generation.
- Space: $\mathcal{O}(n)$ to track the current working solution and the global best solution.

#### Usage Example

```python
from logic.src.policies.adapters import PolicyFactory

# Instantiate the SANS adapter
sans_policy = PolicyFactory.get_adapter("sans")

# Execute policy with geometric cooling and multi-neighborhood search
tour, cost, _ = sans_policy.execute(
    packing=packing_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config={
        "sans": {
            "initial_temp": 1000.0,
            "cooling_schedule": "geometric",
            "cooling_rate": 0.95,
            "max_iterations": 10000,
            "neighborhoods": ["swap", "relocate", "2opt"]
        }
    }
)
```

#### References

1. Jorge, D., Antunes, A. P., Ramos, T. R. P., & Barbosa-Póvoa, A. P. (2022). "A hybrid metaheuristic for smart waste collection problems with workload concerns." Computers & Operations Research, 137, 105518.

### 5.5 RL-HVPL (Reinforcement Learning Hybrid Volleyball Premier League)

**Directory**: `reinforcement_learning_hybrid_volleyball_premier_league/`
**Adapters**: `policy_rl_hvpl.py`

#### Overview

The **Reinforcement Learning Hybrid Volleyball Premier League (RL-HVPL)** is a metaheuristic algorithm that combines:

1. **Enhanced ACO with Q-Learning** for intelligent solution construction
2. **Enhanced ALNS with SARSA** for adaptive destroy/repair operator selection
3. **Population-based VPL framework** for global search and diversity management

This implementation bridges the gap between:

- **HVPL** (basic ACO + ALNS in population framework)
- **RL-AHVPL** (advanced with HGS genetic operators, CMAB crossover selection, GLS, reactive tabu)

RL-HVPL provides the benefits of reinforcement learning-enhanced operators **without** the complexity of genetic evolution.

#### Architecture

##### Core Components

```text
RL-HVPL
├── ACO with Q-Learning (Construction)
│   ├── K-Sparse pheromone matrix
│   ├── Probabilistic ant construction
│   └── Q-Learning for local search operator selection
│
├── ALNS with SARSA (Coaching/Improvement)
│   ├── 9 Destroy operators (random, worst, cluster, shaw, string, unstring I-IV)
│   ├── 9 Repair operators (greedy, regret-k, blink, string I-IV)
│   ├── Perturbation operators (kick, random)
│   └── SARSA for adaptive operator selection
│
└── VPL Framework (Population Management)
    ├── Population (teams) of candidate solutions
    ├── Coaching phase (ALNS-SARSA improvement)
    ├── Competition (best solutions survive)
    ├── Substitution (weak solutions replaced)
    └── Pheromone updates (global guidance)
```

##### Algorithm Flow

```text
1. INITIALIZATION
   ├── Create population using ACO-Q (n_teams solutions)
   └── Track best solution globally

2. MAIN LOOP (max_iterations)
   ├── COACHING PHASE
   │   ├── Elite teams → Intensive ALNS-SARSA (elite_coaching_iterations)
   │   └── Regular teams → Light ALNS-SARSA (regular_coaching_iterations)
   │
   ├── COMPETITION
   │   └── Update global best if improved
   │
   ├── PHEROMONE UPDATE
   │   ├── Evaporate existing pheromones
   │   └── Deposit on best solution edges (profit-based or cost-based)
   │
   └── SUBSTITUTION
       └── Replace weakest teams with new ACO-Q solutions

3. RETURN best solution found
```

#### Key Features

##### 1. Reinforcement Learning Integration

- **Q-Learning in ACO**: Dynamically selects the most effective local search operators during construction
- **SARSA in ALNS**: Adaptively chooses destroy/repair operator pairs based on search state
- **State Representation**: Progress, stagnation, and diversity features discretized into states
- **Adaptive Exploration**: Epsilon-greedy with decay for exploration-exploitation balance

##### 2. Pheromone Update Strategies

**Profit-based (default):**

```python
delta = elitist_weight * profit_weight * profit / cost
```

- Reinforces high profit-to-cost ratio solutions
- Better for profit-maximization problems (VRPP, WCVRP)

**Cost-based (classic ACS):**

```python
delta = elitist_weight / cost
```

- Reinforces low-cost solutions
- Better for pure distance minimization (TSP, CVRP)

##### 3. Adaptive Coaching

Elite teams (top `elite_size`) receive **intensive coaching** with more ALNS-SARSA iterations, while regular teams get **lighter coaching** to balance computational resources.

#### Parameters

##### Core Parameters

| Parameter                     | Default | Description                               |
| ----------------------------- | ------- | ----------------------------------------- |
| `n_teams`                     | 10      | Population size                           |
| `max_iterations`              | 100     | Number of league seasons                  |
| `time_limit`                  | 60.0    | Time budget (seconds)                     |
| `sub_rate`                    | 0.2     | Fraction of teams replaced each iteration |
| `elite_size`                  | 3       | Number of teams receiving elite coaching  |
| `elite_coaching_iterations`   | 300     | ALNS iterations for elite teams           |
| `regular_coaching_iterations` | 100     | ALNS iterations for regular teams         |

##### Pheromone Strategy

| Parameter                   | Default    | Description                     |
| --------------------------- | ---------- | ------------------------------- |
| `pheromone_update_strategy` | `"profit"` | `"profit"` or `"cost"`          |
| `profit_weight`             | 1.0        | Weight for profit-based updates |

##### ACO Parameters

Configured via `ks_aco_params` (KSACOParams):

- `n_ants`: Number of ants per iteration (default: 10)
- `k_sparse`: Sparse pheromone neighbors (default: 10)
- `alpha`: Pheromone influence (default: 1.0)
- `beta`: Heuristic influence (default: 2.0)
- `rho`: Evaporation rate (default: 0.1)
- `q0`: Exploitation probability (default: 0.9)
- `local_search`: Enable local search (default: True)
- `local_search_iterations`: Local search budget (default: 50)

##### ALNS Parameters

Configured via `alns_params` (ALNSParams):

- `max_iterations`: ALNS iterations (default: 200)
- `start_temp`: Initial temperature (default: 100.0)
- `cooling_rate`: Temperature decay (default: 0.97)
- `max_removal_pct`: Max fraction of nodes to remove (default: 0.3)
- `perturb_k`: Perturbation strength (default: 3)

##### RL Parameters

Configured via `rl_config` (RLConfig):

**Q-Learning (ACO):**

- `alpha`: Learning rate (default: 0.1)
- `gamma`: Discount factor (default: 0.9)
- `epsilon`: Exploration rate (default: 0.2)
- `epsilon_decay`: Decay multiplier (default: 0.995)
- `epsilon_min`: Minimum epsilon (default: 0.01)

**SARSA (ALNS):**

- `alpha`: Learning rate (default: 0.1)
- `gamma`: Discount factor (default: 0.9)
- `epsilon`: Exploration rate (default: 0.3)
- `epsilon_decay`: Decay multiplier (default: 0.99)
- `epsilon_min`: Minimum epsilon (default: 0.05)

#### Usage

##### Basic Usage

```python
import numpy as np
from logic.src.policies.reinforcement_learning_hybrid_volleyball_premier_league import (
    RLHVPLSolver,
    RLHVPLParams,
)

# Problem instance
dist_matrix = np.array([[...]])  # Distance matrix (n+1 x n+1, depot at 0)
wastes = {1: 10.0, 2: 15.0, ...}  # Node waste amounts
capacity = 100.0  # Vehicle capacity
R = 10.0  # Revenue per unit waste
C = 1.0   # Cost per unit distance

# Configure solver
params = RLHVPLParams(
    n_teams=10,
    max_iterations=100,
    time_limit=60.0,
)

# Solve
solver = RLHVPLSolver(
    dist_matrix=dist_matrix,
    wastes=wastes,
    capacity=capacity,
    R=R,
    C=C,
    params=params,
    seed=42,
)

routes, profit, cost = solver.solve()

print(f"Profit: {profit:.2f}")
print(f"Cost: {cost:.2f}")
print(f"Routes: {routes}")
```

##### Advanced Configuration

```python
from logic.src.configs.policies.helpers import RLConfig
from logic.src.policies.ant_colony_optimization_k_sparse.params import KSACOParams
from logic.src.policies.adaptive_large_neighborhood_search.params import ALNSParams

# Custom RL configuration
rl_config = RLConfig()
rl_config.td_learning.alpha = 0.15
rl_config.td_learning.gamma = 0.95
rl_config.td_learning.epsilon = 0.25

# Custom ACO parameters
aco_params = KSACOParams(
    n_ants=15,
    k_sparse=15,
    alpha=1.2,
    beta=2.5,
    rho=0.15,
    local_search_iterations=100,
)

# Custom ALNS parameters
alns_params = ALNSParams(
    max_iterations=300,
    start_temp=150.0,
    cooling_rate=0.98,
    max_removal_pct=0.4,
)

# Create solver with custom parameters
params = RLHVPLParams(
    n_teams=15,
    max_iterations=150,
    elite_size=5,
    elite_coaching_iterations=500,
    regular_coaching_iterations=150,
    pheromone_update_strategy="profit",
    profit_weight=1.5,
    rl_config=rl_config,
    aco_params=aco_params,
    alns_params=alns_params,
)

solver = RLHVPLSolver(dist_matrix, wastes, capacity, R, C, params, seed=42)
routes, profit, cost = solver.solve()
```

##### Adapter Usage

```python
from logic.src.policies.adapters import PolicyFactory

rl_hvpl_policy = PolicyFactory.get_adapter("rl_hvpl")
tour, cost, _ = rl_hvpl_policy.execute(
    packing=packing_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config={
        "rl_hvpl": {
            "n_teams": 10,
            "max_iterations": 100,
            "time_limit": 60.0,
            "elite_size": 3,
            "pheromone_update_strategy": "profit"
        }
    }
)
```

#### Performance Characteristics

##### Computational Complexity

- **Per iteration**: O(n_teams × (ACO_cost + ALNS_cost))
- **ACO_cost**: O(n_ants × n² × k_sparse)
- **ALNS_cost**: O(max_iterations × n²)

##### Memory Usage

- Pheromone matrix: O(n × k_sparse)
- Population: O(n_teams × n)
- RL Q-tables: O(n_states × n_actions)

##### Scalability

| Problem Size          | Recommended Settings           |
| --------------------- | ------------------------------ |
| Small (n < 50)        | n_teams=5, max_iterations=50   |
| Medium (50 ≤ n < 100) | n_teams=10, max_iterations=100 |
| Large (100 ≤ n < 200) | n_teams=15, max_iterations=150 |
| Very Large (n ≥ 200)  | n_teams=20, max_iterations=200 |

#### References

1. **Volleyball Premier League Algorithm**
   - Population-based metaheuristic framework

2. **Ant Colony Optimization**
   - Dorigo & Gambardella, "Ant Colony System", IEEE Trans., 1997

3. **Adaptive Large Neighborhood Search**
   - Pisinger & Ropke, "A general heuristic for VRP", Computers & OR, 2007

4. **Q-Learning**
   - Watkins & Dayan, "Q-Learning", Machine Learning, 1992

5. **SARSA**
   - Sutton & Barto, "Reinforcement Learning: An Introduction", 2018

#### File Structure

```text
reinforcement_learning_hybrid_volleyball_premier_league/
├── __init__.py          # Module exports
├── params.py            # RLHVPLParams configuration
└── rl_hvpl.py           # RLHVPLSolver main implementation
```

### 5.6 FILO (Fast Iterative Localized Optimization)

**Location**: `logic/src/policies/fast_iterative_localized_optimization/`
**Adapters**: `policy_filo.py`

Fast Iterative Localized Optimization (FILO) is a scalable metaheuristic built specifically to solve large Capacitated Vehicle Routing Problems (CVRP). It introduces dynamic parameters to selectively evaluate the neighborhood space.

#### Algorithm

```text
Algorithm: Fast Iterative Localized Optimization
1. Generate initial solution S
2. S_best ← S
3. While max iterations or time limit not reached:
   a. Select a seed node or seed route randomly
   b. Identify a localized sub-problem (subset of spatially close routes)
   c. Extract the localized sub-problem from S to form S_local
   d. S'_local ← RuinAndRecreate(S_local) OR LocalSearch(S_local)
   e. Reintegrate S'_local into S to form S'
   f. If f(S') < f(S):
        S ← S'
   g. If f(S) < f(S_best):
        S_best ← S
   h. Periodically apply global diversification
4. Return S_best
```

#### Key Features

- **Extreme Scalability (Localization)**: FILO is specifically designed for very large-scale instances (up to tens of thousands of nodes). By extracting and optimizing only a small, spatially localized subset of routes at any given time, the algorithm operates with near-linear asymptotic complexity $O(n \log n)$ relative to the total problem size.
- **Granular Neighborhoods**: Utilizes granular tabu search principles to restrict evaluated moves to a candidate list of promising edges, effectively filtering out structurally poor moves and dramatically accelerating neighborhood evaluations.
- **Localized Ruin and Recreate**: Applies intense perturbation (ruin and recreate) strictly within the extracted localized sub-problem, preserving the global structure of the solution while deeply optimizing regional routing inefficiencies.

#### Mathematical Formulation

**Granular Distance Metric:**

$$
d'_{ij} = d_{ij} - \alpha * (\mu_i + \mu_j)
$$

Where:

- $\alpha$: sparsity parameter
- $\mu$: node-specific potentials (e.g., waiting times or penalties), used to build the restricted candidate list (RCL).

**Key Parameters**

- `localization_radius`: The maximum geographic or topological distance to include routes in the local sub-problem.
- `max_routes_in_subproblem`: Hard cap on the number of routes extracted per iteration to ensure $O(1)$ scaling behavior relative to $n$.
- `ruin_fraction`: The percentage of nodes within the sub-problem to remove during the localized ruin phase.

**Complexity:**

- Time: $O(T \times k^2)$ where $k$ is the constant size of the localized sub-problem (independent of global size $n$). Global iteration complexity scales at $O(n \log n)$.
- Space: $O(n)$ for solution representation and granular neighbor lists.

#### Usage Example

```python
from logic.src.policies.fast_iterative_localized_optimization import FILOSolver, FILOParams

params = FILOParams(
    max_routes_in_subproblem=10,
    ruin_fraction=0.3,
    alpha_sparsity=0.5,
    max_iterations=50000,
    time_limit=600.0
)

solver = FILOSolver(
    dist_matrix=large_distance_matrix,
    demands=demands,
    capacity=capacity,
    params=params
)

best_routes, best_cost = solver.solve()
```

#### References

1. Accorsi, L., & Vigo, D. (2021). "A fast and scalable heuristic for the solution of large-scale capacitated vehicle routing problems." Transportation Science, 55(4), 832-856.

### 5.7 ILS-RVND-SP (Iterated Local Search - Randomized Variable Neighborhood Descent - Set Partitioning)

**Directory**: `iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning/`
**Adapters**: `policy_ils_rvns_sp.py`

Iterated Local Search - Randomized Variable Neighborhood Descent - Set Partitioning (ILS-RVND-SP) combines the Iterated Local Search (ILS) and Randomized Variable Neighborhood Descent (RVND) metaheuristics with the Set Partitioning (SP) exact method to solve the Vehicle Routing Problem (VRP).

#### Algorithm

```text
Algorithm: ILS-RVND-SP
1. Initialize route pool P ← ∅
2. S_best ← ∞
3. For restart = 1 to Max_Restarts:
   a. S ← GenerateInitialSolution()
   b. iter_ILS ← 0
   c. While iter_ILS < Max_Iter_ILS:
      i.   S' ← Perturbation(S) (e.g., multiple node shifts/swaps)
      ii.  S'' ← RVND(S')
           # RVND Loop:
           # 1. N_list ← Randomly shuffle neighborhood operators (e.g., 2-opt, Swap, Relocate)
           # 2. While N_list is not empty:
           #    a. Apply operator N_i from N_list to S''
           #    b. If local optimum improved:
           #         Update S'' and completely reshuffle N_list
           #    c. Else:
           #         Remove N_i from N_list
      iii. Add all unique valid routes from S'' to pool P
      iv.  If f(S'') < f(S):
             S ← S''
             iter_ILS ← 0
           Else:
             iter_ILS ← iter_ILS + 1
      v.   If f(S'') < f(S_best):
             S_best ← S''
4. Set Partitioning Phase:
   a. S_MIP ← Solve Set Partitioning MIP model over route pool P using exact solver
5. Return better of S_best and S_MIP
```

#### Key Features

- **Randomized Variable Neighborhood Descent (RVND)**: Instead of applying local search operators in a fixed deterministic order (as in standard VND), RVND dynamically shuffles the list of inter- and intra-route operators (e.g., Relocate, Swap, 2-Opt, Cross). Whenever an improvement is found, the sequence is re-randomized, preventing the search from falling into static cyclic traps and exploring a richer variety of local minima.

- **Multi-Start Iterated Local Search**: Employs an aggressive perturbation phase combined with multiple restarts to deeply explore the solution space. The ILS phase effectively acts as a high-quality "mass-route generator" for the exact phase rather than just a standalone solver.

- **Exact-Heuristic Hybridity (Matheuristic)**: Extracts all unique, actively explored route structures discovered during the metaheuristic phase into a global route pool. The final solution is resolved purely mathematically through a Set Partitioning (SP) model, guaranteeing the optimal combination of the discovered sub-spaces using commercial MIP solvers (e.g., Gurobi).

#### Mathematical Formulation

Set Partitioning (SP) Model: The exact phase solves the following Mixed Integer Programming (MIP) model over the generated pool of routes $P$:

$$
\begin{align}
\text{minimize } & \sum_{r \in P} c_r \lambda_r \\
\text{subject to } & \sum_{r \in P} a_{ir} \lambda_r = 1  \quad \forall i \in V \\
& \lambda_r \in \{0, 1\} \quad \forall r \in P
\end{align}
$$

Where:

- $P$ is the set of all unique valid routes found during the ILS-RVND phase.
- $c_j$ is the cost (e.g., total distance) of route $j$.
- $a_{ij}$ is a binary parameter equal to $1$ if route $j$ visits customer $i$, and $0$ otherwise.
- $\lambda_j$ is the binary decision variable indicating whether route $j$ is selected for the final solution.

**Key Parameters:**

- max_restarts: Number of independent multi-start executions.
- max_iter_ils: Consecutive non-improving ILS iterations before terminating the current restart.
- perturbation_strength: Number of elements modified during the perturbation phase.
- mip_time_limit: Maximum wall-clock time allocated to the exact MIP solver during the SP phase.

**Complexity:**

- Time (Metaheuristic Phase): $\mathcal{O}(R \times I \times n^2)$ where $R$ is restarts, $I$ is ILS iterations, and $n^2$ bounds the RVND local search descents.
- Time (Exact Phase): $\mathcal{O}(2^{|P|})$ worst-case, bounded by the mip_time_limit and solver pruning efficiency.
- Space: $\mathcal{O}(|P| \times n)$ to maintain the sparse matrix of the global route pool.

#### Usage Example

```python
from logic.src.policies.ils_rvnd_sp import ILSRVNDSPSolver, ILSRVNDSPParams

# Configure parameters based on the paper's multi-start setup
params = ILSRVNDSPParams(
    max_restarts=10,
    max_iter_ils=50,
    perturbation_strength=2,
    use_set_partitioning=True,
    mip_time_limit=120.0,
    time_limit=300.0
)

# Initialize the matheuristic solver
solver = ILSRVNDSPSolver(
    dist_matrix=distance_matrix,
    demands=node_demands,
    capacity=vehicle_capacity,
    params=params,
    seed=42
)

# Solve
best_routes, best_cost = solver.solve()
print(f"Best exact-resolved cost: {best_cost:.2f}")
```

#### References

1. Subramanian et al. "A hybrid algorithm for a class of vehicle routing problems", _Computers & Operations Research_, 2013.

### 5.8 KGLS (Knowledge-Guided Local Search)

**Location**: `logic/src/policies/knowledge_guided_local_search/`
**Adapters**: `policy_kgls.py`

Knowledge-Guided Local Search (KGLS) utilizes problem-specific domain logic—particularly spatial and geometric routing constraints—to perturb solutions dynamically, rather than purely mathematically or randomly tearing sub-graphs.

#### Algorithm

```text
Algorithm: Knowledge-Guided Local Search
1. Generate initial solution S
2. S ← BaselineLocalSearchDescent(S)
3. S_best ← S
4. While stopping criteria not met:
   a. Compute geometric route features (e.g., Route Width, Route Length)
   b. Identify sub-optimal edge connections (haphazard weaving, long crossings)
   c. Inflate the cost of identified bad edges in the distance matrix:
        d'_{ij} ← d_{ij} + penalty(i, j)
   d. Fast Local Search Descent starting from penalized targets using d'
   e. Remove penalties (d' ← d)
   f. Fast Local Search Descent using pure baseline distance matrix
   g. If f(S) < f(S_best):
        S_best ← S
   h. Rotate penalization criteria (Width → Length → Width/Length)
5. Return S_best
```

#### Key Features

- **Problem-Specific Geometric Perturbation**: Departs from standard randomized mathematical perturbation. KGLS actively analyzes route geometries—identifying routes that are "too wide" or edges that are abnormally long compared to the route average—and actively targets them for destruction.
- **Dynamic Cost Matrix Inflation**: Perturbation is achieved implicitly rather than explicitly. Instead of manually tearing apart a route, KGLS heavily penalizes the cost of the identified "bad" edges in the internal distance matrix. Subsequent local search operations automatically break these edges by naturally following the newly altered gradient.
- **Systematic Penalty Cycles**: Avoids getting trapped in repetitive evaluation cycles by systematically rotating the geometric criteria used to generate penalties (e.g., width-based penalization, followed by length-based penalization).

#### Mathematical Formulation

**Cost Inflation (Penalized Matrix):**

$$
c'_{ij} = c_{ij} + (p_{ij} × base_penalty_factor)
$$

where:

- $c_{ij}$ is the original distance between nodes $i$ and $j$.
- $p_{ij}$ is the accumulated penalty count for edge $(i, j)$ incremented whenever the edge is flagged by the geometric evaluation functions.
- $base_penalty_factor$ is a constant factor that scales the penalty.

**Key Parameters:**

- `num_perturbations`: Number of times the cost-inflation and subsequent local search cycle is triggered per iteration block.
- `neighborhood_size`: The maximum number of nearest neighbors evaluated during the Fast Local Search descent phase.
- `penalization_cycle`: Array denoting the order of geometric criteria to apply (e.g., ["width", "length", "width_length"]).

**Complexity:**

- Time: Dominated by the Local Search descent $O(T \times n^2)$.
- Space: $O(n^2)$ for maintaining the base distance matrix and the dynamic penalty matrices.

#### Usage Example

```python
from logic.src.policies.knowledge_guided_local_search import KGLSSolver, KGLSParams

params = KGLSParams(
    num_perturbations=3,
    neighborhood_size=20,
    penalization_cycle=["width", "length", "width_length"],
    time_limit=60.0
)

solver = KGLSSolver(
    dist_matrix=distance_matrix,
    demands=demands,
    capacity=capacity,
    node_coordinates=coordinates, # Required for geometric analysis
    params=params
)

best_routes, best_cost = solver.solve()
```

#### References

1. Arnold, F., & Sörensen, K. (2019). "Knowledge-guided local search for the vehicle routing problem." Computers & Operations Research, 105, 32-46.

---

### 5.9 (μ,λ) Evolution Strategy

**Location:** `logic/src/policies/evolution_strategy_mu_comma_lambda/`
**Adapters:** `policy_es_mcl.py`

A generational (μ,λ) Evolution Strategy that enforces a memoryless state transition. It maintains a parent population of size μ and generates an offspring population of size λ at each iteration.

#### Algorithm

```text
Algorithm: (μ,λ) Evolution Strategy
1. Initialize population of μ parent solutions using nearest-neighbor heuristic
2. For each generation:
   a. Variation (Offspring Generation):
      For 1 to λ:
        i.   Select two parents uniformly at random
        ii.  Recombination: Extract a random subset of nodes from Parent 2 and remove them from Parent 1
        iii. Mutation: Destroy a fraction of the remaining nodes (n_removal)
        iv.  Repair: Greedily reinsert destroyed and extracted nodes
        v.   Education: Optimize repaired solution using ACO Local Search
   b. Evaluation: Calculate net profit for all λ offspring
   c. Selection (Generational): Sort the λ offspring descending by fitness
   d. Select the top μ offspring to become the new parents. The previous μ parents are entirely discarded.
3. Return the global best solution
```

#### Key Features

- **Strict Generational Transition**: Unlike (μ+λ)-ES, the selection operator perfectly enforces the Markov property. The previous parents are discarded completely rather than competing against offspring, helping the algorithm escape local optima.
- **Truncation Selection**: Deterministically selects the top μ individuals purely from the λ generated offspring.
- **Robust Neighborhood Exploration**: Uses discrete variable neighborhood structures (random removal, greedy insertion) integrated tightly with robust ACO Local search.

#### Mathematical Formulation

**Selection Operator:**

$$
P_{t+1} = \text{Top}_{\mu}(O_t)
$$

where:

- $O_t$ is the offspring pool of size $\lambda$ generated at generation $t$.
- $\text{Top}_{\mu}$ selects the $\mu$ best individuals based on fitness.

**Key Parameters:**

- `mu` (μ): Number of parent individuals.
- `lambda_` (λ): Number of offspring generated per generation (must be $\ge \mu$).
- `n_removal`: Mutation strength for discrete destruction operators.
- `local_search_iterations`: Number of inter-route and intra-route improvements applied to each newly repaired offspring.

**Complexity:**

- Time: $O(T \times \lambda \times n^2)$ where $T$ is the number of iterations and $n^2$ bounds the continuous local search phase.
- Space: $O((\mu + \lambda) \times n)$ to track pop and offspring buffers.

#### Usage Example

```python
from logic.src.policies.evolution_strategy_mu_comma_lambda import MuCommaLambdaESSolver, MuCommaLambdaESParams

params = MuCommaLambdaESParams(
    mu=10,                      # Number of parents
    lambda_=40,                 # Number of offspring (λ > μ)
    n_removal=3,                # Discrete mutation strength
    local_search_iterations=20, # Optimization power on offspring
    max_iterations=100
)

solver = MuCommaLambdaESSolver(
    dist_matrix=distance_matrix,
    wastes=waste_dict,
    capacity=100.0,
    R=1.0,
    C=1.0,
    params=params,
    seed=42
)

best_routes, best_profit, best_cost = solver.solve()
```

---

### 5.10 (μ,κ,λ) Evolution Strategy

**Location:** `logic/src/policies/evolution_strategy_mu_kappa_lambda/`
**Adapters**: `policy_es_mkl.py`

#### Algorithm

```text
Algorithm: (μ,κ,λ)-Evolution Strategy
1. Initialize population P₀ of size μ:
   For each individual:
       x ← random vector from search space
       σ ← initial_sigma
       age ← 1
2. Evaluate fitness for all individuals in P₀
3. While generation t < max_iterations:
   a. Offspring Generation (Offspring Pool O = ∅):
      For 1 to λ:
          i.   Select ρ parents from Pₜ via independent uniform sampling with replacement
          ii.  Recombine selected parents to generate offspring (x_child, σ_child)
          iii. Mutate strategy parameters (σ') using log-normal self-adaptation
          iv.  Mutate object variables: x' ← x_child + σ' · N(0,1)
          v.   Evaluate fitness f(x')
          vi.  Set age of offspring = 1
          vii. Add individual to O
   b. Age Filtering (Eligible Parent Pool P_eligible):
      P_eligible ← { p ∈ Pₜ | age(p) ≤ κ }
   c. Selection Pool (T):
      T ← P_eligible ∪ O
   d. Survival Selection:
      Pₜ₊₁ ← Select the top μ individuals from T based on fitness (truncation)
   e. Aging:
      For each surviving individual p ∈ Pₜ₊₁:
          If p was in P_eligible:
              age(p) ← age(p) + 1
   f. Update global best solution found so far
   g. t ← t + 1
4. Return best individual
```

#### Key Features

- **Age-Based Truncation (κ)**: Implements a systematic survivor constraint that bridges the gap between non-elitist $(\mu,\lambda)$ and elitist $(\mu+\lambda)$ strategies. By limiting parental lifespan to $\kappa$ generations, the algorithm ensures population turnover and prevents the search from being trapped by aging "super-individuals" in multi-modal landscapes.
- **Self-Adaptive Mutation Control**: Treats the search distribution (step-size $\sigma$) as part of the evolved genetic material. By adapting mutation strength via log-normal updates, the algorithm automatically discovers the local topology of the fitness landscape to maintain optimal progress rates without manual parameter tuning.
- **Independent Parent Sampling**: Recombination participants are selected via uniform sampling with replacement. This allows individuals with high relative fitness to influence multiple offspring within a single cycle, enhancing the "genetic repair" effect and amplifying beneficial traits.
- **Markovian State Transition**: The generational transition is modeled as a memoryless state update governed by the age counter. This functional structure facilitates clear data ownership and enables parallelized evaluation of the $\lambda$ offspring pool.

#### Mathematical Formulation

**Step-size mutation:**

```text
N_global ~ N(0,1)  (shared across all dimensions)
σ'ᵢ ← σᵢ · exp(τ_global · N_global + τ_local · Nᵢ(0,1))
```

**Decision variable mutation:**

```text
x'ᵢ ← xᵢ + σ'ᵢ · N(0,1)
```

**Learning rates:**

```text
τ_local  = 1/√(2d)
τ_global = 1/(2√d)
```

**Key Parameters:**

- `mu` (μ): Number of parent individuals.
- `kappa` (κ): Maximum age for parents before they are discarded (controls population turnover).
- `lambda_` (λ): Number of offspring generated per generation.
- `rho` (ρ): Number of parents involved in recombination (1 for discrete, μ for intermediate).
- `initial_sigma`: Initial step size for mutation (recommended ~5% of search domain).

**Terminology Mapping:**

- "Eligible Parents" → Stay in the selection pool if age ≤ κ
- "Age" → Counter for how many generations an individual has survived
- "Self-Adaptation" → Step sizes (σ) evolve alongside decision variables (x)

**Complexity:**

- Time: O(T × λ × d) where T = iterations, λ = offspring, d = dimensions
- Space: O((μ + λ) × d) for population + offspring

#### Usage Example

```python
import numpy as np
from logic.src.policies.evolution_strategy_mu_kappa_lambda import (
    MuKappaLambdaESSolver,
    MuKappaLambdaESParams
)

# Define objective function
def sphere(x):
    return np.sum(x**2)

# Configure parameters
params = MuKappaLambdaESParams(
    mu=15,              # Number of parents
    kappa=7,            # Maximum age
    lambda_=100,        # Number of offspring
    rho=2,              # Recombination size
    initial_sigma=1.0,  # Initial step size
    max_iterations=100,
    bounds_min=-5.0,
    bounds_max=5.0,
)

# Create solver
solver = MuKappaLambdaESSolver(
    objective_function=sphere,
    dimension=10,
    params=params,
    seed=42,
    minimize=True,
)

# Solve
best_x, best_fitness = solver.solve()
print(f"Best fitness: {best_fitness:.6e}")
```

#### References

1. Emmerich, M., Shir, O. M., & Wang, H. (2015). "Evolution Strategies." In: Handbook of Natural Computing, Springer.

### 5.11 Rigorous Meta-Heuristic Implementations

This section describes the mathematically rigorous implementations that replace metaphor-based algorithms in the WSmart+ Route codebase.

#### Overview

The "metaphor controversy" in optimization research refers to algorithms that obscure standard mathematical operations with biological, physical, or social metaphors. We have replaced these with canonical implementations using proper Operations Research terminology.

---

#### Implementation Map

| Metaphor-Based Algorithm                | Rigorous Implementation                       | Mathematical Foundation                                                                                           |
| --------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Harmony Search (HS)                     | **(μ+λ) Evolution Strategy [with λ=1]**       | Population-based search with recombination and mutation                                                           |
| Firefly Algorithm (FA)                  | **PSO Distance-Based Algorithm (PSODA)**      | Particle swarm with distance-based update                                                                         |
| Artificial Bee Colony (ABC)             | **Differential Evolution (DE)**               | Isomorphic to DE mutation and proportional selection                                                              |
| Soccer League Competition (SLC)         | **Memetic Algorithm Island Model (MA-IM)**    | Hierarchical Island GA with intensive intra-island local search.                                                  |
| Hybrid Volleyball Premier League (HVPL) | **Hybrid Memetic Search (HMS)**               | 3-Phase hybrid pipeline combining ACO, GA, and ALNS.                                                              |
| Volleyball Premier League (VPL)         | **Memetic Algorithm Dual Population (MA-DP)** | Multi-island GA with dual population (active + reserve).                                                          |
| League Championship Algorithm (LCA)     | **Memetic Algorithm Tolerance Based (MA-TB)** | Pairwise tournament selection with infeasibility tolerance.                                                       |
| Sine Cosine Algorithm (SCA)             | **Particle Swarm Optimization (PSO)**         | PSO that uses the sine and cosine functions to control the search, instead of a uniform/Gaussian random variable. |

---

#### Detailed Implementations

##### 1. (μ+λ) Evolution Strategy

**Replaces:** Harmony Search (HS)

**Location:** `logic/src/policies/evolution_strategy_mu_plus_lambda/`

**Algorithm:**

```text
Algorithm: (μ+λ) Evolution Strategy
1. Initialize population of μ parent solutions using nearest-neighbor heuristic
2. For each generation:
   a. Variation (Offspring Generation):
      For 1 to λ:
        i.   Select two parents uniformly at random
        ii.  Recombination: Extract a random subset of nodes from Parent 2 and remove them from Parent 1
        iii. Mutation: Destroy a fraction of the remaining nodes (n_removal)
        iv.  Repair: Greedily reinsert destroyed and extracted nodes
        v.   Education: Optimize repaired solution using ACO Local Search
   b. Evaluation: Calculate net profit for all λ offspring
   c. Selection (Elitist): Combine the μ parents and λ offspring into a single pool (size μ+λ)
   d. Sort combined pool descending by fitness
   e. Select the top μ individuals to survive into the next generation
3. Return the global best solution
```

**Key Parameters:**

- `population_size` (μ): Number of parent solutions.
- `offspring_size` (λ): Number of offspring generated per iteration.
- `n_removal`: Mutation strength for discrete destruction operators.
- `local_search_iterations`: Iterations for ACO local search during mutation/repair.

**Why it replaces Harmony Search (HS):**

Harmony search relies on a "Harmony Memory" which is mathematically equivalent to a population archive, and "improvisations" which map perfectly to offspring generations.

**Terminology Mapping (HS → (μ+λ) ES):**

- "Harmony Memory" → Population/Archive
- "Improvisation" → Offspring generation
- "HMCR (Memory Consideration Rate)" → Recombination rate
- "Pitch Adjustment" → Mutation operator (Destroy/Repair + Local Search)

**Complexity:**

- Time: O(T × λ × (n + n²)) where T = iterations and n² bounds the local search
- Space: O((μ + λ) × n) to store the populations

**Reference:**

> Rechenberg, I. (1973). "Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution."

---

##### 2. Distance-Based Particle Swarm Optimization

**Replaces:** Firefly Algorithm (FA)

**Location:** `logic/src/policies/particle_swarm_optimization_distance_based_algorithm/`

**Algorithm:**

```
TRUE PSO with distance-based attraction and velocity momentum:

1. Initialize swarm of particles (solutions) with velocities
2. For each iteration:
   a. For each particle i:
      - Update velocity using PSO equation with distance-based attraction:
        v_i(t+1) = w(t) × v_i(t) + c₁ × r₁ × (pbest_i - x_i) + c₂ × β(d) × (gbest - x_i)
        where β(d) = β₀ × exp(-γ × d²) is distance-dependent social coefficient
      - Update position: x_i(t+1) = x_i(t) + v_i(t+1)
      - Evaluate fitness f(x_i(t+1))
      - Update personal best if f(x_i(t+1)) > f(pbest_i)
   b. Update global best from all personal bests
   c. Linearly decrease inertia weight w(t)

Discrete Adaptation:
- Velocity represented as set of nodes
- Distance-based attraction uses Hamming distance between solutions
- Destroy-repair operators for position updates
```

**Key Parameters:**

- `inertia_weight_start` (w₀): Initial inertia for exploration (0.9)
- `inertia_weight_end` (w_T): Final inertia for exploitation (0.4)
- `cognitive_coef` (c₁): Personal best attraction coefficient (2.0)
- `social_coef` (c₂): Global best base attraction coefficient (2.0)
- `initial_attraction` (β₀): Distance-based attraction scaling (1.0)
- `distance_decay` (γ): Exponential decay rate (0.01)

**Why PSO-DA Replaces Firefly Algorithm:**

The Firefly Algorithm (FA) obscures PSO mechanics with "light intensity" metaphor:

```
FA Update: x_i = x_i + β₀ × exp(-γ × d²) × (x_j - x_i) + α × random_walk
PSO-DA:    v_i = w × v_i + c₁ × r₁ × (pbest - x_i) + c₂ × β(d) × (gbest - x_i)
```

**PSO-DA Improvements over FA:**

1. ✓ **Velocity momentum** (inertia term w×v) - MISSING in FA
2. ✓ **Personal best tracking** (cognitive term) - MISSING in FA
3. ✓ **Distance-based social attraction** - Same as FA's β(d), but explicit
4. ✓ **Linearly decreasing inertia** - Exploration → exploitation
5. ✓ **Proper PSO foundation** - 30+ years of theory vs firefly metaphor

**Terminology Mapping (FA → PSO-DA):**

- "Fireflies" → Particles with velocity and personal best
- "Light intensity" → Fitness value
- "Attractiveness β(d)" → Distance-dependent social coefficient
- "Random walk" → Exploration via velocity momentum

**Mathematical Foundation:**

PSO-DA Velocity Update:

```
v_i(t+1) = w(t) × v_i(t) + c₁ × r₁ × (pbest_i - x_i) + c₂ × β(d_ij) × (x_j - x_i)
```

Where:

- **Inertia term (w×v)**: Maintains previous movement direction
- **Cognitive term (c₁·(pbest - x))**: Learns from personal best
- **Social term (c₂·β(d)·(gbest - x))**: Learns from swarm best with distance weighting

Distance-Dependent Attraction:

```
β(d_ij) = β₀ × exp(-γ × d²)
d = Hamming distance between solutions i and j
```

Inertia Weight Decay:

```
w(t) = w_start - (w_start - w_end) × (t / T_max)
```

**Complexity:**

- Time: O(T × N² × n²) where T = iterations, N = swarm size
  - N² from pairwise distance calculations
  - n² from Hamming distance computation
- Space: O(N × n) for swarm + velocities + personal bests

**Reference:**

> Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization." Proceedings of ICNN'95 - International Conference on Neural Networks.

> Yang, X.-S. (2008). "Nature-Inspired Metaheuristic Algorithms." Luniver Press. [Note: Firefly Algorithm is PSO with distance-based attraction but without velocity momentum - superseded by this implementation]

---

##### 3. Differential Evolution (DE)

**Replaces:** Artificial Bee Colony (ABC)

**Location:** `logic/src/policies/differential_evolution/`

**Algorithm:**

```
TRUE DIFFERENTIAL EVOLUTION with greedy selection (Storn & Price 1997):

1. Initialize population of NP solution vectors
2. For each generation:
   a. For each target vector x_i:
      - Mutation: Create mutant v_i = x_r1 + F × (x_r2 - x_r3)
        where r1, r2, r3 are distinct random indices ≠ i
      - Crossover: Create trial u_i by binomial crossover:
        u_ij = v_ij  if rand() < CR or j = j_rand
               x_ij  otherwise
      - Selection: Greedy replacement:
        x_i(t+1) = u_i  if f(u_i) ≥ f(x_i)
                   x_i  otherwise
   b. Track global best solution

Discrete Adaptation:
- Differential mutation via set operations on node sets
- Destroy-repair operators for discrete routing space
- Local search applied to trial vectors (memetic DE)
```

**Key Parameters:**

- `pop_size` (NP): Population size (50)
- `mutation_factor` (F): Differential weight ∈ [0, 2] (0.8)
  - Controls amplification of differential variation (x_r2 - x_r3)
- `crossover_rate` (CR): Crossover probability ∈ [0, 1] (0.9)
  - Controls inheritance from mutant vs. target vector
- `n_removal`: Mutation strength for discrete operators (3)

**Why DE Replaces ABC:**

The Artificial Bee Colony is mathematically isomorphic to Differential Evolution with fitness-proportionate selection:

```
ABC Employed: v = x_i + φ(x_i - x_k)  [peer-based perturbation]
DE Mutation:  v = x_r1 + F × (x_r2 - x_r3)  [differential mutation]

ABC Onlooker: Roulette-wheel selection (fitness-proportionate)
DE Selection: Greedy one-to-one replacement

ABC Scout: Abandon sources if trials > limit
DE: No abandonment (differential mutation provides diversity)
```

**ABC's Flaws:**

1. Fitness-proportionate selection is slower than greedy selection
2. Three agent types (employed/onlooker/scout) are unnecessary abstractions
3. Trial counter and limit parameter add complexity without benefit
4. "Food source" metaphor obscures actual DE mechanics
5. Often requires canonical GA crossover injections for discrete spaces

**DE's Advantages:**

1. ✓ Greedy selection is faster and more effective
2. ✓ Explicit crossover operator (CR parameter) controls exploration
3. ✓ Single population - no metaphorical agent types
4. ✓ Simpler algorithm with proven convergence properties
5. ✓ 25+ years of theoretical foundation vs bee foraging metaphor

**Terminology Mapping (ABC → DE):**

- "Employed bee exploiting food source" → Target vector update via mutation+crossover
- "Onlooker bee probabilistic selection" → Greedy one-to-one selection
- "Scout bee abandonment" → Differential mutation (no explicit abandonment)
- ~~"Food source quality"~~ → **Proper DE**: Fitness for greedy selection

**Mathematical Foundation:**

DE/rand/1/bin Strategy:

```
Mutation (rand/1):
v_i = x_r1 + F × (x_r2 - x_r3)

Crossover (bin - binomial):
u_ij = v_ij  if rand() < CR or j = j_rand
       x_ij  otherwise

Selection (greedy):
x_i(t+1) = u_i  if f(u_i) ≥ f(x_i)
           x_i  otherwise
```

Where:

- **Mutation**: Differential variation scaled by F
- **Crossover**: Binomial inheritance from mutant vs. target
- **Selection**: Deterministic greedy replacement (not probabilistic)

**Complexity:**

- Time: O(G × NP × n²) where G = generations, NP = pop_size
- Space: O(NP × n) to store population
- **Faster than ABC**: No fitness-proportionate selection overhead

**References:**

> Storn, R., & Price, K. (1997). "Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces." Journal of Global Optimization, 11(4), 341-359.

> Karaboga, D. (2005). "An idea based on honey bee swarm for numerical optimization." Technical Report TR06, Erciyes University. [Note: ABC is DE with fitness-proportionate selection - superseded by this implementation]

##### 4. Genetic Algorithm Memetic Island Model (GA-MIM)

**Replaces:** Soccer League Competition (SLC)

**Location:** `logic/src/policies/genetic_algorithm_memetic_island_model/`

**Algorithm:**

```

1. Initialize K islands with N individuals each
2. For each generation:
   a. Intra-island Evolution:
   i. Perturbation: Apply random removal + greedy insertion
   ii. Local Search: Apply ACOLocalSearch refinement
   iii. Survival: Keep improved solution
   b. Inter-island Competition:
   i. Stochastic Tournament: Pairwise competition between islands
   ii. Recombination: Weak island adopts structure from strongest via crossover
   c. Stagnation Check: If island fitness unchanged for stagnation_limit, regenerate
   d. Global best: Track champion across all islands

```

**Key Parameters:**

- `n_islands` (K): Number of parallel sub-populations (islands)
- `island_size` (N): Individuals per island
- `local_search_iterations`: Search budget for ACOLocalSearch
- `stagnation_limit`: Generations before island regeneration

**Terminology Mapping:**

- "Teams" → Islands (sub-populations)
- "Players" → Individuals
- "Seasons" → Generations
- "Intra-team Competition" → Intra-island evolution (Perturbation + LS)
- "Inter-team Competition" → Inter-island tournament + recombination
- "Stagnation" → Stagnation limit/Regeneration

**Complexity:**

- Time: O(T × K × N × LS_cost)
- Space: O(K × N × n)

**Reference:**

> Whitley, D., Rana, S., & Heckendorn, R. B. (1998). "The island model genetic algorithm."

---

##### 5. Hybrid Memetic Search (HMS)

**Replaces:** Hybrid Volleyball Premier League (HVPL)

**Location:** `logic/src/policies/hybrid_memetic_search/`

**Algorithm:**

HMS utilizes a **3-Phase Hybrid Pipeline** to balance construction, exploration, and exploitation:

```

1. Phase 1: ACO Construction (K-Sparse Ant Colony Optimization)
   - Initialize diverse population using probabilistic pheromone-guided construction.
2. Phase 2: GA Evolution (HGS-inspired Genetic Search)
   - Evolve solutions via population-based crossover and diversity management.
   - Maintain a "Passive Reserve Pool" for elite substitution.
3. Phase 3: ALNS Refinement (Adaptive Large Neighborhood Search)
   - Apply trajectory search to the best discovered solutions for fine-grained refinement.

```

**Key Parameters:**

- `n_teams`: Population size
- `max_iterations`: Main loop seasons
- `reserve_pool_size`: Size of the elite replacement buffer
- `aco_params`: Parameters for the construction phase
- `alns_params`: Parameters for the refinement phase

**Terminology Mapping:**

- "Teams" → Population members
- "Coaching" → ALNS refinement
- "Competition" → Selection/Crossover
- "Pheromone Update" → Global guidance/Learning
- "Substitution" → Passive reserve pool replacement

**Complexity:**

- Time: O(ACO_cost + T × (GA_cost + ALNS_cost))
- Space: O(Population_size × n)

---

##### 6. Island Model STGA (IMGA-ST)

**Replaces:** Volleyball Premier League (VPL)

**Location:** `logic/src/policies/genetic_algorithm_island_model_stochastic_tournaments/`

**Algorithm:**

```

1. Initialize K islands with N individuals each
2. For each generation:
   a. Local Improvement: Reconstruct via greedy profit insertion + ALNS
   b. Stochastic Tournament: Pairwise competition using sigmoid win probability
   c. Crossover: Ordered recombination of selected parents
   d. Mutation: Population perturbation via random node removal
   e. Migration: Ring-topology elite exchange between islands
   f. Record: Global best solution tracking

```

**Key Parameters:**

- `n_islands`: Number of parallel sub-populations
- `island_size`: Population size per island
- `selection_pressure (β)`: Sigmoid coefficient for win probability
- `migration_interval`: Generations between migration events
- `crossover_rate`: Probability of recombination

**Terminology Mapping:**

- "Active Teams" → Island individuals (sub-populations)
- "Passive Teams" → Diversity reservoir (migration pool)
- "Substitution" → Mutation/Migration operators
- "Coaching/Learning" → Stochastic tournament + crossover + ALNS
- "Seasons" → Generations

**Selection Probability:**
P(i beats j) = σ(β × (f(i) - f(j)))

**Reference:**

> Moghdani, R., & Salimifard, K. (2018). "Volleyball Premier League Algorithm." Applied Soft Computing.

---

##### 7. Stochastic Tournament Genetic Algorithm

**Replaces:** League Championship Algorithm (LCA)

**Location:** `logic/src/policies/genetic_algorithm_stochastic_tournament/`

**Algorithm:**

```

1. Initialize population of N chromosomes
2. For each generation:
   a. Fitness evaluation for all individuals
   b. Stochastic tournament selection:
   - Each individual competes against k random opponents
   - Win probability: P(i > j) = σ(β × (f_i - f_j))
   - Select winners for mating pool
     c. Crossover (recombination) of selected parents
     d. Mutation of offspring
     e. Elitist replacement (keep best individuals)

```

**Key Parameters:**

- `population_size` (N): Number of chromosomes
- `tournament_competitors` (k): Opponents per individual
- `selection_pressure` (β): Sigmoid coefficient for win probability
- `crossover_rate`: Probability of recombination
- `mutation_rate`: Probability of perturbation
- `elitism_count`: Top individuals preserved unchanged

**Terminology Mapping:**

- "League Schedule/Fixtures" → Pairwise fitness evaluation cycles
- "Playing Strength" → Objective function value (fitness)
- "Match Outcome (Win/Loss)" → Stochastic tournament result
- "Team Formation" → Crossover/recombination operator

**Mathematical Foundation:**

- **Stochastic Tournament Selection:** P(i defeats j) = σ(β × (f(i) - f(j)))
- **Sigmoid Function:** σ(x) = 1/(1 + exp(-x))
- **Selection Pressure:** Higher β → more deterministic selection

**Complexity:**

- Time: O(T × N × k × eval_cost) where T = generations
- Space: O(N × n)
- Selection: O(N × k) for tournament comparisons

**Reference:**

> Goldberg, D. E., & Deb, K. (1991). "A comparative analysis of selection schemes used in genetic algorithms." Foundations of Genetic Algorithms.

---

##### 8. Particle Swarm Optimization (PSO)

**Replaces:** Sine Cosine Algorithm (SCA)

**Location:** `logic/src/policies/particle_swarm_optimization/`

**Algorithm:**

```

TRUE PSO with inertia-weighted velocity updates (Kennedy & Eberhart 1995):

1. Initialize swarm with random positions and velocities
2. For each iteration:
   a. For each particle i:
      - Update velocity: v(t+1) = w*v(t) + c₁*r₁*(pbest - x) + c₂*r₂*(gbest - x)
      - Update position: x(t+1) = x(t) + v(t+1)
      - Clamp velocity and position to bounds
      - Evaluate fitness f(x(t+1))
      - Update personal best if f(x(t+1)) > f(pbest_i)
   b. Update global best from all personal bests
   c. Linearly decrease inertia weight w

Encoding:
- Continuous position vectors in [-1, 1]^n
- Sigmoid binarization for node selection: select if sigmoid(x_i) > 0.5
- Largest Rank Value (LRV) ordering: sort by x_i descending

```

**Key Parameters:**

- `inertia_weight_start` (w₀): Initial inertia for exploration (0.9)
- `inertia_weight_end` (w_T): Final inertia for exploitation (0.4)
- `cognitive_coef` (c₁): Personal best attraction coefficient (2.0)
- `social_coef` (c₂): Global best attraction coefficient (2.0)
- `velocity_max`: Maximum velocity magnitude (clamping)

**Why PSO Replaces SCA:**

The Sine Cosine Algorithm is mathematically identical to PSO without velocity momentum:

```
SCA Update: X' = X + r₁·sin(r₂)·|r₃·P - X|
PSO Social: X' = X + w·(G_best - X)  where w is simple random weight
```

**SCA's Flaws:**

1. sin(r₂) where r₂~U(0,2π) is just random weight ∈ [-1,1] (expensive transcendental call)
2. No periodicity exploitation (r₂ resampled every iteration)
3. cos/sin switch is redundant (same distribution, phase-shifted)
4. **MISSING velocity momentum** - no memory of previous movements
5. **MISSING personal best** - no individual particle learning

**PSO's Advantages:**

1. ✓ Velocity maintains momentum (inertia term w\*v)
2. ✓ Personal best enables individual learning (cognitive term)
3. ✓ Simpler arithmetic (no transcendental functions)
4. ✓ Linearly decreasing inertia (exploration → exploitation)
5. ✓ 30+ years of theoretical foundation vs metaphor-based naming

**Terminology Mapping (SCA → PSO):**

- "Sine/Cosine oscillation" → Velocity momentum
- "Random position update" → Position = Position + Velocity
- "Best solution attraction" → Social term (gbest attraction)
- ~~"Light intensity"~~ → **Proper PSO**: Personal + Global best

**Mathematical Foundation:**

PSO Velocity Update:

```
v(t+1) = w*v(t) + c₁*r₁*(pbest - x(t)) + c₂*r₂*(gbest - x(t))
```

Where:

- **Inertia term (w\*v)**: Maintains previous movement direction
- **Cognitive term (c₁·(pbest - x))**: Learns from personal best
- **Social term (c₂·(gbest - x))**: Learns from swarm best

Inertia Weight Decay (Shi & Eberhart 1998):

```
w(t) = w_start - (w_start - w_end) × (t / T_max)
```

**Complexity:**

- Time: O(T × N × n) where T = iterations, N = pop_size
- Space: O(N × n) for swarm + velocities + personal bests
- **Faster than SCA**: No sin/cos transcendental function calls

**References:**

> Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization." Proceedings of ICNN'95 - International Conference on Neural Networks.

> Shi, Y., & Eberhart, R. (1998). "A modified particle swarm optimizer." IEEE International Conference on Evolutionary Computation.

> Mirjalili, S. (2016). "SCA: A Sine Cosine Algorithm..." Knowledge-Based Systems. [Note: SCA is PSO without velocity - superseded by this implementation]

---

#### Usage Examples

##### Example 1: (μ+λ) Evolution Strategy

```python
from logic.src.policies import MuPlusLambdaESSolver, MuPlusLambdaESParams

# Configure parameters
params = MuPlusLambdaESParams(
    population_size=10,        # μ parameter
    offspring_size=5,          # λ parameter
    recombination_rate=0.95,   # Archive recombination probability
    mutation_rate=0.3,         # Local mutation probability
    max_iterations=500,
    time_limit=60.0
)

# Initialize solver
solver = MuPlusLambdaESSolver(
    dist_matrix=distance_matrix,
    wastes=waste_dict,
    capacity=100.0,
    R=1.0,
    C=1.0,
    params=params,
    mandatory_nodes=[1, 5, 10],
    seed=42
)

# Execute optimization
best_routes, best_profit, best_cost = solver.solve()
```

##### Example 2: Distance-Based PSO

```python
from logic.src.policies import DistancePSOSolver, DistancePSOParams

params = DistancePSOParams(
    population_size=20,
    initial_attraction=1.0,    # β₀
    distance_decay=0.01,       # γ
    exploration_rate=0.1,      # α
    max_iterations=500
)

solver = DistancePSOSolver(
    dist_matrix=distance_matrix,
    wastes=waste_dict,
    capacity=100.0,
    R=1.0,
    C=1.0,
    params=params,
    seed=42
)

best_routes, best_profit, best_cost = solver.solve()
```

##### Example 3: Particle Swarm Optimization (PSO) - Replaces SCA

```python
from logic.src.policies import PSOSolver, PSOParams

# TRUE PSO with velocity momentum (replaces Sine Cosine Algorithm)
params = PSOParams(
    pop_size=30,
    inertia_weight_start=0.9,  # w(0) - exploration
    inertia_weight_end=0.4,    # w(T) - exploitation
    cognitive_coef=2.0,        # c₁ - personal best
    social_coef=2.0,           # c₂ - global best
    velocity_max=0.5,          # velocity clamping
    max_iterations=500
)

solver = PSOSolver(
    dist_matrix=distance_matrix,
    wastes=waste_dict,
    capacity=100.0,
    R=1.0,
    C=1.0,
    params=params,
    seed=42
)

best_routes, best_profit, best_cost = solver.solve()

# Note: This implementation is superior to SCA because:
# - Velocity momentum (MISSING in SCA)
# - Personal best tracking (MISSING in SCA)
# - No expensive sin/cos calls (SCA uses transcendental functions)
```

##### Example 4: Differential Evolution (DE/rand/1/bin) - Replaces ABC

```python
from logic.src.policies import DESolver, DEParams

# TRUE DE with differential mutation and binomial crossover (replaces ABC)
params = DEParams(
    pop_size=50,             # NP - population size
    mutation_factor=0.8,     # F - differential weight
    crossover_rate=0.9,      # CR - crossover probability
    n_removal=3,             # mutation strength for discrete operators
    max_iterations=500,
    local_search_iterations=100
)

solver = DESolver(
    dist_matrix=distance_matrix,
    wastes=waste_dict,
    capacity=100.0,
    R=1.0,
    C=1.0,
    params=params,
    seed=42
)

best_routes, best_profit, best_cost = solver.solve()

# Note: This implementation is superior to ABC because:
# - Greedy selection (FASTER than ABC's fitness-proportionate)
# - Explicit crossover with CR parameter (MISSING in ABC)
# - No metaphorical agent types (employed/onlooker/scout)
# - Simpler algorithm with proven convergence properties
```

##### Example 5: Memetic Algorithm Island Model (MA-IM)

```python
from logic.src.policies import MemeticAlgorithmIslandModelSolver, MemeticAlgorithmIslandModelParams

params = MemeticAlgorithmIslandModelParams(
    n_islands=10,              # K sub-populations
    island_size=10,            # N individuals per island
    max_generations=50,
    migration_interval=5,
    tournament_size=3
)

solver = MemeticAlgorithmIslandModelSolver(
    dist_matrix=distance_matrix,
    wastes=waste_dict,
    capacity=100.0,
    R=1.0,
    C=1.0,
    params=params,
    seed=42
)

best_routes, best_profit, best_cost = solver.solve()
```

---

#### Performance Characteristics

| Algorithm                | Time Complexity   | Space Complexity               | Best Use Case                                   |
| ------------------------ | ----------------- | ------------------------------ | ----------------------------------------------- |
| (μ+λ)-ES                 | O(T × λ × n²)     | O((μ+λ) × n)                   | Small-medium instances, fast convergence        |
| Distance-Based PSO       | O(T × N² × n²)    | O(N × n)                       | Medium instances, global exploration            |
| PSO (Velocity)           | O(T × N × n)      | O(N × n + velocities + pbests) | Continuous optimization, replaces SCA           |
| DE                       | O(G × NP × n²)    | O(NP × n)                      | Global optimization, replaces ABC               |
| GA-MIM                   | O(T × K × N × LS) | O(K × N × n)                   | Large instances, parallel island execution      |
| HMS                      | O(T × Pop × n²)   | O(Pop × n)                     | Hybrid search (ACO+GA+ALNS)                     |
| Stochastic Tournament GA | O(T × N × k × n²) | O(N × n)                       | Medium instances, controlled selection pressure |

**Note:** PSO (Velocity) is faster than Distance-Based PSO (O(N) vs O(N²)) and significantly faster than SCA (no transcendental function overhead).

---

#### Deprecation Notice

The following metaphor-based implementations are now superseded by rigorous alternatives:

- ❌ `harmony_search/` → Use `evolution_strategy_mu_plus_lambda/`
- ❌ `firefly_algorithm/` → Use `particle_swarm_optimization_distance_based_algorithm/`
- ❌ `artificial_bee_colony/` → Use `differential_evolution/`
- ❌ `hybrid_volleyball_premier_league/` → Use `hybrid_memetic_search/`
- ❌ `league_championship_algorithm/` → Use `genetic_algorithm_stochastic_tournament/`
- ❌ `soccer_league_competition/` → Use `genetic_algorithm_memetic_island_model/`
- ❌ `sine_cosine_algorithm/` → Use `particle_swarm_optimization/`

The original implementations remain for backward compatibility but should not be used for new development.

**Note on SCA:** The Sine Cosine Algorithm is mathematically equivalent to PSO without velocity momentum, but with expensive trigonometric operations (sin/cos) that provide no optimization benefit. The new `particle_swarm_optimization/` implementation includes proper velocity momentum, personal best tracking, and uses simple arithmetic for superior performance.

**Note on ABC:** The Artificial Bee Colony is mathematically isomorphic to Differential Evolution (DE) with fitness-proportionate selection instead of greedy selection, but with unnecessary "bee foraging" metaphor (employed/onlooker/scout bees). The new `differential_evolution/` implementation uses proper DE/rand/1/bin mechanics with greedy selection, explicit crossover parameter (CR), and simpler algorithm structure for superior performance.

---

## 6. Exact Optimization

### 6.1 BPC (Branch-and-Price-and-Cut)

**Directory**: `branch_cut_and_price/`
**Adapters**: `policy_bpc.py`

Exact MIP solvers for optimal solutions.

#### Engine Selection

```python
from logic.src.policies import run_bpc

# Dispatcher selects best available engine
routes, cost = run_bpc(
    distance_matrix=dist_matrix,
    wastes=wastes_dict,
    capacity=200.0,
    engine="auto"  # Auto-select: Gurobi > OR-Tools > VRPy
)

# Force specific engine
routes, cost = run_bpc(
    distance_matrix=dist_matrix,
    wastes=wastes_dict,
    capacity=200.0,
    engine="gurobi",  # or "ortools", "vrpy"
    time_limit=300.0,
    mip_gap=0.01
)
```

**Available Engines**:

1. **Gurobi** (`gurobi_engine.py`): Commercial solver (fastest, requires license)
2. **OR-Tools** (`ortools_engine.py`): Google's open-source solver
3. **VRPy** (`vrpy_engine.py`): Python-based column generation

#### Usage via Adapter

```python
bpc_policy = PolicyFactory.get_adapter("bpc")
tour, cost, _ = bpc_policy.execute(
    packing=packing_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config={
        "bpc": {
            "engine": "gurobi",
            "time_limit": 600.0,
            "mip_gap": 0.001  # 0.1% optimality tolerance
        }
    }
)
```

### 6.2 VRPP Exact Solvers

**Directory**: `vehicle_routing_problem_with_profits/`
**Adapters**: `policy_swc_tcf.py`

Exact optimization for VRPP (profit-oriented VRP).

```python
from logic.src.policies import run_swc_tcf_optimizer

# Gurobi SWC-TCF
routes, cost, selected_nodes = run_swc_tcf_optimizer(
    distance_matrix=dist_matrix,
    wastes=wastes_dict,  # Profit per node
    capacity=200.0,
    max_length=100.0,
    engine="gurobi",
    Omega=1.0,      # Profit weight
    delta=0.5,      # Distance penalty
    psi=10.0        # Unvisited penalty
)

# Hexaly VRPP (high-performance local search)
routes, cost, selected_nodes = run_vrpp_optimizer(
    distance_matrix=dist_matrix,
    wastes=wastes_dict,
    capacity=200.0,
    engine="hexaly",
    time_limit=60.0
)
```

---

## 7. Neural Policies

**Directory**: `neural_agent/`
**Adapters**: `policy_neural.py`

Deep Reinforcement Learning-based routing policies.

### 7.1 NeuralAgent

**File**: `neural_agent/agent.py`

Wrapper around trained neural models (AM, TAM, DDAM).

```python
from logic.src.policies import NeuralAgent
from logic.src.models import load_model

# Load trained model
model = load_model("model_weights/am_wcvrp_100.pt")

# Create agent
agent = NeuralAgent(
    model=model,
    device="cuda",
    decode_type="greedy",  # or "sampling", "beam_search"
    temperature=1.0,
    beam_width=5
)

# Solve single instance
tour, cost = agent.solve(
    coordinates=coords,
    wastes=wastes,
    capacity=200.0
)

# Solve batch
tours, costs = agent.solve_batch(
    batch_coords=batch_coords,
    batch_wastes=batch_wastes,
    capacity=200.0
)
```

### 7.2 Via Adapter

```python
neural_policy = PolicyFactory.get_adapter("neural")

# Configure model path
tour, cost, _ = neural_policy.execute(
    packing=packing_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config={
        "neural": {
            "model_path": "model_weights/best_model.pt",
            "decode_type": "beam_search",
            "beam_width": 10
        }
    }
)
```

### 7.3 Supported Models

| Model        | Architecture      | Best For                          |
| ------------ | ----------------- | --------------------------------- |
| **AM**       | Attention Model   | General VRP, fast inference       |
| **DDAM**     | Deep Decoder AM   | Large-scale problems (200+ nodes) |
| **TAM**      | Temporal AM       | Multi-day scenarios, WCVRP        |
| **TransGCN** | Transformer + GCN | Graph-structured problems         |

---

## 8. Operators Library

**Directory**: `operators/`
**Purpose**: 26 specialized operators for solution manipulation

### 8.1 Operator Categories

1. **Move Operators** (`move/`): Single-node relocations
2. **Route Operators** (`route/`): Intra-route improvements
3. **Destroy Operators** (`destroy/`): Solution deconstruction (ALNS)
4. **Repair Operators** (`repair/`): Solution reconstruction (ALNS)
5. **Perturbation Operators** (`perturbation/`): Diversification
6. **Exchange Operators** (`exchange/`): Multi-node swaps

### 8.1 Move Operators

**Directory**: `operators/move/`

#### Relocate

**File**: `relocate.py`

Move a single node from one route to another.

```python
from logic.src.policies.operators import move_relocate

# Relocate node 5 from route 0 position 2 to route 1 position 1
new_routes, delta_cost = move_relocate(
    routes=[[0, 3, 5, 7, 0], [0, 2, 4, 0]],
    distance_matrix=dist_matrix,
    from_route=0,
    from_pos=2,
    to_route=1,
    to_pos=1
)
# Result: [[0, 3, 7, 0], [0, 2, 5, 4, 0]]
```

#### Swap

**File**: `swap.py`

Exchange two nodes between or within routes.

```python
from logic.src.policies.operators import move_swap

# Swap node at route 0 position 1 with node at route 1 position 1
new_routes, delta_cost = move_swap(
    routes=[[0, 3, 5, 0], [0, 2, 4, 0]],
    distance_matrix=dist_matrix,
    route1=0,
    pos1=1,
    route2=1,
    pos2=1
)
# Result: [[0, 2, 5, 0], [0, 3, 4, 0]]
```

### 8.2 Route Operators

**Directory**: `operators/route/`

#### 2-Opt Intra-Route

**File**: `two_opt_intra.py`

Reverse a segment within a single route.

```python
from logic.src.policies.operators import move_2opt_intra

# 2-opt on route 0 between positions 1 and 3
new_routes, delta_cost = move_2opt_intra(
    routes=[[0, 3, 5, 7, 9, 0]],
    distance_matrix=dist_matrix,
    route_idx=0,
    i=1,
    j=3
)
# [0, 3, 5, 7, 9, 0] → [0, 3, 7, 5, 9, 0] (reversed segment 5-7)
```

#### 2-Opt\* (Inter-Route)

**File**: `two_opt_star.py`

Exchange tails between two routes.

```python
from logic.src.policies.operators import move_2opt_star

# Exchange tails after position 1 in both routes
new_routes, delta_cost = move_2opt_star(
    routes=[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]],
    distance_matrix=dist_matrix,
    route1=0,
    route2=1,
    pos1=1,
    pos2=1
)
# Result: [[0, 1, 5, 6, 0], [0, 4, 2, 3, 0]]
```

#### 3-Opt Intra-Route

**File**: `three_opt_intra.py`

Advanced intra-route optimization with 3 edges.

```python
from logic.src.policies.operators import move_3opt_intra

new_routes, delta_cost = move_3opt_intra(
    routes=[[0, 1, 2, 3, 4, 5, 0]],
    distance_matrix=dist_matrix,
    route_idx=0,
    i=1,
    j=3,
    k=5
)
```

### 8.3 Destroy Operators (ALNS)

**Directory**: `operators/destroy/`

#### Random Removal

**File**: `random.py`

Remove random nodes from routes.

```python
from logic.src.policies.operators import random_removal

# Remove 5 random nodes
removed_nodes, modified_routes = random_removal(
    routes=[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]],
    num_remove=5,
    rng=np.random.default_rng(42)
)
```

#### Worst Removal

**File**: `worst.py`

Remove nodes with highest cost contribution.

```python
from logic.src.policies.operators import worst_removal

# Remove 3 worst nodes
removed_nodes, modified_routes = worst_removal(
    routes=[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]],
    distance_matrix=dist_matrix,
    num_remove=3,
    randomness=0.1  # Add some randomness
)
```

#### Cluster Removal

**File**: `cluster.py`

Remove spatially clustered nodes.

```python
from logic.src.policies.operators import cluster_removal

# Remove a cluster of 4 nodes
removed_nodes, modified_routes = cluster_removal(
    routes=[[0, 1, 2, 3, 4, 5, 0]],
    distance_matrix=dist_matrix,
    num_remove=4,
    rng=np.random.default_rng(42)
)
```

#### Shaw Removal

**File**: `shaw.py`

Remove similar nodes (by distance, waste, time).

```python
from logic.src.policies.operators import shaw_removal

# Remove 4 similar nodes
removed_nodes, modified_routes = shaw_removal(
    routes=[[0, 1, 2, 3, 4, 5, 0]],
    distance_matrix=dist_matrix,
    wastes=wastes_dict,
    num_remove=4,
    relatedness_weights=(0.5, 0.3, 0.2)  # distance, waste, time
)
```

#### String Removal (SISR)

**File**: `string.py`

Remove consecutive subsequences.

```python
from logic.src.policies.operators import string_removal

# Remove strings of length 2-4
removed_nodes, modified_routes = string_removal(
    routes=[[0, 1, 2, 3, 4, 5, 0]],
    min_string_len=2,
    max_string_len=4,
    num_strings=2
)
```

### 8.4 Repair Operators (ALNS)

**Directory**: `operators/repair/`

#### Greedy Insertion

**File**: `greedy.py`

Insert nodes at best positions.

```python
from logic.src.policies.operators import greedy_insertion

# Insert removed nodes greedily
new_routes = greedy_insertion(
    routes=[[0, 1, 0], [0, 3, 0]],
    unrouted_nodes=[2, 4, 5],
    distance_matrix=dist_matrix,
    wastes=wastes_dict,
    capacity=200.0
)
```

#### Regret-k Insertion

**File**: `regret.py`

Insert nodes based on regret value (difference between best and k-th best position).

```python
from logic.src.policies.operators import regret_k_insertion

# Regret-2 insertion
new_routes = regret_k_insertion(
    routes=[[0, 1, 0]],
    unrouted_nodes=[2, 3, 4],
    distance_matrix=dist_matrix,
    wastes=wastes_dict,
    capacity=200.0,
    k=2  # Regret-2
)
```

#### Greedy Blink Insertion (SISR)

**File**: `greedy_blink.py`

Greedy insertion with random "blinks" for exploration.

```python
from logic.src.policies.operators import greedy_insertion_with_blinks

# Insert with 1% blink probability
new_routes = greedy_insertion_with_blinks(
    routes=[[0, 1, 0]],
    unrouted_nodes=[2, 3, 4],
    distance_matrix=dist_matrix,
    wastes=wastes_dict,
    capacity=200.0,
    blink_probability=0.01
)
```

### 8.6 Exchange Operators

**Directory**: `operators/exchange/`

#### Or-Opt

**File**: `or_opt.py`

Move a chain of 1-3 consecutive nodes.

```python
from logic.src.policies.operators import move_or_opt

# Move chain of length 2
new_routes, delta_cost = move_or_opt(
    routes=[[0, 1, 2, 3, 4, 0]],
    distance_matrix=dist_matrix,
    route_idx=0,
    start_pos=1,
    chain_length=2,
    insert_pos=4
)
# [0, 1, 2, 3, 4, 0] → [0, 3, 4, 1, 2, 0]
```

#### Cross-Exchange

**File**: `cross.py`

Exchange segments between two routes.

```python
from logic.src.policies.operators import cross_exchange

# Exchange segments
new_routes, delta_cost = cross_exchange(
    routes=[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]],
    distance_matrix=dist_matrix,
    route1=0,
    route2=1,
    start1=1,
    len1=2,
    start2=1,
    len2=2
)
```

#### Ejection Chain

**File**: `ejection.py`

Complex multi-route reconfiguration.

```python
from logic.src.policies.operators import ejection_chain

# Perform ejection chain
new_routes, delta_cost = ejection_chain(
    routes=[[0, 1, 2, 0], [0, 3, 4, 0], [0, 5, 6, 0]],
    distance_matrix=dist_matrix,
    chain_length=3
)
```

#### λ-Interchange

**File**: `lambda_interchange.py`

Exchange up to λ nodes between routes.

```python
from logic.src.policies.operators import lambda_interchange

# (1,1)-interchange
new_routes, delta_cost = lambda_interchange(
    routes=[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]],
    distance_matrix=dist_matrix,
    lambda1=1,
    lambda2=1
)
```

---

## 9. packing algorithms

**Directory**: `other/packing/`
**Purpose**: Pre-selection of mandatory collection nodes

### 9.1 Selection Framework

#### SelectionContext

**File**: `base/selection_context.py`

Context object for selection decisions.

```python
from logic.src.policies.helpers import SelectionContext

context = SelectionContext(
    bins=bins_state,
    distance_matrix=dist_matrix,
    area="riomaior",
    waste_type="plastic",
    day=5,
    history=historical_data
)
```

#### PackingFactory

**File**: `base/selection_factory.py`

Factory for creating selectors.

```python
from logic.src.policies.helpers import PackingFactory

# Create selector
selector = PackingFactory.create_selector(
    strategy="last_minute",
    threshold=0.9
)

# Execute selection
packing_bins = selector.select(context)
```

### 9.2 Selection Strategies

#### Last-Minute Selector

**File**: `selection_last_minute.py`

Select bins exceeding fill threshold.

```python
from logic.src.policies.helpers.packing import LastMinuteSelector

selector = LastMinuteSelector(threshold=0.9)

# Select bins with fill ≥ 90%
packing = selector.select(context)
```

#### Regular Selector

**File**: `selection_regular.py`

Select bins on fixed-frequency schedule.

```python
from logic.src.policies.helpers.packing import RegularSelector

# Collect every 3 days
selector = RegularSelector(frequency=3)

packing = selector.select(context)
```

#### Lookahead Selector

**File**: `selection_lookahead.py`

Select bins predicted to overflow within N days.

```python
from logic.src.policies.helpers.packing import LookaheadSelector

# Predict 7 days ahead
selector = LookaheadSelector(
    lookahead_days=7,
    predictor=fill_predictor_model
)

packing = selector.select(context)
```

#### Revenue Selector

**File**: `selection_revenue.py`

Select bins where profit exceeds cost.

```python
from logic.src.policies.helpers.packing import RevenueSelector

selector = RevenueSelector(
    revenue_per_kg=1.0,
    cost_per_km=0.5
)

packing = selector.select(context)
```

#### Service-Level Selector

**File**: `selection_service_level.py`

Statistical overflow prediction.

```python
from logic.src.policies.helpers.packing import ServiceLevelSelector

# 95% service level
selector = ServiceLevelSelector(
    service_level=0.95,
    fill_std_dev=0.1
)

packing = selector.select(context)
```

#### Combined Selector

**File**: `selection_combined.py`

Combine multiple strategies with AND/OR logic.

```python
from logic.src.policies.helpers.packing import CombinedSelector

selector = CombinedSelector(
    logic="or",  # or "and"
    strategies=[
        {"strategy": "last_minute", "threshold": 0.9},
        {"strategy": "regular", "frequency": 7}
    ]
)

# Select if last_minute OR regular triggers
packing = selector.select(context)
```

### 9.3 Usage in Simulation

```python
from logic.src.policies import create_policy
from logic.src.policies.helpers import PackingFactory

# Create policy
policy = create_policy("hgs")

# Create mandatory selector
selector = PackingFactory.create_selector(
    strategy="combined",
    logic="or",
    strategies=[
        {"strategy": "last_minute", "threshold": 0.85},
        {"strategy": "regular", "frequency": 5}
    ]
)

# Execute selection
packing = selector.select(context)

# Execute policy
tour, cost, _ = policy.execute(
    packing=packing,
    bins=context.bins,
    distance_matrix=context.distance_matrix
)
```

---

## 10. route improvement & Refinement

**Directory**: `other/route_improvement/`
**Purpose**: Automatic route improvement after initial solution

### 10.1 RouteImproverFactory

**File**: `factory.py`

```python
from logic.src.policies.helpers import RouteImproverFactory

# Create post-processor
post_processor = RouteImproverFactory.create_processor("fast_tsp")

# Apply to tour
improved_tour = post_processor.process(
    tour=[0, 5, 3, 8, 2, 0],
    distance_matrix=dist_matrix
)
```

### 10.2 route improvement Methods

#### Fast TSP

**File**: `fast_tsp.py`

Fast TSP heuristic using `fast_tsp` library.

```python
from logic.src.policies.helpers.route_improvement import FastTSPRouteImprover

processor = FastTSPRouteImprover()

# Improve single-route tour
improved_tour = processor.process(
    tour=[0, 5, 3, 8, 2, 7, 0],
    distance_matrix=dist_matrix
)
```

#### Classical Local Search

**File**: `local_search.py`

2-opt and 3-opt intra-route improvements.

```python
from logic.src.policies.helpers.route_improvement import ClassicalLocalSearchRouteImprover

processor = ClassicalLocalSearchRouteImprover(
    operators=["2opt", "3opt"],
    max_iterations=100
)

improved_tour = processor.process(
    tour=[0, 1, 2, 3, 4, 5, 0],
    distance_matrix=dist_matrix
)
```

#### Randomized Local Search

**File**: `random_ls.py`

Randomized operator selection.

```python
from logic.src.policies.helpers.route_improvement import RandomLocalSearchRouteImprover

processor = RandomLocalSearchRouteImprover(
    operator_probs={
        "2opt": 0.4,
        "swap": 0.3,
        "relocate": 0.3
    },
    max_iterations=50
)

improved_tour = processor.process(
    tour=[0, 1, 2, 3, 0],
    distance_matrix=dist_matrix
)
```

#### Iterated Local Search

**File**: `ils.py`

ILS with perturbation for escaping local minima.

```python
from logic.src.policies.helpers.route_improvement import IteratedLocalSearchRouteImprover

processor = IteratedLocalSearchRouteImprover(
    n_restarts=5,
    ls_iterations=50,
    perturbation_strength=0.2
)

improved_tour = processor.process(
    tour=[0, 1, 2, 3, 4, 5, 0],
    distance_matrix=dist_matrix
)
```

#### Path Improvement

**File**: `path.py`

Specialized path optimization.

```python
from logic.src.policies.helpers.route_improvement import PathRouteImprover

processor = PathRouteImprover()

improved_tour = processor.process(
    tour=[0, 1, 2, 3, 0],
    distance_matrix=dist_matrix
)
```

### 10.3 Pipeline Configuration

```python
from logic.src.policies.helpers import RouteImproverFactory

# Multi-stage refinement pipeline
processors = [
    RouteImproverFactory.create_processor("fast_tsp"),
    RouteImproverFactory.create_processor("classical_ls"),
    RouteImproverFactory.create_processor("ils")
]

tour = initial_tour
for processor in processors:
    tour = processor.process(tour, distance_matrix)

print(f"Final tour cost: {compute_cost(tour, distance_matrix)}")
```

---

## 11. Local Search

**Directory**: `local_search/`
**Purpose**: Base classes for local search implementations

### 11.1 LocalSearch Base

**File**: `local_search_base.py`

```python
from logic.src.policies.local_search import LocalSearch

class CustomLocalSearch(LocalSearch):
    """Custom local search implementation."""

    def search(self, solution, distance_matrix, **kwargs):
        """Perform local search."""
        improved = True
        while improved:
            improved = False
            for operator in self.operators:
                new_solution = operator.apply(solution)
                if self.is_better(new_solution, solution):
                    solution = new_solution
                    improved = True
        return solution
```

### 11.2 HGS Local Search

**File**: `local_search_hgs.py`

```python
from logic.src.policies.local_search import HGSLocalSearch

ls = HGSLocalSearch(
    operators=["relocate", "swap", "2opt", "2opt_star"],
    max_iterations=100
)

improved_routes = ls.search(
    routes=[[0, 1, 2, 0], [0, 3, 4, 0]],
    distance_matrix=dist_matrix
)
```

### 11.3 ACO Local Search

**File**: `local_search_aco.py`

```python
from logic.src.policies.local_search import ACOLocalSearch

ls = ACOLocalSearch(
    operators=["2opt", "3opt"],
    max_iterations=50
)

improved_routes = ls.search(
    routes=[[0, 1, 2, 3, 0]],
    distance_matrix=dist_matrix
)
```

---

## 12. Integration Examples

### 12.1 Complete Simulation Pipeline

```python
from logic.src.policies import create_policy
from logic.src.policies.helpers import (
    PackingFactory,
    RouteImproverFactory,
    SelectionContext
)

# 1. Create mandatory selector
selector = PackingFactory.create_selector(
    strategy="combined",
    logic="or",
    strategies=[
        {"strategy": "last_minute", "threshold": 0.9},
        {"strategy": "regular", "frequency": 3}
    ]
)

# 2. Create policy
policy = create_policy("hgs")

# 3. Create post-processor
post_processor = RouteImproverFactory.create_processor("fast_tsp")

# 4. Execute simulation day
context = SelectionContext(
    bins=bins_state,
    distance_matrix=dist_matrix,
    area="riomaior",
    waste_type="plastic",
    day=5
)

# 5. Select mandatory bins
packing = selector.select(context)

# 6. Generate tour with policy
tour, cost, _ = policy.execute(
    packing=packing,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config=config_dict
)

# 7. Post-process tour
improved_tour = post_processor.process(tour, dist_matrix)
final_cost = compute_cost(improved_tour, dist_matrix)

print(f"Initial cost: {cost:.2f}")
print(f"Final cost: {final_cost:.2f}")
print(f"Improvement: {(cost - final_cost) / cost * 100:.1f}%")
```

### 12.2 Multi-Policy Comparison

```python
from logic.src.policies import create_policy

policies = ["hgs", "alns", "bpc", "neural"]
results = {}

for policy_name in policies:
    policy = create_policy(policy_name)

    tour, cost, metadata = policy.execute(
        packing=packing_bins,
        bins=bins_state,
        distance_matrix=dist_matrix,
        config=config
    )

    results[policy_name] = {
        "tour": tour,
        "cost": cost,
        "metadata": metadata
    }

# Find best
best_policy = min(results, key=lambda p: results[p]["cost"])
print(f"Best policy: {best_policy} with cost {results[best_policy]['cost']:.2f}")
```

### 12.3 Custom Policy Implementation

```python
from logic.src.policies.adapters import BaseRoutingPolicy, RoutingAlgorithmRegistry

@RoutingAlgorithmRegistry.register("my_custom_policy")
class MyCustomPolicy(BaseRoutingPolicy):
    """Custom routing policy implementation."""

    def _get_config_key(self) -> str:
        return "my_custom"

    def _run_solver(
        self,
        sub_dist_matrix,
        sub_wastes,
        capacity,
        revenue,
        cost_unit,
        values,
        **kwargs
    ):
        """Implement custom solver logic."""
        # 1. Initialize solution
        routes = self._initialize_routes(sub_dist_matrix, sub_wastes, capacity)

        # 2. Improve with local search
        for iteration in range(values.get("max_iterations", 100)):
            routes = self._improve_routes(routes, sub_dist_matrix)

        # 3. Compute cost
        cost = self._compute_routes_cost(routes, sub_dist_matrix)

        return routes, cost

    def _initialize_routes(self, dist_matrix, wastes, capacity):
        """Create initial solution."""
        # Custom initialization logic
        return [[1, 2], [3, 4]]

    def _improve_routes(self, routes, dist_matrix):
        """Apply improvement operators."""
        # Custom improvement logic
        return routes

    def _compute_routes_cost(self, routes, dist_matrix):
        """Calculate total cost."""
        # Custom cost calculation
        return 0.0

# Usage
policy = create_policy("my_custom_policy")
tour, cost, _ = policy.execute(
    packing=packing_bins,
    bins=bins_state,
    distance_matrix=dist_matrix
)
```

### 12.4 Operator Composition

```python
from logic.src.policies.operators import (
    random_removal,
    worst_removal,
    greedy_insertion,
    regret_k_insertion
)

def custom_alns_iteration(routes, dist_matrix, wastes, capacity):
    """Custom ALNS iteration with specific operators."""
    # Destroy phase: alternate between random and worst removal
    if iteration % 2 == 0:
        removed, routes = random_removal(routes, num_remove=5)
    else:
        removed, routes = worst_removal(routes, dist_matrix, num_remove=5)

    # Repair phase: alternate between greedy and regret
    if len(removed) < 3:
        routes = greedy_insertion(routes, removed, dist_matrix, wastes, capacity)
    else:
        routes = regret_k_insertion(routes, removed, dist_matrix, wastes, capacity, k=2)

    return routes

# Use in optimization loop
routes = initial_routes
for iteration in range(100):
    new_routes = custom_alns_iteration(routes, dist_matrix, wastes, capacity)
    if compute_cost(new_routes) < compute_cost(routes):
        routes = new_routes
```

---

## 13. Best Practices

### 13.1 Recommended Practices

**Use PolicyFactory for Instantiation**

```python
# ✅ GOOD: Dynamic policy creation
policy = PolicyFactory.get_adapter("hgs")

# ❌ BAD: Direct import and instantiation
from logic.src.policies.adapters.policy_hgs import HGSPolicy
policy = HGSPolicy()
```

**Register Custom Policies**

```python
# ✅ GOOD: Register for discoverability
@RoutingAlgorithmRegistry.register("my_policy")
class MyPolicy(IPolicyAdapter):
    ...

# ❌ BAD: Unregistered policies
class MyPolicy(IPolicyAdapter):
    ...
```

**Use mandatory Selectors**

```python
# ✅ GOOD: Configurable selection strategy
selector = PackingFactory.create_selector("last_minute", threshold=0.9)
packing = selector.select(context)

# ❌ BAD: Hardcoded selection logic
packing = [i for i, fill in enumerate(bins.c) if fill > 0.9]
```

**Apply route improvement**

```python
# ✅ GOOD: Refine solutions
tour, cost, _ = policy.execute(...)
improved_tour = post_processor.process(tour, dist_matrix)

# ❌ BAD: Use raw policy output
tour, cost, _ = policy.execute(...)
# No improvement applied
```

**Handle Engine Availability**

```python
# ✅ GOOD: Graceful fallback
try:
    policy = PolicyFactory.get_adapter("bpc")
    tour, cost, _ = policy.execute(..., config={"bpc": {"engine": "gurobi"}})
except ImportError:
    print("Gurobi not available, falling back to OR-Tools")
    tour, cost, _ = policy.execute(..., config={"bpc": {"engine": "ortools"}})

# ❌ BAD: Assume engine exists
policy = PolicyFactory.get_adapter("bpc")
tour, cost, _ = policy.execute(..., config={"bpc": {"engine": "gurobi"}})
# Crashes if Gurobi not installed
```

### 13.2 Anti-Patterns

**Don't Modify Operators In-Place**

```python
# ❌ BAD: Modifying input routes
def bad_operator(routes):
    routes[0].append(5)  # Mutates input!
    return routes

# ✅ GOOD: Return new routes
def good_operator(routes):
    new_routes = [route.copy() for route in routes]
    new_routes[0].append(5)
    return new_routes
```

**Don't Skip Validation**

```python
# ❌ BAD: No input validation
def solve(routes, dist_matrix):
    # Assumes routes is valid
    return optimize(routes)

# ✅ GOOD: Validate inputs
def solve(routes, dist_matrix):
    if not routes:
        return [0, 0], 0.0
    if len(dist_matrix) == 0:
        raise ValueError("Empty distance matrix")
    return optimize(routes)
```

**Don't Ignore Capacity Constraints**

```python
# ❌ BAD: Insert without capacity check
def insert_node(route, node, waste):
    route.insert(1, node)

# ✅ GOOD: Check capacity
def insert_node(route, node, waste, capacity, wastes):
    current_load = sum(wastes[n] for n in route if n != 0)
    if current_load + waste <= capacity:
        route.insert(1, node)
        return True
    return False
```

### 13.3 Performance Optimization

**Batch Processing**

```python
# ✅ GOOD: Batch process for neural policies
from logic.src.policies import NeuralAgent

agent = NeuralAgent(model)
tours, costs = agent.solve_batch(
    batch_coords=batch_coords,
    batch_wastes=batch_wastes
)

# ❌ BAD: Loop over instances
tours, costs = [], []
for coords, wastes in zip(batch_coords, batch_wastes):
    tour, cost = agent.solve(coords, wastes)
    tours.append(tour)
    costs.append(cost)
```

**Operator Selection**

```python
# ✅ GOOD: Use fast operators for large problems
if num_nodes > 200:
    operators = ["relocate", "swap", "2opt"]  # O(n²)
else:
    operators = ["relocate", "swap", "2opt", "3opt", "or_opt"]  # Include O(n³)

# ❌ BAD: Always use expensive operators
operators = ["3opt", "ejection_chain"]  # Very slow on large instances
```

**Caching**

```python
# ✅ GOOD: Cache distance calculations
@lru_cache(maxsize=10000)
def get_distance(i, j, dist_matrix):
    return dist_matrix[i][j]

# ❌ BAD: Repeated lookups
for i in range(n):
    for j in range(n):
        d = dist_matrix[i][j]  # No caching
```

---

## 14. Quick Reference

### 14.1 Common Imports

```python
# Policy creation
from logic.src.policies.adapters import PolicyFactory, RoutingAlgorithmRegistry

# Main policies
from logic.src.policies import (
    run_alns, run_hgs, run_bpc,
    run_k_sparse_aco, run_hyper_heuristic_aco,
    NeuralAgent
)

# Operators
from logic.src.policies.operators import (
    # Move
    move_relocate, move_swap,
    # Route
    move_2opt_intra, move_2opt_star, move_3opt_intra,
    # Destroy
    random_removal, worst_removal, cluster_removal,
    # Repair
    greedy_insertion, regret_k_insertion
)

# packing
from logic.src.policies.helpers import (
    PackingFactory, PackingRegistry,
    SelectionContext
)

# route improvement
from logic.src.policies.helpers import (
    RouteImproverFactory, RouteImproverRegistry
)

# TSP/CVRP
from logic.src.policies import find_route, find_routes
```

### 14.2 Policy Summary

| Policy        | Type          | Best For                             | Typical Runtime |
| ------------- | ------------- | ------------------------------------ | --------------- |
| **BPC**       | Exact         | Small instances (<50 nodes), optimal | Minutes-Hours   |
| **HGS**       | Metaheuristic | Medium-large instances, high quality | Seconds-Minutes |
| **ALNS**      | Metaheuristic | Diverse instances, robustness        | Seconds-Minutes |
| **HGS-ALNS**  | Hybrid        | Large instances, best quality        | Minutes         |
| **ACO**       | Bio-inspired  | Sparse graphs, parallel execution    | Minutes         |
| **SANS**      | SA-based      | Complex constraints                  | Seconds-Minutes |
| **SISR**      | Specialized   | String patterns in routes            | Seconds         |
| **Neural**    | Deep RL       | Fast inference, generalization       | Milliseconds    |
| **LKH-3 LNS** | Matheuristic  | CVRP/VRPP/TSP high quality           | Seconds         |
| **VRPP**      | Exact         | Profit-oriented VRP                  | Minutes         |

### 14.3 Operator Complexity

| Operator         | Time Complexity | Space Complexity | Use Case                     |
| ---------------- | --------------- | ---------------- | ---------------------------- |
| Relocate         | O(n²)           | O(1)             | General improvement          |
| Swap             | O(n²)           | O(1)             | General improvement          |
| 2-Opt            | O(n²)           | O(1)             | Intra-route improvement      |
| 3-Opt            | O(n³)           | O(1)             | Deep intra-route improvement |
| 2-Opt\*          | O(n²)           | O(1)             | Inter-route improvement      |
| Or-Opt           | O(n²)           | O(1)             | Chain moves                  |
| Cross-Exchange   | O(n⁴)           | O(1)             | Complex inter-route          |
| Ejection Chain   | O(n^k)          | O(k)             | Deep exploration             |
| Random Removal   | O(n)            | O(n)             | ALNS destroy                 |
| Worst Removal    | O(n²)           | O(n)             | ALNS destroy (quality)       |
| Greedy Insertion | O(n²)           | O(n)             | ALNS repair                  |
| Regret-k         | O(kn²)          | O(kn)            | ALNS repair (quality)        |

### 14.4 Configuration Examples

**ALNS Configuration**

```yaml
alns:
  time_limit: 60.0
  max_iterations: 5000
  start_temp: 100.0
  cooling_rate: 0.995
  reaction_factor: 0.1
  min_removal: 1
  max_removal_pct: 0.3
  engine: custom
```

**HGS Configuration**

```yaml
hgs:
  time_limit: 120.0
  population_size: 50
  elite_size: 10
  mutation_rate: 0.2
  crossover_type: ox
  local_search_iterations: 100
```

**Neural Configuration**

```yaml
neural:
  model_path: model_weights/am_wcvrp_100.pt
  decode_type: beam_search
  beam_width: 10
  temperature: 1.0
  device: cuda
```

### 14.5 File Locations

| Component         | Location                                      |
| ----------------- | --------------------------------------------- |
| Policy Adapters   | `adapters/policy_*.py` (13 files)             |
| Operators         | `operators/` (26 implementations)             |
| mandatory Selectors | `other/packing/selection_*.py` (6 strategies) |
| Post-Processors   | `other/route_improvement/*.py` (5 methods)      |
| ALNS              | `adaptive_large_neighborhood_search/`         |
| HGS               | `hybrid_genetic_search/`                      |
| ACO               | `ant_colony_optimization/`                    |
| SANS              | `simulated_annealing_neighborhood_search/`    |
| SISR              | `slack_induction_by_string_removal/`          |
| BPC               | `branch_and_price_and_cut/`                   |
| Neural            | `neural_agent/`                               |

### 14.6 Related Documentation

- [CONFIGS_MODULE.md](CONFIGS_MODULE.md) - Policy configuration dataclasses
- [MODELS_MODULE.md](MODELS_MODULE.md) - Neural network architectures
- [ENVS_MODULE.md](ENVS_MODULE.md) - Problem environments
- [CLAUDE.md](../CLAUDE.md) - Coding standards and guidelines

---

**Last Updated**: January 2026
**Maintainer**: WSmart+ Route Development Team
**Status**: ✅ Active - Comprehensive routing policies library
