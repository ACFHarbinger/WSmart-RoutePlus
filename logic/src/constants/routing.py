"""
Constants for routing algorithms.

This module defines operational parameters, solver settings, and penalty values
for classical routing algorithms. Used by:
- logic/src/policies/ (HGS, ALNS, local search operators)
- logic/src/configs/policies/ (Gurobi, BPC configurations)

Parameter Categories
--------------------
1. **Physical Constraints**: Real-world operational parameters (speeds, times)
2. **Local Search Settings**: Epsilon thresholds for move acceptance
3. **Exact Solver Tuning**: Gurobi MIP parameters
4. **Penalty Values**: Constraint violation costs

Units and Valid Ranges
-----------------------
- Time: minutes (collection) or seconds (optimization timeout)
- Speed: km/h (kilometers per hour)
- Capacity: percent (0-100)
- Penalties: cost units (dimensionless, scaled to route cost)
- MIP gap: ratio (0.0-1.0, where 0.01 = 1% optimality gap)

Attributes:
    IMPROVEMENT_EPSILON: Minimum cost improvement to accept a local search move
    COLLECTION_TIME_MINUTES: Average time to collect one bin (minutes)
    VEHICLE_SPEED_KMH: Average vehicle speed in urban waste collection (km/h)
    PENALTY_MANDATORY_NODES_MISSED: Penalty for missing a mandatory bin
    MAX_WASTE: Maximum bin fill level (1.0 = full capacity)
    MIP_GAP: MIP optimality gap tolerance (ratio)
    HEURISTICS_RATIO: Heuristic effort ratio (0.0-1.0)
    NODEFILE_START_GB: Memory threshold for writing nodes to disk (GB)
    SOLVER_OUTPUT_FLAG: Solver output flag (0 = suppress, 1 = enable)
    DEFAULT_SHIFT_DURATION: Default shift duration (minutes)
    DEFAULT_V_VALUE: Default value function parameter (dimensionless)
    DEFAULT_COMBINATION: Default OG SANS hyperparameter combination
    DEFAULT_TIME_LIMIT: Default optimization time limit (seconds)
    DEFAULT_EVAL_BATCH_SIZE: Default evaluation batch size for model inference
    DEFAULT_ROLLOUT_BATCH_SIZE: Default rollout batch size for baseline computation
    SCALE: Scale factor for TSP and CVRP distance matrices
    DEFAULT_HIGH_PENALTY: Default penalty for inter-dummy edges (prevents empty intermediate routes)

Example:
    >>> from logic.src.constants import IMPROVEMENT_EPSILON, COLLECTION_TIME_MINUTES, VEHICLE_SPEED_KMH, PENALTY_MANDATORY_NODES_MISSED, MAX_WASTE, MIP_GAP, HEURISTICS_RATIO, NODEFILE_START_GB, SOLVER_OUTPUT_FLAG, DEFAULT_SHIFT_DURATION, DEFAULT_V_VALUE, DEFAULT_COMBINATION, DEFAULT_TIME_LIMIT, DEFAULT_EVAL_BATCH_SIZE, DEFAULT_ROLLOUT_BATCH_SIZE, SCALE, DEFAULT_HIGH_PENALTY
    >>> IMPROVEMENT_EPSILON
    1e-3
    >>> COLLECTION_TIME_MINUTES
    3.0
    >>> VEHICLE_SPEED_KMH
    40.0
    >>> PENALTY_MANDATORY_NODES_MISSED
    10000.0
    >>> MAX_WASTE
    1.0
    >>> MIP_GAP
    0.01
    >>> HEURISTICS_RATIO
    0.5
    >>> NODEFILE_START_GB
    0.5
    >>> SOLVER_OUTPUT_FLAG
    0
    >>> DEFAULT_SHIFT_DURATION
    390
    >>> DEFAULT_V_VALUE
    1.0
    >>> DEFAULT_COMBINATION
    [500, 75, 0.95, 0, 0.095, 0, 0]
    >>> DEFAULT_TIME_LIMIT
    600
    >>> DEFAULT_EVAL_BATCH_SIZE
    1024
    >>> DEFAULT_ROLLOUT_BATCH_SIZE
    64
    >>> SCALE
    10000
    >>> DEFAULT_HIGH_PENALTY
    1e9
"""

# Local Search Improvement Threshold
# Minimum cost improvement (in cost units) to accept a local search move.
# Prevents cycling on near-equal solutions. Used in: 2-opt, swap, relocate operators.
# Range: [1e-6, 1e-2]. Smaller = more thorough (slower), larger = faster (may miss improvements).
IMPROVEMENT_EPSILON: float = 1e-3  # Default: 0.1% of typical route cost

# Collection Parameters - Real-World Operational Constraints
# -----------------------------------------------------------
# Used in: Time window calculations, route duration estimation, capacity planning

# Average time to collect one bin (minutes)
# Includes: approach, emptying, compaction, departure
# Typical real-world values: 2-5 minutes depending on bin type
COLLECTION_TIME_MINUTES = 3.0  # minutes per bin

# Average vehicle speed in urban waste collection (km/h)
# Includes: traffic, turns, narrow streets, residential speed limits
# Typical real-world values: 30-50 km/h (not highway speed)
VEHICLE_SPEED_KMH = 40.0  # km/h (urban residential average)

# Optimization Penalties - Constraint Violation Costs
# ----------------------------------------------------
# Used in: Penalty-based MIP formulations, constraint relaxation

# Penalty for missing a mandatory bin (bins flagged for mandatory collection)
# Should be >> typical route cost to ensure mandatory bins are prioritized.
# Typical route cost: 50-200 units. Penalty ensures mandatory bins are never skipped.
PENALTY_MANDATORY_NODES_MISSED = 10000.0  # cost units (high penalty)

# Capacity Constraints
# --------------------
# Maximum bin fill level (1.0 = full capacity)
# Used in: Overflow detection, capacity feasibility checks.
MAX_WASTE: float = 1.0  # [0.0, 1.0] capacity range (bins can exceed this, triggering overflow penalty)

# Gurobi MIP Solver Parameters
# -----------------------------
# Tuned for VRP-class problems. Trade-off: solution quality vs runtime.
# Used in: logic/src/policies/adapters/policy_bpc.py, policy_swc_tcf.py

# MIP optimality gap tolerance (ratio)
# Solver stops when: (best_bound - incumbent) / incumbent ≤ MIP_GAP
# 0.01 = 1% gap (industry standard for VRP)
# Smaller gap = longer runtime, better solution quality
MIP_GAP = 0.01  # 1% optimality gap (solver stops when proven within 1% of optimal)

# Heuristic effort ratio (0.0-1.0)
# Controls how much time Gurobi spends on heuristics vs branch-and-bound.
# 0.5 = balanced (default), 0.0 = no heuristics, 1.0 = maximum heuristics
HEURISTICS_RATIO = 0.5  # balanced heuristic vs exact search

# Node file threshold (GB)
# When memory usage exceeds this, Gurobi writes nodes to disk (slower).
# 0.5 GB is conservative; increase for large instances on high-RAM machines.
NODEFILE_START_GB = 0.5  # GB (start disk-based node storage)

# Solver Output Flag
# 0 = suppress solver output, 1 = enable solver output
SOLVER_OUTPUT_FLAG = 0  # for Gurobi solver

# Simulated Annealing Neighborhood Search Constants
# -------------------------------------------------
# Used in: logic/src/policies/adapters/policy_sans.py

# Default shift duration (minutes)
# Typical waste collection shift: 6-8 hours = 360-480 minutes
DEFAULT_SHIFT_DURATION = 390  # minutes (6.5 hours)

# Default value function parameter (dimensionless)
# Controls tradeoff between collection cost and overflow risk in value function
DEFAULT_V_VALUE = 1.0  # weight (1.0 = equal weight to cost and risk)

# Default OG SANS hyperparameter combination
# Format: [max_iter, beam_width, cooling_rate, init_temp, pert_strength, ...]
# These are algorithm-specific tuning parameters. See SANS paper for details.
DEFAULT_COMBINATION = [500, 75, 0.95, 0, 0.095, 0, 0]  # OG SANS hyperparameters (7-tuple)

# Default OG SANS time limit (seconds)
# Total optimization time before returning best solution found
DEFAULT_TIME_LIMIT = 600  # seconds (10 minutes)

# Batch Size Defaults
# --------------------
# Default batch sizes for evaluation and rollout operations.
# Tuned for 12GB+ GPU memory. Reduce for smaller GPUs.

# Default evaluation batch size for model inference.
# Used in: logic/src/models/hrl_manager/manager.py (HRL policy evaluation)
# Larger batches = faster throughput, but higher memory usage.
# 1024 is optimal for 12GB VRAM; reduce to 512 for 8GB GPUs.
DEFAULT_EVAL_BATCH_SIZE: int = 1024  # instances per batch (HRL manager default)

# Default rollout batch size for baseline computation.
# Used in: logic/src/pipeline/learning/common/baselines/rollout.py (REINFORCE/PPO baselines)
# Smaller than eval batch size because rollouts compute full episodes.
# 64 balances memory and baseline quality; increase for more samples.
DEFAULT_ROLLOUT_BATCH_SIZE: int = 64  # episodes per rollout (baseline computation default)

# TSP and CVRP distance matrix scaling factor
SCALE = 10000

# Default penalty for inter-dummy edges (prevents empty intermediate routes)
DEFAULT_HIGH_PENALTY = 1e9
