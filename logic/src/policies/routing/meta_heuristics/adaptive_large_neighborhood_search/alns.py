r"""Adaptive Large Neighborhood Search (ALNS) for VRPP.

This module implements ALNS following Ropke & Pisinger (2005) methodology,
adapted for the Vehicle Routing Problem with Profits (VRPP) as a profit
maximization problem. The solver builds on the shared
:class:`~logic.src.policies.routing.meta_heuristics.large_neighborhood_search.engine.LNSBase`
loop skeleton (WP-11 P-31); ALNS-specific semantics (segment-based weights,
visited-solution tracking, shallow-copy policy, Ropke & Pisinger removal
bounds, factory-built acceptance criterion) live in the hooks below.

Attributes:
    ALNSSolver: Core ALNS implementation.

Example:
    >>> from logic.src.policies.routing.base.sub_problem import build_subproblem
    >>> from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns import ALNSSolver
    >>> problem = build_subproblem(dist_matrix, wastes, 100.0, 1.5, 0.5)
    >>> solver = ALNSSolver(problem, params)
Key Implementation Features:
============================

1. **Operator Weight Updates** (Section 3.4.4):
   - Segment-based learning: w_{i,j+1} = w_{i,j}(1-r) + r * (π_i / θ_i)
   - Reaction factor r controls adaptation speed (default: 0.1)
   - Weights updated every `segment_size` iterations (default: 100)

2. **Scoring Mechanism** (Section 3.3):
   - σ₁: New global best solution found (default: 33)
   - σ₂: Better solution not visited before (default: 9)
   - σ₃: Accepted worse solution not visited before (default: 13)
   - Hash table tracks visited solutions to prevent rewarding revisits

3. **Simulated Annealing Acceptance**:
   - For VRPP profit maximization: P(accept) = exp(-Δ/T) where Δ = current_profit - new_profit.
   - This inversion (current - new) correctly handles profit maximization where
     solutions with Δ <= 0 (new >= current) are always accepted.

4. **Randomized Worst Removal** (Section 3.4.1):
   - Index selection: floor(y^p * |L|) where y ~ U(0,1) and p >= 1
   - Higher p values bias toward deterministically worst nodes
   - Default p = 3.0 provides good exploration/exploitation balance

5. **Adaptive Noise in Repair** (Section 3.4.3 / Configuration 15):
   - Applied additively to final insertion cost: C' = max{0, C + noise}
   - Noise ~ U[-η * max_dist, η * max_dist] where η (default: 0.025)
   - Clean and noisy variants are **separate operator slots** in the weight
     mechanism (paper Conf. 15), so the algorithm learns which is more
     effective rather than using a fixed 50% coin flip.
   - Operator naming: "<Name>Clean" / "<Name>Noisy"

6. **Operator Registry** (Section 3.3):
   - Destroy: random, worst, shaw [+ string, cluster, neighbor if extended_operators=True]
   - Repair: greedy × {clean,noisy}, regret-k × {clean,noisy} for k in regret_pool
   - Roulette-wheel selection (Eq. 20): p_j = w_j / Σ w_i

7. **Profit-Aware Operators** (Novel Contribution):
   - Speculative Seeding: seed_hurdle = -0.5 * detour_cost
     Allows initially unprofitable routes that may become profitable
   - Profit-based removal: targets nodes with lowest marginal profit
   - Enable via `profit_aware_operators=True` for ablation studies

Performance Optimizations:
==========================
- Shallow copy via list comprehension (replaces deepcopy): ~10x faster
- Visited solutions tracked via canonical tuple hashing
- Incremental route evaluation during operator application
- _viz_record pre-assigned as no-op when recorder=None (avoids getattr
  in hot loop)

Usage for Ablation Studies:
============================
Toggle `profit_aware_operators` in ALNSConfig to compare:
- Standard ALNS (False): cost-minimization operators
- Profit-aware ALNS (True): revenue-cost maximization operators

Toggle `extended_operators` in ALNSConfig to compare:
- Core destroy set (False): random, worst, shaw  [3 operators]
- Extended destroy set (True): + string, cluster, neighbor  [6 operators]

References:
===========
[1] Ropke, S., & Pisinger, D. (2006). "An adaptive large neighborhood search
    heuristic for the pickup and delivery problem with time windows."
    Transportation Science, 40(4), 455-472.

The class is split into mixins: the destroy and repair operator registry builders live
in ``alns_operator_registry.py`` (:class:`ALNSOperatorRegistryMixin`), the LNSBase hooks
and core loop helpers in ``alns_hooks.py`` (:class:`ALNSHooksMixin`).
"""

import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.policies.helpers.operators import (
    build_greedy_routes,
)
from logic.src.policies.routing.base.sub_problem import SubProblem, coerce_config
from logic.src.policies.routing.meta_heuristics.large_neighborhood_search.engine import LNSBase
from logic.src.policies.routing.meta_heuristics.large_neighborhood_search.weight_updates import (
    RopkePisingerSegments,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .alns_hooks import ALNSHooksMixin
from .alns_operator_registry import ALNSOperatorRegistryMixin

__all__ = ["ALNSSolver"]


class ALNSSolver(ALNSOperatorRegistryMixin, ALNSHooksMixin, LNSBase):
    r"""Custom implementation of Adaptive Large Neighborhood Search for VRPP.

    Follows Ropke & Pisinger (2005) with segment-based weight updates,
    clean/noisy operator pairs (paper Configuration 15), and optional
    extended destroy operator registry. Built on the shared
    :class:`LNSBase` loop skeleton.

    Attributes:
        problem: The SubProblem this solver was built for.
        dist_matrix: Distance matrix between nodes.
        wastes: Dictionary mapping node indices to waste amounts.
        capacity: Vehicle capacity.
        R: Revenue per unit collected.
        C: Cost per unit distance.
        params: ALNS parameters.
        cfg: ALNSConfig backing ``params``.
        mandatory_nodes: List of nodes that must be visited.
        profit_aware_operators: Whether to use profit-aware operators.
        extended_operators: Whether to use extended destroy set.
        random: Random number generator for selection.
        np_random: NumPy generator for worst removal.
        _viz_record: Callable for recording solver state.
        n_nodes: Number of customer nodes.
        nodes: List of customer node indices.
        destroy_ops: List of destroy operator functions.
        destroy_names: Names of destroy operators.
        repair_ops: List of repair operator functions.
        repair_names: Names of repair operators.
        repair_noisy: Boolean flags for noisy repair operators.
        segment_size: Iterations per weight update segment.
        destroy_weights: Adaptive weights for destroy operators.
        repair_weights: Adaptive weights for repair operators.
        destroy_scores: Accumulated scores for destroy operators.
        repair_scores: Accumulated scores for repair operators.
        destroy_counts: Usage counts for destroy operators.
        repair_counts: Usage counts for repair operators.
        weight_decay: Weight decay factor (1 - reaction_factor).
        weight_learning_rate: Weight learning rate (reaction_factor).
        visited_solutions: Set of hashed solution states.
        acceptance_criterion: Logic for accepting candidate solutions.

    Example:
        >>> problem = build_subproblem(dist_matrix, wastes, 100.0, 1.5, 0.5)
        >>> solver = ALNSSolver(problem, params)
        >>> best_routes, best_profit, best_cost = solver.solve()
    """

    def __init__(
        self,
        problem: SubProblem,
        cfg: Optional[Union[ALNSConfig, Dict[str, Any]]] = None,
        recorder: Optional[PolicyStateRecorder] = None,
    ):
        """
        Initializes the ALNS solver.

        Args:
            problem: The routing sub-problem (distance matrix, wastes,
                capacity, revenue, cost unit, mandatory nodes).
            cfg: ALNS configuration dataclass, dictionary, or arbitrary object
                (filtered leniently to ALNSConfig fields).
            recorder: Optional recorder for solver state; falls back to
                ``problem.recorder``.
        """
        self.problem = problem
        self.dist_matrix = problem.dist_matrix
        self.wastes = problem.wastes
        self.capacity = problem.capacity
        self.R = problem.revenue
        self.C = problem.cost_unit

        config = coerce_config(cfg, ALNSConfig)
        self.cfg: ALNSConfig = config
        self.params = config
        self.mandatory_nodes = problem.mandatory_nodes
        self.profit_aware_operators = getattr(config, "profit_aware_operators", False)
        self.extended_operators = getattr(config, "extended_operators", False)
        self.random = random.Random(config.seed) if config.seed is not None else random.Random()
        self.np_random = np.random.default_rng(config.seed) if config.seed is not None else np.random.default_rng()

        # Pre-assign no-op to avoid getattr in the hot iteration loop.
        effective_recorder = recorder if recorder is not None else problem.recorder
        self._viz_record: Callable = effective_recorder.record if effective_recorder is not None else lambda **k: None

        self.n_nodes = len(self.dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # ------------------------------------------------------------------
        # Destroy operator registry
        # ------------------------------------------------------------------
        self.destroy_ops, self.destroy_names = self._build_destroy_ops()

        # ------------------------------------------------------------------
        # Repair operator registry — Section 3.6 / Conf. 15:
        # Clean and noisy variants are separate operator slots so that the
        # adaptive weight mechanism can learn which is more effective.
        # Slots are interleaved: [Op0_clean, Op0_noisy, Op1_clean, Op1_noisy, ...]
        # ------------------------------------------------------------------
        self.repair_ops, self.repair_names, self.repair_noisy = self._build_repair_ops()

        # ------------------------------------------------------------------
        # Segment-based weight update (Ropke & Pisinger 2005, Section 3.4.4)
        # ------------------------------------------------------------------
        self.segment_size = config.segment_size
        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)
        self._weights = RopkePisingerSegments(self.destroy_weights, self.repair_weights, config.reaction_factor)
        # Alias the strategy accumulators so the historic attribute surface
        # (destroy_scores/repair_scores/destroy_counts/repair_counts) is kept.
        self.destroy_scores = self._weights.destroy_scores
        self.repair_scores = self._weights.repair_scores
        self.destroy_counts = self._weights.destroy_counts
        self.repair_counts = self._weights.repair_counts
        self.weight_decay = self._weights.weight_decay
        self.weight_learning_rate = self._weights.weight_learning_rate

        # Visited solutions tracking (Ropke & Pisinger 2005, Section 3.3)
        self.visited_solutions: Set[Tuple[Tuple[int, ...], ...]] = set()

        # Acceptance Criterion Injection - fresh per-solve instance (P-11)
        from logic.src.policies.acceptance_criteria.base.factory import (
            AcceptanceCriterionFactory,
        )

        self.acceptance_criterion = AcceptanceCriterionFactory.build(self.cfg.acceptance_criterion)

    # Removed _calculate_dynamic_start_temp in favor of criterion.setup()

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def build_initial_solution(self) -> List[List[int]]:
        """Build an initial solution using the greedy profit-aware heuristic.

        Returns:
            List of routes for the initial solution.
        """
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )
