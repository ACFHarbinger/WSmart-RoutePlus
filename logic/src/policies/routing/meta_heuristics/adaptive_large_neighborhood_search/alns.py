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
"""

import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from logic.src.configs.policies.alns import ALNSConfig
from logic.src.policies.helpers.operators import (
    build_greedy_routes,
    cluster_profit_removal,
    cluster_removal,
    greedy_insertion,
    greedy_profit_insertion,
    neighbor_profit_removal,
    neighbor_removal,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    regret_3_insertion,
    regret_3_profit_insertion,
    regret_4_insertion,
    regret_4_profit_insertion,
    savings_insertion,
    savings_profit_insertion,
    shaw_profit_removal,
    shaw_removal,
    string_profit_removal,
    string_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.routing.base.sub_problem import SubProblem
from logic.src.policies.routing.meta_heuristics.large_neighborhood_search.engine import (
    MIN_REMOVAL_LOWER_BOUND,
    LNSBase,
    coerce_config,
)
from logic.src.policies.routing.meta_heuristics.large_neighborhood_search.weight_updates import (
    RopkePisingerSegments,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder
from logic.src.envs.tasks.common.expand_pool import current_expand_pool

__all__ = ["ALNSSolver", "MIN_REMOVAL_LOWER_BOUND"]


class ALNSSolver(LNSBase):
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
        self.visited_solutions: Set[Tuple[int, ...]] = set()

        # Acceptance Criterion Injection - fresh per-solve instance (P-11)
        from logic.src.policies.acceptance_criteria.base.factory import (
            AcceptanceCriterionFactory,
        )

        acc_val = getattr(self.cfg, "acceptance_criterion", None)
        if hasattr(acc_val, "accept"):
            self.acceptance_criterion = acc_val
        elif acc_val is not None and hasattr(acc_val, "method"):
            self.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acc_val.method,
                config=acc_val.params,
            )
        elif isinstance(acc_val, dict) and "method" in acc_val:
            self.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acc_val["method"],
                config=acc_val.get("params"),
            )
        else:
            self.acceptance_criterion = AcceptanceCriterionFactory.create(
                name="bmc",
                initial_temp=self.cfg.start_temp or 100.0,
                alpha=self.cfg.cooling_rate,
                seed=self.cfg.seed,
            )

    # ------------------------------------------------------------------
    # Operator registry builders
    # ------------------------------------------------------------------

    def _build_destroy_ops(self) -> Tuple[List[Callable], List[str]]:
        """
        Build the destroy operator registry.

        Core set (always included):
            random, worst[_profit], shaw[_profit]

        Extended set (when extended_operators=True):
            + string[_profit], cluster[_profit], neighbor[_profit]

        Returns:
            Tuple of (operator callables, operator names).
        """
        params = self.params
        pa = self.profit_aware_operators

        if pa:
            core: List[Tuple[str, Callable]] = [
                (
                    "RandomRemoval",
                    lambda r, n: random_removal(r, n, rng=self.random),
                ),
                (
                    "WorstProfitRemoval",
                    lambda r, n: worst_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        self.wastes,
                        self.R,
                        self.C,
                        p=params.worst_removal_randomness,
                        rng=self.random,  # type: ignore[arg-type]
                    ),
                ),
                (
                    "ShawProfitRemoval",
                    lambda r, n: shaw_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        R=self.R,
                        C=self.C,
                        randomization_factor=params.shaw_randomization,
                        rng=self.random,
                    ),
                ),
            ]
            extended: List[Tuple[str, Callable]] = [
                (
                    "StringProfitRemoval",
                    lambda r, n: string_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        R=self.R,
                        C=self.C,
                        rng=self.random,
                    ),
                ),
                (
                    "ClusterProfitRemoval",
                    lambda r, n: cluster_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        R=self.R,
                        C=self.C,
                        rng=self.random,
                    ),
                ),
                (
                    "NeighborProfitRemoval",
                    lambda r, n: neighbor_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        R=self.R,
                        C=self.C,
                        rng=self.random,
                    ),
                ),
            ]
        else:
            core = [
                (
                    "RandomRemoval",
                    lambda r, n: random_removal(r, n, rng=self.random),
                ),
                (
                    "WorstRemoval",
                    lambda r, n: worst_removal(
                        r, n, self.dist_matrix, p=params.worst_removal_randomness, rng=self.np_random
                    ),
                ),
                (
                    "ShawRemoval",
                    lambda r, n: shaw_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        randomization_factor=params.shaw_randomization,
                        rng=self.random,
                    ),
                ),
            ]
            extended = [
                (
                    "StringRemoval",
                    lambda r, n: string_removal(r, n, self.dist_matrix, rng=self.random),
                ),
                (
                    "ClusterRemoval",
                    lambda r, n: cluster_removal(r, n, self.dist_matrix, nodes=self.nodes, rng=self.random),
                ),
                (
                    "NeighborRemoval",
                    lambda r, n: neighbor_removal(r, n, self.dist_matrix, rng=self.random),
                ),
            ]

        registry = core + extended if self.extended_operators else core
        ops = [fn for _, fn in registry]
        names = [nm for nm, _ in registry]
        return ops, names

    def _build_repair_ops(self) -> Tuple[List[Callable], List[str], List[bool]]:
        """Build the repair operator registry with clean/noisy operator pairs.

        Noise-capable operators (greedy, regret-k) produce two consecutive slots:
            [<Name>Clean, <Name>Noisy]
        The adaptive weight mechanism tracks them independently, learning
        whether noisy or clean variants perform better (paper Conf. 15).

        Operators that do not accept a noise parameter (e.g. savings) produce
        a single slot only — including them in both slots would be redundant.

        Operator set is controlled by params.regret_pool:
            "regret2"   → greedy, regret-2                        (4 slots)
            "regret234" → greedy, regret-2, regret-3, regret-4    (8 slots, default)
            "regretAll" → greedy, regret-2, regret-3, regret-4,
                          savings                                  (9 slots)

        Returns:
            Tuple of (operator callables, operator names, is_noisy flags).
        """
        params = self.params
        pa = self.profit_aware_operators
        regret_pool: str = getattr(params, "regret_pool", "regret234")

        # ------------------------------------------------------------------
        # Define the logical repair base functions.
        # Each entry: (label, fn, supports_noise)
        #   supports_noise=True  → expanded into (Clean, Noisy) slot pair
        #   supports_noise=False → single slot only
        # ------------------------------------------------------------------
        common_kw: Dict = dict(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=current_expand_pool(),
        )
        profit_kw: Dict = dict(**common_kw, R=self.R, C=self.C)

        if pa:
            base_repair: List[Tuple[str, Callable, bool]] = [
                (
                    "GreedyProfit",
                    lambda r, n, _noise=0.0: greedy_profit_insertion(r, n, noise=_noise, **profit_kw),
                    True,
                ),
                (
                    "Regret2Profit",
                    lambda r, n, _noise=0.0: regret_2_profit_insertion(r, n, noise=_noise, **profit_kw),
                    True,
                ),
            ]
            if regret_pool in ("regret234", "regretAll"):
                base_repair += [
                    (
                        "Regret3Profit",
                        lambda r, n, _noise=0.0: regret_3_profit_insertion(r, n, noise=_noise, **profit_kw),
                        True,
                    ),
                    (
                        "Regret4Profit",
                        lambda r, n, _noise=0.0: regret_4_profit_insertion(r, n, noise=_noise, **profit_kw),
                        True,
                    ),
                ]
            if regret_pool == "regretAll":
                base_repair += [
                    (
                        "SavingsProfit",
                        lambda r, n, _noise=0.0: savings_profit_insertion(r, n, **profit_kw),
                        False,  # savings_profit_insertion has no noise parameter; _noise ignored
                    ),
                ]
        else:
            base_repair = [
                (
                    "Greedy",
                    lambda r, n, _noise=0.0: greedy_insertion(r, n, noise=_noise, **common_kw),
                    True,
                ),
                (
                    "Regret2",
                    lambda r, n, _noise=0.0: regret_2_insertion(r, n, noise=_noise, **common_kw),
                    True,
                ),
            ]
            if regret_pool in ("regret234", "regretAll"):
                base_repair += [
                    (
                        "Regret3",
                        lambda r, n, _noise=0.0: regret_3_insertion(r, n, noise=_noise, **common_kw),
                        True,
                    ),
                    (
                        "Regret4",
                        lambda r, n, _noise=0.0: regret_4_insertion(r, n, noise=_noise, **common_kw),
                        True,
                    ),
                ]
            if regret_pool == "regretAll":
                base_repair += [
                    (
                        "Savings",
                        lambda r, n, _noise=0.0: savings_insertion(r, n, **common_kw),
                        False,  # savings_insertion has no noise parameter; _noise ignored
                    ),
                ]

        # ------------------------------------------------------------------
        # Expand each base heuristic:
        #   supports_noise=True  → [<Name>Clean (even idx), <Name>Noisy (odd idx)]
        #   supports_noise=False → [<Name>] (single slot, always clean)
        #
        # All slots are called as repair_op(routes, removed, noise_val) from
        # _select_and_apply_operators, so every lambda accepts a 3rd positional
        # _noise argument (even if it is discarded for clean / no-noise ops).
        # ------------------------------------------------------------------
        ops: List[Callable] = []
        names: List[str] = []
        noisy_flags: List[bool] = []

        for label, fn, supports_noise in base_repair:
            if supports_noise:
                # Clean slot: always passes 0.0; the _noise arg (from the call site) is discarded.
                # Using positional args avoids any keyword-name mismatch across base lambdas.
                ops.append(lambda r, n, _noise=0.0, _fn=fn: _fn(r, n, 0.0))
                names.append(f"{label}Clean")
                noisy_flags.append(False)
                # Noisy slot: forwards _noise (the noise value computed by _get_noise) positionally.
                ops.append(lambda r, n, _noise=0.0, _fn=fn: _fn(r, n, _noise))
                names.append(f"{label}Noisy")
                noisy_flags.append(True)
            else:
                # Single slot: no noise expansion; fn already accepts and discards _noise=0.0.
                ops.append(fn)
                names.append(label)
                noisy_flags.append(False)

        return ops, names, noisy_flags

    # ------------------------------------------------------------------
    # LNSBase semantic hooks (ALNS flavour)
    # ------------------------------------------------------------------

    def _removal_bounds(self, current_n_nodes: int) -> Tuple[int, int]:
        """Formal bounds per Ropke & Pisinger (2006) Section 4.3.1.

        Args:
            current_n_nodes: Number of nodes in the current solution.

        Returns:
            Tuple of (lower_bound, upper_bound) for the removal count.
        """
        lower_bound = min(current_n_nodes, MIN_REMOVAL_LOWER_BOUND)
        upper_bound = min(self.params.max_removal_cap, int(self.params.xi * self.n_nodes))
        upper_bound = max(lower_bound, upper_bound)
        return lower_bound, upper_bound

    def _scoring(
        self,
        new_profit: float,
        best_profit: float,
        current_profit: float,
        accepted: bool,
        is_new: bool,
    ) -> float:
        """Ropke & Pisinger (2005, Section 3.4) scoring.

        σ₁ — new global best (no "not visited" qualifier)
        σ₂ — better than current, not visited before, accepted
        σ₃ — worse than current, not visited before, accepted

        Called before the best-solution bookkeeping, so ``best_profit`` is the
        pre-update value.

        Args:
            new_profit: Profit of the candidate solution.
            best_profit: Best profit found so far (pre-update).
            current_profit: Profit of the current incumbent.
            accepted: Whether the candidate was accepted.
            is_new: Whether the candidate was not visited before.

        Returns:
            The score earned by this iteration's operators.
        """
        if new_profit > best_profit + 1e-6:
            return self.params.sigma_1
        if accepted and is_new:
            return self.params.sigma_2 if new_profit > current_profit + 1e-6 else self.params.sigma_3
        return 0.0

    def _snapshot_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """Shallow copy via list comprehension (replaces deepcopy): ~10x faster.

        Args:
            routes: Routes to copy.

        Returns:
            The copied routes.
        """
        return [r[:] for r in routes]

    def _hash_solution(self, routes: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
        """Create a canonical hash of a solution for tracking visited states.

        Args:
            routes: List of routes to hash.

        Returns:
            Tuple representation of sorted routes for hashing.
        """
        sorted_routes = sorted([tuple(route) for route in routes if route])
        return tuple(sorted_routes)

    def _track_and_check_new(self, solution_hash: Tuple[Tuple[int, ...], ...]) -> bool:
        """Check whether the hashed solution was visited before.

        Args:
            solution_hash: Output of ``_hash_solution``.

        Returns:
            True when the solution is not in the visited set.
        """
        return solution_hash not in self.visited_solutions

    def _track_accepted(self, solution_hash: Tuple[Tuple[int, ...], ...]) -> None:
        """Record an accepted solution in the visited set.

        Args:
            solution_hash: Output of ``_hash_solution``.
        """
        self.visited_solutions.add(solution_hash)

    def _prepare_search(self, best_routes: List[List[int]], current_profit: float) -> None:
        """Per-solve prologue: T_start calibration, visited reset, criterion setup.

        Dynamic T_start calibration (Ropke & Pisinger 2006, Section 3.5):
        non-positive initial profits used to skip calibration and leave T = 0,
        i.e. pure hill-climbing; the fallback scale=1.0 fixes that.

        Args:
            best_routes: Best routes after initialization.
            current_profit: Profit of the initial solution.
        """
        if self.params.start_temp == 0.0:
            # Calibrate from |profit| (fallback 1.0)
            scale = abs(current_profit) if abs(current_profit) > 1e-9 else 1.0
            delta = self.params.start_temp_control * scale
            t_start = delta / np.log(2.0)

            # Inject dynamically calculated temperature into the acceptance criterion
            if hasattr(self.acceptance_criterion, "set_temperature"):
                self.acceptance_criterion.set_temperature(t_start)  # type: ignore[union-attr]

        self.visited_solutions.clear()
        self.visited_solutions.add(self._hash_solution(best_routes))

        if self.acceptance_criterion is not None:
            self.acceptance_criterion.setup(current_profit)

    def _accept_move(
        self,
        current_profit: float,
        new_profit: float,
        best_profit: float,
        iteration: int,
    ) -> bool:
        """Acceptance decision via the injected criterion (hill-climb when None).

        Args:
            current_profit: Profit of the current incumbent.
            new_profit: Profit of the candidate solution.
            best_profit: Best profit found so far.
            iteration: Current iteration index.

        Returns:
            True if the candidate is accepted.
        """
        if self.acceptance_criterion is None:
            return new_profit > current_profit + 1e-6
        accept, _ = self.acceptance_criterion.accept(
            current_obj=current_profit,
            candidate_obj=new_profit,
            f_best=best_profit,
            iteration=iteration,
            max_iterations=self.params.max_iterations,
        )
        return accept

    def _extra_iteration_end(self, **kwargs: Any) -> None:
        """Segment-end weight update and criterion cooling.

        Args:
            kwargs: Iteration context (iteration, current_profit, new_profit,
                best_profit, accepted, ...).
        """
        iteration = kwargs["iteration"]
        if (iteration + 1) % self.segment_size == 0:
            self._end_segment()

        if self.acceptance_criterion is not None:
            self.acceptance_criterion.step(
                current_obj=kwargs["current_profit"],
                candidate_obj=kwargs["new_profit"],
                accepted=kwargs["accepted"],
                f_best=kwargs["best_profit"],
                iteration=iteration,
                max_iterations=self.params.max_iterations,
            )

    def _record_iteration(self, **kwargs: Any) -> None:
        """Record one telemetry snapshot for the current iteration.

        Args:
            kwargs: Iteration context (iteration, d_idx, r_idx, best_profit,
                current_profit, accepted, score).
        """
        self._viz_record(
            iteration=kwargs["iteration"],
            d_idx=kwargs["d_idx"],
            r_idx=kwargs["r_idx"],
            best_profit=kwargs["best_profit"],
            current_profit=kwargs["current_profit"],
            acceptance_state=self.acceptance_criterion.get_state() if self.acceptance_criterion else {},
            accepted=int(kwargs["accepted"]),
            score=kwargs["score"],
        )

    # ------------------------------------------------------------------
    # Core solver loop helpers
    # ------------------------------------------------------------------

    def _get_noise(self) -> float:
        """
        Generate noise for repair operators (Ropke & Pisinger 2005, Section 3.4.3).

        Noise is scaled relative to the maximum distance in the problem instance:
        noise ~ U[-eta * max_dist, eta * max_dist]

        Returns:
            float: Noise value in range [-eta * max_dist, eta * max_dist]
        """
        max_dist = self.dist_matrix.max()
        return self.random.uniform(-self.params.noise_factor, self.params.noise_factor) * max_dist

    def _update_weights(self, d_idx: int, r_idx: int, score: float) -> None:
        """Update scores and usage counts for operators.

        Args:
            d_idx: Index of the destroy operator used.
            r_idx: Index of the repair operator used.
            score: Score to add based on solution quality.
        """
        self._weights.update(d_idx, r_idx, score)

    def _end_segment(self) -> None:
        """
        Update operator weights at the end of a segment (Ropke & Pisinger 2005, Eq. 4.1).

        Formula: w_{i,j+1} = w_{i,j} * (1-r) + r * (π_i / θ_i)
        where:
            - w_{i,j} is the weight of operator i in segment j
            - r is the reaction factor (learning rate)
            - π_i is the total score accumulated by operator i
            - θ_i is the number of times operator i was used
        """
        self._weights.end_segment()

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
