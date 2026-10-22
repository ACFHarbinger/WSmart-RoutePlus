"""
Large Neighborhood Search (LNS) policy module.

This module provides the PG-CLNS coaching LNS. The solver builds on the
shared :class:`~logic.src.policies.routing.meta_heuristics.large_neighborhood_search.engine.LNSBase`
loop skeleton (WP-11 P-31) while keeping its OWN frozen operator copies from
this package's ``operators/`` directory (owner ruling 2026-10-07: the PG-CLNS
operators must not be swapped for the shared library). PG-CLNS-specific
semantics (EMA weights, deep-copy policy, ``max_removal_pct``/``xi`` removal
bounds, no visited set, 3/1 scoring, inline-SA acceptance) live in the hooks
below.

Reference:
    Pisinger, D., & Ropke, S. (2007). A general heuristic for vehicle routing problems.
    Computers & Operations Research, 34(8), 2403-2435.

Attributes:
    LNSSolver: Implementation of Large Neighborhood Search for VRP variants.

Examples:
    >>> from logic.src.policies.routing.base.sub_problem import build_subproblem
    >>> problem = build_subproblem(dist_matrix, wastes, capacity, R, C)
    >>> solver = LNSSolver(problem, params, seed=42)
    >>> routes, profit, cost = solver.solve()
"""

import copy
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from logic.src.configs.policies.routing.meta_heuristics.pg_clns import PGCLNSLNSConfig
from logic.src.policies.routing.base.sub_problem import SubProblem, coerce_config
from logic.src.policies.routing.meta_heuristics.large_neighborhood_search.acceptance import (
    LegacySAAcceptance,
)
from logic.src.policies.routing.meta_heuristics.large_neighborhood_search.engine import LNSBase
from logic.src.policies.routing.meta_heuristics.large_neighborhood_search.weight_updates import (
    EmaWeightUpdate,
)
from logic.src.tracking.viz_mixin import PolicyVizMixin

from .operators import cluster_removal, greedy_insertion, random_removal, regret_2_insertion, worst_removal


class LNSSolver(LNSBase, PolicyVizMixin):
    """
    Custom implementation of Large Neighborhood Search for VRP variants.

    PG-CLNS keeps its own frozen operator copies and its pre-P-11 semantics:
    per-iteration EMA weight update, deep-copy solution snapshots, no visited
    set, and the legacy inline simulated-annealing acceptance packaged as
    :class:`LegacySAAcceptance` (rebuilt per ``solve()`` so the temperature
    trajectory resets).

    Attributes:
        problem: The SubProblem this solver was built for.
        dist_matrix (np.ndarray): NxN distance matrix.
        wastes (Dict[int, float]): Dictionary of node wastes.
        capacity (float): Maximum vehicle capacity.
        R (float): Revenue multiplier.
        C (float): Cost multiplier.
        params (PGCLNSLNSConfig): Detailed LNS parameters.
        cfg (PGCLNSLNSConfig): Configuration backing ``params``.
        mandatory_nodes (Optional[List[int]]): List of mandatory node indices.
        random (random.Random): Random number generator.
        destroy_ops (List[Callable]): List of destroy operators.
        repair_ops (List[Callable]): List of repair operators.
        repair_noisy (List[bool]): Noisy-slot flags (all False for this engine).
        destroy_weights (List[float]): Weights for destroy operators.
        repair_weights (List[float]): Weights for repair operators.
        acceptance_criterion (LegacySAAcceptance): Fresh per-solve SA acceptance.
        n_nodes (int): Number of nodes.
        nodes (List[int]): List of nodes.
    """

    def __init__(
        self,
        problem: SubProblem,
        cfg: Optional[Union[PGCLNSLNSConfig, Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the LNS solver.

        Args:
            problem: The routing sub-problem (distance matrix, wastes,
                capacity, revenue, cost unit, mandatory nodes).
            cfg: Detailed PGCLNSLNSConfig or dictionary of parameters (the nested
                ``lns`` section of PGCLNSConfig).
            seed: Random seed for reproducibility. Effective seed sequence:
                explicit kwarg > ``cfg.seed`` > 42.
        """
        self.problem = problem
        self.dist_matrix = problem.dist_matrix
        self.wastes = problem.wastes
        self.capacity = problem.capacity
        self.R = problem.revenue
        self.C = problem.cost_unit

        config = coerce_config(cfg, PGCLNSLNSConfig)
        self.cfg: PGCLNSLNSConfig = config
        self.params = config
        self.mandatory_nodes = problem.mandatory_nodes

        if seed is not None:
            actual_seed = seed
        elif config.seed is not None:
            actual_seed = config.seed
        else:
            actual_seed = 42
        self.random = random.Random(actual_seed)

        self.n_nodes = len(self.dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # ------------------------------------------------------------------
        # Operator registry — PG-CLNS keeps its OWN frozen operator copies
        # (owner ruling 2026-10-07). The 2-arg operators get swallow-noise
        # wrappers so they plug into the shared 3-arg repair slot shape.
        # ------------------------------------------------------------------
        self.destroy_ops: List[Callable] = [
            lambda r, n: random_removal(r, n, rng=self.random),
            lambda r, n: worst_removal(r, n, self.dist_matrix),
            lambda r, n: cluster_removal(r, n, self.dist_matrix, self.nodes, rng=self.random),
        ]
        self.repair_ops: List[Callable] = [
            lambda r, n, _noise=0.0: greedy_insertion(
                r,
                n,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
                cost_unit=self.C,
            ),
            lambda r, n, _noise=0.0: regret_2_insertion(
                r,
                n,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
                cost_unit=self.C,
            ),
        ]
        self.repair_noisy = [False] * len(self.repair_ops)

        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)
        self._weights = EmaWeightUpdate(self.destroy_weights, self.repair_weights)

    # ------------------------------------------------------------------
    # LNSBase semantic hooks (PG-CLNS flavour)
    # ------------------------------------------------------------------

    def _removal_bounds(self, current_n_nodes: int) -> Tuple[int, int]:
        """PG-CLNS removal bounds: ``max_removal_pct`` (falling back to ``xi``).

        Args:
            current_n_nodes: Number of nodes in the current solution.

        Returns:
            Tuple of (lower_bound, upper_bound) for the removal count.
        """
        lower_bound = min(current_n_nodes, 2)
        max_pct = getattr(self.params, "max_removal_pct", getattr(self.params, "xi", 0.4))
        max_pct_remove = int(current_n_nodes * max_pct)
        upper_bound = max(lower_bound + 1, max_pct_remove)
        upper_bound = min(upper_bound, current_n_nodes)
        return lower_bound, upper_bound

    def _scoring(
        self,
        new_profit: float,
        best_profit: float,
        current_profit: float,
        accepted: bool,
        is_new: bool,
    ) -> float:
        """PG-CLNS scoring: 3 on accepted new best, 1 on any other acceptance.

        (Improving candidates are always accepted by the SA rule, so a new best
        can only occur on an accepted move — matching the legacy inline loop.)

        Args:
            new_profit: Profit of the candidate solution.
            best_profit: Best profit found so far (pre-update).
            current_profit: Profit of the current incumbent.
            accepted: Whether the candidate was accepted.
            is_new: Unused (PG-CLNS does not track visited solutions).

        Returns:
            The score earned by this iteration's operators.
        """
        if not accepted:
            return 0
        if new_profit > best_profit + 1e-6:
            return 3
        return 1

    def _snapshot_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """Deep-copy snapshot (the pre-P-31 PG-CLNS copy policy).

        Args:
            routes: Routes to copy.

        Returns:
            The copied routes.
        """
        return copy.deepcopy(routes)

    def _prepare_search(self, best_routes: List[List[int]], current_profit: float) -> None:
        """Build a fresh legacy-SA acceptance per solve (temperature resets).

        Args:
            best_routes: Best routes after initialization (unused).
            current_profit: Profit of the initial solution (unused).
        """
        spec = self.params.acceptance_criterion
        method = spec.get("method") if isinstance(spec, dict) else spec.method
        if method != "bmc":
            raise ValueError("PG-CLNS coaching requires bmc acceptance with the solver RNG")
        params = spec.get("params", {}) if isinstance(spec, dict) else spec.params
        read = params.get if isinstance(params, dict) else lambda key: getattr(params, key)
        self.acceptance_criterion = LegacySAAcceptance(
            start_temp=read("initial_temp"),
            cooling_rate=read("alpha"),
            rng=self.random,
        )

    def _accept_move(
        self,
        current_profit: float,
        new_profit: float,
        best_profit: float,
        iteration: int,
    ) -> bool:
        """SA acceptance decision (draws from the solver's ``random`` stream).

        Args:
            current_profit: Profit of the current incumbent.
            new_profit: Profit of the candidate solution.
            best_profit: Best profit found so far (unused).
            iteration: Current iteration index (unused).

        Returns:
            True if the candidate is accepted.
        """
        accepted, _ = self.acceptance_criterion.accept(current_obj=current_profit, candidate_obj=new_profit)
        return accepted

    def _extra_iteration_end(self, **kwargs: Any) -> None:
        """Cool the legacy-SA temperature (``T *= cooling_rate``) once per iteration.

        Args:
            kwargs: Iteration context (current_profit, new_profit, accepted, ...).
        """
        self.acceptance_criterion.step(
            current_obj=kwargs["current_profit"],
            candidate_obj=kwargs["new_profit"],
            accepted=kwargs["accepted"],
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
            temperature=self.acceptance_criterion.T,
            accepted=int(kwargs["accepted"]),
            score=kwargs["score"],
        )

    # ------------------------------------------------------------------
    # Constructive heuristic (PG-CLNS keeps its own copy)
    # ------------------------------------------------------------------

    def build_initial_solution(self) -> List[List[int]]:
        """
        Build a basic feasible solution using a greedy constructive heuristic.
        For VRPP, it only includes profitable nodes.

        Returns:
            List[List[int]]: Initial routes.
        """
        nodes = self.nodes[:]
        self.random.shuffle(nodes)
        routes = []
        curr_route = []
        load = 0.0
        mandatory_set = set(self.mandatory_nodes) if self.mandatory_nodes else set()

        for node in nodes:
            waste = self.wastes.get(node, 0)
            revenue = waste * self.R
            is_mandatory = node in mandatory_set

            # Basic VRPP check for initial solution: is node potentially profitable?
            # (Heuristic: revenue > cost to depot and back)
            if not is_mandatory and revenue < (self.dist_matrix[0][node] + self.dist_matrix[node][0]) * self.C:
                continue

            if load + waste <= self.capacity:
                curr_route.append(node)
                load += waste
            else:
                if curr_route:
                    routes.append(curr_route)
                curr_route = [node]
                load = waste
        if curr_route:
            routes.append(curr_route)
        return routes
