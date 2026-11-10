"""Operator-sequence construction and pheromone bookkeeping of the ACO hyper-heuristic.

:class:`HyperACOPheromoneMixin` holds how an ant builds and applies an operator
sequence (``build_solution``), the node-level pheromone interface used by external
callers (``deposit_edge``, ``evaporate_all``) and the private helpers for the operator
context, sequence selection, routing cost, pheromone deposit and evaporation, incumbent
update, feasibility and objective evaluation. Moved unchanged from ``hyper_aco.py``.
"""

import copy
import math
import random
import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from logic.src.configs.policies.routing.hyper_heuristics.aco_hh import HyperHeuristicACOConfig

from .hyper_operators import (
    HYPER_OPERATORS,
    HyperOperatorContext,
)

if TYPE_CHECKING:
    from logic.src.policies.routing.meta_heuristics.ant_colony_optimization_k_sparse.pheromones import (
        SparsePheromoneTau,
    )


class HyperACOPheromoneMixin:
    """Sequence construction, pheromone and evaluation helpers of :class:`HyperHeuristicACO` (moved unchanged from ``hyper_aco.py``)."""

    # Provided by HyperHeuristicACO.__init__; annotations only.
    cfg: HyperHeuristicACOConfig
    random: random.Random
    np_rng: np.random.Generator
    dist_matrix: np.ndarray
    wastes: Dict[int, float]
    capacity: float
    R: float
    C: float
    mandatory_nodes: List[int]
    n_operators: int
    operator_names: List[str]
    op_to_idx: Dict[str, int]
    sequence_length: int
    tau: np.ndarray
    eta: np.ndarray
    edge_count: np.ndarray
    pv: float
    initial_pv: float
    Z: float
    Z_alpha: float
    pheromone: "SparsePheromoneTau"

    # ------------------------------------------------------------------
    # Bootstrap insertion portfolio
    # ------------------------------------------------------------------

    def build_solution(
        self,
        base_solution: List[List[int]],
        start_op_idx: int,
    ) -> Tuple[List[List[int]], List[str], np.ndarray, int]:
        """
        Build a solution by constructing and applying an operator sequence.

        Args:
            base_solution: Base solution to improve.
            start_op_idx: The operator vertex this ant currently occupies.
                Sequences are constructed beginning from this vertex, not from
                the virtual start node (Bug #2 fix).

        Returns:
            Tuple of:
                - Improved routes.
                - Sequence of operator names applied.
                - Per-edge eta update matrix accumulated during this sequence.
                - Ending operator index (the ant's new position for next iteration).
        """
        sequence, end_op_idx = self._select_sequence(start_op_idx)

        # Strategic Oscillation: relax capacity when search is stagnating
        effective_capacity = float("inf") if self.pv < self.initial_pv else self.capacity

        # Optimization: only compute SWAP* cache if the sequence contains it
        use_cache = "swap_star" in sequence
        ctx = self._make_context(base_solution, effective_capacity, use_cache=use_cache)

        eta_updates = np.zeros_like(self.eta)
        stability_constant = 1e-3
        prev_op_idx = start_op_idx  # first edge departs from the ant's current real vertex
        for op_name in sequence:
            next_op_idx = self.op_to_idx[op_name]
            op_func = HYPER_OPERATORS.get(op_name)
            if op_func:
                obj_before = self._evaluate_objective(ctx.routes)
                t0 = time.perf_counter()
                op_func(ctx)
                execution_time = time.perf_counter() - t0
                obj_after = self._evaluate_objective(ctx.routes)

                improvement = obj_before - obj_after

                # ----------------------------------------------------------------
                # Bug #3 context: visibility contribution λ^I or exp(I/Z).
                # Paper: λ=1.0001 as base of exponential (λ^{I_kj}).
                # VRPP extension: exp(I/Z) with running EMA normaliser Z.
                # ----------------------------------------------------------------
                if self.cfg.use_dynamic_lambda:
                    # Scale-invariant extension: update EMA of |improvement|
                    abs_imp = abs(improvement)
                    if abs_imp > 1e-6:
                        self.Z = (1 - self.Z_alpha) * self.Z + self.Z_alpha * abs_imp
                    scaled_improvement = improvement / max(self.Z, 1e-6)
                    capped = max(min(scaled_improvement, 10.0), -10.0)
                    lam = math.exp(capped)
                else:
                    # Paper-faithful: λ = 1.0001 as exponential base.
                    # λ^I is positive for any real I, converging to 0 for large
                    # negative I (rare operators still get a tiny signal).
                    lam = self.cfg.lambda_val**improvement

                self.edge_count[prev_op_idx][next_op_idx] += 1
                safe_count = max(self.edge_count[prev_op_idx][next_op_idx], 1)

                if self.cfg.time_weighted_visibility:
                    eta_updates[prev_op_idx][next_op_idx] += lam / ((execution_time + stability_constant) * safe_count)
                else:
                    eta_updates[prev_op_idx][next_op_idx] += lam / safe_count

            prev_op_idx = next_op_idx

        return ctx.routes, sequence, eta_updates, end_op_idx

    # ------------------------------------------------------------------
    # Public pheromone interface (city-graph / node-level)
    # ------------------------------------------------------------------

    def deposit_edge(self, i: int, j: int, delta: float) -> None:
        """Deposit pheromone on city-graph edge (i, j).

        Adds ``delta`` to the node-level pheromone value for the directed
        edge ``i → j``.  This is the public analogue of
        ``SparsePheromoneTau.deposit_edge`` and is used by external callers
        (e.g. HVPL) to reinforce city-graph transitions without accessing
        the internal operator-sequence matrix ``self.tau``.

        Args:
            i: Source node index (0 = depot).
            j: Destination node index (0 = depot).
            delta: Amount of pheromone to add.

        """
        self.pheromone.deposit_edge(i, j, delta)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_context(
        self, routes: List[List[int]], effective_capacity: float, use_cache: bool = False
    ) -> "HyperOperatorContext":
        """Construct a HyperOperatorContext for the given solution snapshot.

        Args:
            routes: Current routes (will be deep-copied inside).
            effective_capacity: Capacity to enforce (may be inf during oscillation).
            use_cache: Whether to use the cache for operator applications.

        Returns:
            HyperOperatorContext ready for operator application.
        """
        return HyperOperatorContext(
            routes=copy.deepcopy(routes),
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=effective_capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            profit_aware_operators=self.cfg.profit_aware_operators,
            use_cache=use_cache,
        )

    def _select_sequence(self, start_op_idx: int) -> Tuple[List[str], int]:
        """
        Pure Ant System proportional selection, starting from a real operator vertex.

        Bug #2 fix: the sequence now departs from the ant's actual current position
        (a real operator index in [0, n_operators-1]) rather than the virtual start
        node (index n_operators). This means pheromone trails on real operator-to-
        operator edges are reinforced from the very first journey.

        Args:
            start_op_idx: Operator index this ant currently occupies.

        Returns:
            Tuple of:
                - Ordered list of operator names selected for this journey.
                - Ending operator index (ant's new position).
        """
        sequence: List[str] = []
        current_op_idx = start_op_idx

        for _ in range(self.sequence_length):
            numerator = (self.tau[current_op_idx] ** self.cfg.alpha) * (self.eta[current_op_idx] ** self.cfg.beta)
            denom = np.sum(numerator)
            probs = numerator / denom if denom > 1e-12 else np.ones(self.n_operators) / self.n_operators
            next_op_idx = self.np_rng.choice(self.n_operators, p=probs)

            sequence.append(self.operator_names[next_op_idx])
            current_op_idx = next_op_idx

        return sequence, current_op_idx

    def _calculate_routing_cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing cost.

        Args:
            routes: Routes.

        Returns:
            float: Total routing cost.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0, route[0]]
            for i in range(len(route) - 1):
                total += self.dist_matrix[route[i], route[i + 1]]
            total += self.dist_matrix[route[-1], 0]
        return total * self.C

    def _deposit_pheromones(self, ant_results: List[Tuple[List[List[int]], float, float, List[str], int, int]]) -> None:
        """Deposit ``Q + I_k / L_k`` along every improving journey's path, first hop included.

        The journey with the largest improvement deposits ``elitist_weight`` times as much (1.0: like the others).

        Args:
            ant_results: Per-ant results ``(routes, start_obj, final_obj, sequence, ant_idx, start_idx)``.

        """
        improvements = [start - final for _r, start, final, _s, _a, _j in ant_results]
        elite = max(range(len(improvements)), key=lambda k: improvements[k]) if improvements else -1
        for k, (_routes, start_obj_ant, final_obj_ant, sequence, _ant_idx, journey_start) in enumerate(ant_results):
            journey_improvement = start_obj_ant - final_obj_ant
            if journey_improvement > 0:
                delta_tau = self.cfg.Q + journey_improvement / len(sequence)
                if k == elite:
                    delta_tau *= self.cfg.elitist_weight
                prev_op_idx = journey_start
                for op_name in sequence:
                    next_op_idx = self.op_to_idx[op_name]
                    self.tau[prev_op_idx][next_op_idx] += delta_tau
                    prev_op_idx = next_op_idx

    def _update_incumbent(
        self,
        results: List[Any],
        best_routes: List[List[int]],
        best_objective: float,
    ) -> Tuple[List[List[int]], float, bool]:
        """Replace the incumbent with this iteration's best plan that is feasible at the real capacity.

        Args:
            results: Per-ant results ``(routes, start_obj, final_obj, sequence, ant_idx, start_idx)``.
            best_routes: Current incumbent routes.
            best_objective: Routing cost of the incumbent (``inf`` if none is feasible yet).

        Returns:
            Tuple of (incumbent routes, incumbent cost, whether it improved).
        """
        feasible = [r for r in results if self._is_feasible(r[0])]
        if not feasible:
            return best_routes, best_objective, False
        cand_routes = min(feasible, key=lambda r: r[2])[0]
        cand_cost = self._calculate_routing_cost(cand_routes)
        if cand_cost < best_objective:
            return [r for r in copy.deepcopy(cand_routes) if r], cand_cost, True
        return best_routes, best_objective, False

    def _is_feasible(self, routes: List[List[int]]) -> bool:
        """Return True if the plan visits every mandatory node and every route respects the real capacity.

        Args:
            routes: Routes.

        Returns:
            bool: True when no mandatory node is missing and no route's load exceeds ``self.capacity``.
        """
        if self.mandatory_nodes and not set(self.mandatory_nodes) <= {n for r in routes for n in r}:
            return False
        return all(sum(self.wastes.get(n, 0) for n in r) <= self.capacity + 1e-9 for r in routes)

    def _evaluate_objective(self, routes: List[List[int]]) -> float:
        """
        Adaptive objective incorporating Strategic Oscillation penalty.

        Infeasible solutions (capacity violations) are penalised by pv per
        violation. pv is dynamically adjusted during the search (halved on
        stagnation, reset to initial_pv on improvement) per Chen et al. (2007).

        Args:
            routes: Routes.

        Returns:
            float: Penalised objective value.
        """
        routing_cost = self._calculate_routing_cost(routes)
        num_violations = sum((1 for r in routes if sum(self.wastes.get(n, 0) for n in r) > self.capacity), 0)
        return routing_cost + (self.pv * num_violations)

    def _evaporate_pheromones(self) -> None:
        """Pure Ant System pheromone evaporation.

        Paper: τ_ij(t) = ρ · τ_ij(t−n) + Σ_k Δτ^k_ij
        Evaporation is applied before deposit each iteration (equivalent since
        each outer iteration corresponds to exactly one complete journey).

        """
        self.tau *= 1 - self.cfg.rho

    def evaporate_all(self) -> None:
        """Evaporate node-level pheromones."""
        self._evaporate_pheromones()
        self.pheromone.evaporate_all(self.cfg.rho)
