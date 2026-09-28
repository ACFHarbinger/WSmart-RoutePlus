"""
Hyper-Heuristic Ant Colony Optimization (Hyper-ACO).

This module implements a Hyper-Heuristic ACO where ants construct
sequences of local search operators rather than node sequences.
The pheromone matrix encodes transition probabilities between operators.

Key Features:
- Operator graph: Nodes represent local search operators (2-opt, swap, etc.)
- Pheromone trail: tau[i][j] = favorability of applying operator j after operator i
- Heuristic information: Dynamic based on operator success rate / execution time
- Solution evaluation: Apply operator sequence to a base solution

Paper reference: Chen, Kendall & Vanden Berghe (2007),
"An Ant Based Hyper-heuristic for the Travelling Tournament Problem".

Adaptations from TTP to VRPP:
- Objective = total routing cost (× C) − collected revenue (× R), subject to
  capacity constraints; infeasible solutions are penalised via Strategic Oscillation.
- `use_dynamic_lambda=True` (default): replaces the paper's λ=1.0001 base with a
  scale-invariant exp(I/Z) using an EMA of recent |improvement|, improving
  robustness across different problem scales.
- `elitism_ratio < 1.0` (optional): hedged elitism for diversity preservation;
  paper default is 1.0 (all ants sync to global best on improvement).

Attributes:
    HyperHeuristicACO: HyperHeuristicACO class.

Example:
    >>> from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic import HyperHeuristicACO
    >>> solver = HyperHeuristicACO(dist_matrix, wastes, capacity, R, C, params)
    >>> best_solution = solver.solve(initial_solution)
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators import (
    bb_insertion,
    bb_profit_insertion,
    deep_insertion,
    deep_profit_insertion,
    farthest_insertion,
    farthest_profit_insertion,
    geni_insertion,
    geni_profit_insertion,
    greedy_insertion,
    greedy_insertion_with_blinks,
    greedy_profit_insertion,
    greedy_profit_insertion_with_blinks,
    nearest_insertion,
    nearest_profit_insertion,
    regret_2_insertion,
    regret_2_profit_insertion,
    regret_3_insertion,
    regret_3_profit_insertion,
    regret_4_insertion,
    regret_4_profit_insertion,
    savings_insertion,
    savings_profit_insertion,
)

from .hyper_operators import (
    HYPER_OPERATORS,
    HyperOperatorContext,
)
from .params import HyperACOParams


def _elitism_sync_count(n_ants: int, elitism_ratio: float) -> int:
    """Number of ants synced to the global best on improvement (ceil, per the flowchart)."""
    return max(1, math.ceil(n_ants * elitism_ratio))


class HyperHeuristicACO:
    """
    Pure Ant System (AS) Hyper-Heuristic ACO solver.

    Ants construct sequences of local search operators.
    The best sequences are reinforced via pheromone updates.

    Attributes:
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        params: Hyper-heuristics parameters.
        initial_solution: Initial routes.
        mandatory_nodes: Mandatory nodes.
        random: Random number generator.
        np_rng: NumPy random number generator.
        operator_names: List of operator names.
        n_operators: Number of operators.
        op_to_idx: Mapping from operator names to indices.
        tau: Pheromone matrix. Row n_operators is the virtual start row used
            only when an ant has no prior position (first journey step after
            uniform placement on real vertices already seeds the real rows).
        eta: Heuristic visibility matrix (per directed edge i→j).
        edge_count: Cumulative edge traversal count (num(i,j) in the paper).
        sequence_length: Length of operator sequences (= n_operators, per paper).
        pheromone: Node-level sparse pheromone matrix (city graph, i→j edges).
            Maintained separately from ``tau`` so that external callers (e.g.
            HVPL) can reinforce node transitions via ``deposit_edge`` without
            coupling into the operator-graph internals.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Optional[HyperACOParams] = None,
        initial_solution: Optional[List[List[int]]] = None,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize HyperHeuristicACO.

        Args:
            dist_matrix: Distance matrix between nodes.
            wastes: Dictionary mapping node indices to wastes.
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Hyper-heuristics parameters object.
            initial_solution: Optional starting routes.
            mandatory_nodes: List of mandatory node indices.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params or HyperACOParams()
        self.initial_solution = initial_solution or []
        self.mandatory_nodes = mandatory_nodes
        self.random = random.Random(self.params.seed) if self.params.seed is not None else random.Random()
        self.np_rng = (
            np.random.default_rng(self.params.seed) if self.params.seed is not None else np.random.default_rng()
        )

        self.operator_names = list(self.params.operators) if self.params.operators else list(HYPER_OPERATORS.keys())
        self.n_operators = len(self.operator_names)
        self.op_to_idx = {name: i for i, name in enumerate(self.operator_names)}

        # Journey length defaults to n (number of heuristics/vertices), per Chen et al. (2007) §III.B
        self.sequence_length = self.params.sequence_length or self.n_operators

        # tau[i][j]: pheromone on directed edge i→j.
        # Row n_operators is a virtual "no prior position" row — only used on the very
        # first selection step of an ant that somehow lacks a recorded position.
        # After uniform placement (Bug #2 fix) the real rows [0..n_operators-1] are
        # seeded immediately by the initial operator application, so this row is
        # primarily a safety fallback.
        self.tau = np.full((self.n_operators + 1, self.n_operators), self.params.tau_0)

        # eta[i][j]: per-edge visibility (heuristic information).
        # Initialised to 1 per paper: "For each vertex j: set ηj = 1".
        self.eta = np.ones((self.n_operators + 1, self.n_operators))

        # edge_count[i][j]: num(i,j) — cumulative count of traversals, never reset.
        # Grows monotonically from the start of the run (per paper §III.B).
        self.edge_count = np.zeros((self.n_operators + 1, self.n_operators))

        # Strategic Oscillation state (per paper §III.B)
        self.initial_best_cost = self._calculate_routing_cost(self.initial_solution)

        # Node-level (city-graph) pheromone structure.
        # Used by external callers (e.g. HVPL) to deposit on route edges via
        # deposit_edge(i, j, delta), keeping city-graph reinforcement decoupled
        # from the operator-sequence pheromone matrix self.tau.
        # Local import to avoid a circular dependency: hyper_aco.py is loaded
        # inside hyper_heuristics/__init__.py, which is imported before
        # meta_heuristics/__init__.py finishes — a module-level import of
        # SparsePheromoneTau would trigger meta_heuristics/__init__.py and
        # create a cycle.  Deferring to instantiation time is safe because
        # both packages are fully initialized by then.
        from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones import (  # noqa: PLC0415
            SparsePheromoneTau,
        )

        n_nodes = len(dist_matrix)  # includes depot (index 0)
        self.pheromone = SparsePheromoneTau(
            n_nodes=n_nodes,
            tau_0=self.params.tau_0,
            scale=5.0,
            tau_min=self.params.tau_0 / 100.0,
            tau_max=self.params.tau_0 * 10.0,
        )
        self.initial_pv = self.initial_best_cost / 100.0 if self.initial_best_cost > 0 else 10.0
        self.pv = self.initial_pv

        # Dynamic Normalization state for scale-invariant visibility (VRPP extension)
        self.Z = self.initial_best_cost * 0.01 if self.initial_best_cost > 0 else 1.0
        self.Z_alpha = 0.1  # EMA smoothing factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Solve the problem using the Ant Colony Optimization algorithm.

        Returns:
            Tuple[List[List[int]], float, float]:
                - Best routes
                - Collected revenue - cost
                - Final cost
        """
        # The incumbent is tracked at the real capacity: strategic oscillation may let ants
        # explore over-capacity plans, but only a feasible plan can become the returned best.
        best_routes = [r for r in copy.deepcopy(self.initial_solution) if r]
        best_objective = self._calculate_routing_cost(best_routes) if self._is_feasible(best_routes) else math.inf

        ant_solutions = [copy.deepcopy(self.initial_solution) for _ in range(self.params.n_ants)]

        # ----------------------------------------------------------------
        # Bug #2 fix: uniform ant placement on real operator vertices.
        # Paper: "Put m ants uniformly on the n vertices."
        # ----------------------------------------------------------------
        ant_positions: List[int] = [i % self.n_operators for i in range(self.params.n_ants)]

        # ----------------------------------------------------------------
        # Bug #2 fix (cont.): apply starting operator before main loop.
        # Paper: "For each ant k: apply low-level heuristic from ant k's
        #         current location to Sk."
        # ----------------------------------------------------------------
        effective_capacity = float("inf") if self.pv < self.initial_pv else self.capacity
        for ant_idx in range(self.params.n_ants):
            start_op_name = self.operator_names[ant_positions[ant_idx]]
            op_func = HYPER_OPERATORS.get(start_op_name)
            if op_func:
                ctx = self._make_context(ant_solutions[ant_idx], effective_capacity)
                op_func(ctx)
                ant_solutions[ant_idx] = ctx.routes

        stagnation_counter = 0
        start_time = time.perf_counter()

        for _it in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.perf_counter() - start_time > self.params.time_limit:
                break

            total_eta_updates = np.zeros_like(self.eta)
            ant_results: List[Tuple[List[List[int]], float, float, List[str], int, int]] = []

            for ant_idx in range(self.params.n_ants):
                start_obj = self._evaluate_objective(ant_solutions[ant_idx])
                journey_start = ant_positions[ant_idx]
                # Bug #2 fix: pass each ant's current position into build_solution
                routes, sequence, eta_updates, end_op_idx = self.build_solution(ant_solutions[ant_idx], journey_start)
                final_obj = self._evaluate_objective(routes)

                # Update ant position for next iteration (persistent across journeys)
                ant_positions[ant_idx] = end_op_idx

                total_eta_updates += eta_updates
                ant_solutions[ant_idx] = routes
                ant_results.append((routes, start_obj, final_obj, sequence, ant_idx, journey_start))

            # Evaluate iteration results (minimisation); the sort also ranks ants for elitism.
            ant_results.sort(key=lambda x: x[2])
            best_routes, best_objective, global_best_improved = self._update_incumbent(
                ant_results, best_routes, best_objective
            )
            if global_best_improved:
                # ----------------------------------------------------------------
                # Bug #4 context: elitism_ratio=1.0 (paper default) → all ants
                # sync to global best (exact paper behaviour).
                # elitism_ratio < 1.0 → hedged elitism (VRPP diversity option).
                # Paper: "For all ants: Sk = Sb"
                # ----------------------------------------------------------------
                sync_count = _elitism_sync_count(self.params.n_ants, self.params.elitism_ratio)

                for rank, (routes, _, _, _, ant_idx, _) in enumerate(ant_results):
                    if rank >= self.params.n_ants - sync_count:
                        ant_solutions[ant_idx] = copy.deepcopy(best_routes)
                    else:
                        ant_solutions[ant_idx] = routes
            else:
                for routes, _, _, _, ant_idx, _ in ant_results:
                    ant_solutions[ant_idx] = routes

            # Strategic Oscillation management (per paper §III.B)
            if global_best_improved:
                stagnation_counter = 0
                self.pv = self.initial_pv
            else:
                stagnation_counter += 1
                if stagnation_counter >= self.params.stagnation_limit:
                    self.pv /= 2.0

            # Pure AS pheromone evaporation
            self._evaporate_pheromones()

            # ----------------------------------------------------------------
            # Bug #1 fix: pheromone deposit includes Q constant.
            # Paper: Δτ^k_ij = Q + I_k / L_k  when I_k > 0.
            # ----------------------------------------------------------------
            self._deposit_pheromones(ant_results)

            # Visibility integration
            self.eta = (self.params.eta_decay * self.eta) + total_eta_updates

        if math.isinf(best_objective):
            # No feasible plan was ever seen (the starting solution was already infeasible):
            # return the starting solution rather than an over-capacity ant plan.
            best_routes = [r for r in copy.deepcopy(self.initial_solution) if r]

        best_cost = self._calculate_routing_cost(best_routes)
        collected_rev = sum(self.wastes.get(n, 0) * self.R for r in best_routes for n in r)
        final_cost = best_cost / self.C if self.C > 0 else best_cost

        return best_routes, collected_rev - best_cost, final_cost

    def construct(
        self,
        nodes: List[int],
        mandatory_nodes: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """Construct a solution from scratch using pure pheromone-guided exploration.

        This method is analogous to ``SolutionConstructor.construct`` in the
        k-sparse ACO: it builds a complete solution from an empty slate rather
        than improving an existing one.  The construction proceeds in two phases:

        1. **Bootstrap** — A bootstrap insertion operator is chosen **uniformly
           at random** from the full portfolio of ``recreate_repair`` heuristics
           (pure exploration — equivalent to flat tau_0 pheromone, matching MMAS
           semantics).  All nodes are treated as the unvisited pool and inserted
           into an initially-empty route list by the selected heuristic.  When
           ``params.profit_aware_operators`` is enabled the profit-maximising
           variant of each heuristic is used; otherwise the cost-minimising
           variant is used.  The available bootstrap operators are:

           Standard (cost-minimising): greedy, greedy-with-blinks, nearest,
           farthest, savings, regret-2, regret-3, regret-4, deep, GENI,
           branch-and-bound LDS.

           Profit-aware (revenue - cost): greedy_profit, blinks_profit,
           nearest_profit, farthest_profit, savings_profit, regret-2_profit,
           regret-3_profit, regret-4_profit, deep_profit, GENI_profit,
           bb_profit.

        2. **Exploration** — A single ant is placed on a uniformly-random
           operator vertex (mirroring the uniform placement in ``solve``) and a
           full operator sequence of length ``sequence_length`` is selected via
           the pure AS proportional (roulette-wheel) rule through
           ``_select_sequence``.  The sequence is then applied to the bootstrapped
           solution via ``build_solution``.  No local pheromone updates are
           performed inside this method — all reinforcement is expected to happen
           globally after evaluation, consistent with MMAS semantics.

        Args:
            nodes: List of all customer node indices (excluding depot 0) that
                the ant should attempt to route.
            mandatory_nodes: Nodes that must be visited regardless of
                profitability.  Overrides ``self.mandatory_nodes`` when provided;
                falls back to ``self.mandatory_nodes`` when ``None``.

        Returns:
            List of routes, each a list of node indices (excluding depot 0),
            representing the constructed solution.
        """
        effective_mandatory = mandatory_nodes if mandatory_nodes is not None else (self.mandatory_nodes or [])

        # ----------------------------------------------------------------
        # Phase 1 – Bootstrap: choose an insertion heuristic uniformly at
        # random from the full recreate_repair portfolio (pure exploration).
        # All nodes play the role of "removed" nodes; starting from [] mirrors
        # the k-sparse ant beginning with all nodes in the unvisited set.
        #
        # Uniform selection is semantically identical to roulette-wheel
        # selection with a flat tau_0 matrix — maximum entropy, no bias.
        # ----------------------------------------------------------------
        bootstrapped: List[List[int]] = self._bootstrap(
            nodes=nodes,
            effective_mandatory=effective_mandatory,
        )

        # ----------------------------------------------------------------
        # Phase 2 – Exploration: place one ant uniformly on the operator
        # graph and select an operator sequence via pure roulette-wheel
        # (no exploitation bias, matching MMAS / k-sparse construct semantics).
        # Then apply the sequence to the bootstrapped solution.
        # ----------------------------------------------------------------
        start_op_idx: int = self.random.randrange(self.n_operators)
        routes, _sequence, _eta_updates, _end_op_idx = self.build_solution(
            base_solution=bootstrapped,
            start_op_idx=start_op_idx,
        )

        return routes

    # ------------------------------------------------------------------
    # Bootstrap insertion portfolio
    # ------------------------------------------------------------------

    def _bootstrap(
        self,
        nodes: List[int],
        effective_mandatory: List[int],
    ) -> List[List[int]]:
        """Select and apply one bootstrap insertion operator uniformly at random.

        Chooses a heuristic from the full ``recreate_repair`` portfolio with
        equal probability (pure exploration) and applies it to an empty route
        list with ``nodes`` as the unvisited pool.

        When ``params.profit_aware_operators`` is ``True``, the profit-aware
        variant of the chosen operator is used; otherwise the cost-minimising
        variant is used.

        Args:
            nodes: Customer node indices to bootstrap (excluding depot 0).
            effective_mandatory: Mandatory node indices that must be served.

        Returns:
            List of routes produced by the selected bootstrap heuristic.
        """
        if self.params.profit_aware_operators:
            return self._bootstrap_profit(nodes, effective_mandatory)
        return self._bootstrap_standard(nodes, effective_mandatory)

    def _bootstrap_standard(self, nodes: List[int], mandatory: List[int]) -> List[List[int]]:
        """Apply a uniformly-random cost-minimising insertion heuristic.

        Candidates: greedy, greedy-with-blinks, nearest, farthest, savings,
        regret-2, regret-3, regret-4, deep, GENI, branch-and-bound LDS.

        Args:
            nodes: Customer nodes to insert.
            mandatory: Mandatory node indices.

        Returns:
            List of routes.
        """
        choice = self.random.randrange(11)  # 11 standard operators

        dm = self.dist_matrix
        ws = self.wastes
        cap = self.capacity
        man = mandatory

        if choice == 0:
            return greedy_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 1:
            return greedy_insertion_with_blinks(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                rng=self.random,
                expand_pool=False,
            )
        if choice == 2:
            return nearest_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 3:
            return farthest_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 4:
            return savings_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 5:
            return regret_2_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 6:
            return regret_3_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 7:
            return regret_4_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 8:
            return deep_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 9:
            return geni_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                rng=self.random,
                expand_pool=False,
            )
        # choice == 10
        return bb_insertion(
            routes=[],
            removed_nodes=nodes,
            dist_matrix=dm,
            wastes=ws,
            capacity=cap,
            max_discrepancy=0,
            mandatory_nodes=man,
            expand_pool=False,
        )

    def _bootstrap_profit(self, nodes: List[int], mandatory: List[int]) -> List[List[int]]:
        """Apply a uniformly-random profit-aware insertion heuristic.

        Candidates: greedy_profit, blinks_profit, nearest_profit,
        farthest_profit, savings_profit, regret-2_profit, regret-3_profit,
        regret-4_profit, deep_profit, GENI_profit, bb_profit.

        Args:
            nodes: Customer nodes to insert.
            mandatory: Mandatory node indices.

        Returns:
            List of routes.
        """
        choice = self.random.randrange(11)  # 11 profit-aware operators

        dm = self.dist_matrix
        ws = self.wastes
        cap = self.capacity
        R = self.R
        C = self.C
        man = mandatory

        if choice == 0:
            return greedy_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 1:
            return greedy_profit_insertion_with_blinks(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                rng=self.random,
                expand_pool=False,
            )
        if choice == 2:
            return nearest_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 3:
            return farthest_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 4:
            return savings_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 5:
            return regret_2_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 6:
            return regret_3_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 7:
            return regret_4_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 8:
            return deep_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 9:
            return geni_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                rng=self.random,
                expand_pool=False,
            )
        # choice == 10
        return bb_profit_insertion(
            routes=[],
            removed_nodes=nodes,
            dist_matrix=dm,
            wastes=ws,
            capacity=cap,
            R=R,
            C=C,
            max_discrepancy=0,
            mandatory_nodes=man,
            expand_pool=False,
        )

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
                if self.params.use_dynamic_lambda:
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
                    lam = self.params.lambda_val**improvement

                self.edge_count[prev_op_idx][next_op_idx] += 1
                safe_count = max(self.edge_count[prev_op_idx][next_op_idx], 1)

                if self.params.time_weighted_visibility:
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

        Returns:
            None.
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
            profit_aware_operators=self.params.profit_aware_operators,
            vrpp=self.params.vrpp,
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
            numerator = (self.tau[current_op_idx] ** self.params.alpha) * (self.eta[current_op_idx] ** self.params.beta)
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

        Args:
            ant_results: Per-ant results ``(routes, start_obj, final_obj, sequence, ant_idx, start_idx)``.

        Returns:
            None
        """
        for _routes, start_obj_ant, final_obj_ant, sequence, _ant_idx, journey_start in ant_results:
            journey_improvement = start_obj_ant - final_obj_ant
            if journey_improvement > 0:
                delta_tau = self.params.Q + journey_improvement / len(sequence)
                prev_op_idx = journey_start
                for op_name in sequence:
                    next_op_idx = self.op_to_idx[op_name]
                    self.tau[prev_op_idx][next_op_idx] += delta_tau
                    prev_op_idx = next_op_idx

    def _update_incumbent(
        self,
        ant_results: List[Tuple[List[List[int]], float, float, List[str], int, int]],
        best_routes: List[List[int]],
        best_objective: float,
    ) -> Tuple[List[List[int]], float, bool]:
        """Replace the incumbent with this iteration's best plan that is feasible at the real capacity.

        Args:
            ant_results: Per-ant results ``(routes, start_obj, final_obj, sequence, ant_idx, start_idx)``.
            best_routes: Current incumbent routes.
            best_objective: Routing cost of the incumbent (``inf`` if none is feasible yet).

        Returns:
            Tuple of (incumbent routes, incumbent cost, whether it improved).
        """
        feasible = [r for r in ant_results if self._is_feasible(r[0])]
        if not feasible:
            return best_routes, best_objective, False
        cand_routes = min(feasible, key=lambda r: r[2])[0]
        cand_cost = self._calculate_routing_cost(cand_routes)
        if cand_cost < best_objective:
            return [r for r in copy.deepcopy(cand_routes) if r], cand_cost, True
        return best_routes, best_objective, False

    def _is_feasible(self, routes: List[List[int]]) -> bool:
        """Return True if every route respects the real vehicle capacity.

        Args:
            routes: Routes.

        Returns:
            bool: True when no route's load exceeds ``self.capacity``.
        """
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
        num_violations = sum(1 for r in routes if sum(self.wastes.get(n, 0) for n in r) > self.capacity)
        return routing_cost + (self.pv * num_violations)

    def _evaporate_pheromones(self) -> None:
        """Pure Ant System pheromone evaporation.

        Paper: τ_ij(t) = ρ · τ_ij(t−n) + Σ_k Δτ^k_ij
        Evaporation is applied before deposit each iteration (equivalent since
        each outer iteration corresponds to exactly one complete journey).

        Returns:
            None
        """
        self.tau *= 1 - self.params.rho

    def evaporate_all(self) -> None:
        """Evaporate node-level pheromones.

        Returns:
            None
        """
        self._evaporate_pheromones()
        self.pheromone.evaporate_all(self.params.rho)
