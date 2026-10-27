r"""Shared large-neighborhood-search engine skeleton (WP-11 P-31).

:class:`LNSBase` generalises the iteration loop that previously existed as two
near-verbatim copies: the released classic ALNS
(``adaptive_large_neighborhood_search/alns.py``) and the PG-CLNS coaching LNS
(``pheromone_guided_cooperative_large_neighborhood_search/lns.py``). The two
engines differ only through a handful of semantic hooks; everything that must
stay bit-identical — roulette-wheel operator selection, cost evaluation, the
greedy initial solution, and the RNG consumption order — lives here once.

RNG determinism invariant: ``self.random`` / ``self.np_random`` are consumed in
exactly the same order as the pre-P-31 engines: destroy roulette, repair
roulette, removal-count draw, noise draw (noisy repair slots only), then the
operator calls; the PG-CLNS acceptance draw happens at the same point as the
legacy inline ``_accept_solution``.

Attributes:
    LNSBase: Shared LNS loop skeleton.
    MIN_REMOVAL_LOWER_BOUND: Named lower bound for the classic ALNS removal count.
    coerce_config: Lenient engine-level config coercion.

"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from logic.src.policies.helpers.operators import build_greedy_routes
from logic.src.tracking.execution_trace import solution_record

__all__ = ["LNSBase", "MIN_REMOVAL_LOWER_BOUND"]

MIN_REMOVAL_LOWER_BOUND: int = 4


class LNSBase:
    r"""Shared LNS loop skeleton for the classic ALNS and the PG-CLNS LNS.

    Subclasses provide the operator registries (``destroy_ops`` /
    ``repair_ops`` / ``repair_noisy``) and a weight-update strategy in
    ``self._weights``, then specialise the semantic hooks:

    - ``_removal_bounds(current_n_nodes)`` — (lower, upper) removal-count bounds.
    - ``_scoring(new_profit, best_profit, current_profit, accepted, is_new)`` —
      per-iteration operator score.
    - ``_snapshot_routes(routes)`` — copy policy for best routes and the
      working copy handed to the destroy operator.
    - ``_hash_solution`` / ``_track_and_check_new`` / ``_track_accepted`` —
      visited-solution policy (classic ALNS tracks; PG-CLNS does not).
    - ``_accept_move(current_profit, new_profit, best_profit, iteration)`` —
      acceptance decision.
    - ``_prepare_search(best_routes, current_profit)`` — per-solve prologue
      (temperature calibration / fresh acceptance criterion).
    - ``_extra_iteration_end(...)`` — segment boundaries and criterion cooling.
    - ``_record_iteration(...)`` — per-iteration telemetry snapshot.

    Attributes:
        problem: The SubProblem this solver was built for.
        dist_matrix: Distance matrix between nodes.
        wastes: Dictionary mapping node indices to waste amounts.
        capacity: Vehicle capacity.
        R: Revenue per unit collected.
        C: Cost per unit distance.
        params: LNS parameters (an ALNSConfig).
        mandatory_nodes: List of nodes that must be visited.
        random: Random number generator for selection and acceptance.
        destroy_ops: Mutable list of destroy operator callables.
        repair_ops: Mutable list of repair operator callables.
        repair_noisy: Boolean flags for noisy repair operator slots.
        destroy_weights: Adaptive weights for destroy operators.
        repair_weights: Adaptive weights for repair operators.
    """

    problem: Any
    dist_matrix: Any
    wastes: Dict[int, float]
    capacity: float
    R: float
    C: float
    params: Any
    mandatory_nodes: Optional[List[int]]
    random: Any
    destroy_ops: List[Callable]
    repair_ops: List[Callable]
    repair_noisy: List[bool]
    destroy_weights: List[float]
    repair_weights: List[float]
    _weights: Any

    # ------------------------------------------------------------------
    # Semantic hooks (per-class behaviour)
    # ------------------------------------------------------------------

    def _removal_bounds(self, current_n_nodes: int) -> Tuple[int, int]:
        """Return the (lower, upper) bounds for the number of nodes to remove.

        Args:
            current_n_nodes: Number of nodes in the current solution.

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        raise NotImplementedError

    def _scoring(
        self,
        new_profit: float,
        best_profit: float,
        current_profit: float,
        accepted: bool,
        is_new: bool,
    ) -> float:
        """Compute the per-iteration operator score.

        Called before the best-solution bookkeeping, so ``best_profit`` is the
        pre-update value.

        Args:
            new_profit: Profit of the candidate solution.
            best_profit: Best profit found so far (pre-update).
            current_profit: Profit of the current incumbent.
            accepted: Whether the candidate was accepted.
            is_new: Whether the candidate is new under the tracking policy.

        Returns:
            The score earned by this iteration's operators.
        """
        raise NotImplementedError

    def _snapshot_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """Copy ``routes`` according to the engine's copy policy.

        Args:
            routes: Routes to copy.

        Returns:
            The copied routes.
        """
        raise NotImplementedError

    def _hash_solution(self, routes: List[List[int]]) -> Any:
        """Hash a solution for visited-state tracking (engines without tracking return None).

        Args:
            routes: List of routes to hash.

        Returns:
            A hashable representation, or None when tracking is disabled.
        """
        return None

    def _track_and_check_new(self, solution_hash: Any) -> bool:
        """Check whether the hashed solution is new under the tracking policy.

        Args:
            solution_hash: Output of ``_hash_solution``.

        Returns:
            True when the solution is considered new.
        """
        return True

    def _track_accepted(self, solution_hash: Any) -> None:
        """Record an accepted solution under the tracking policy.

        Args:
            solution_hash: Output of ``_hash_solution``.
        """
        return None

    def _prepare_search(self, best_routes: List[List[int]], current_profit: float) -> None:
        """Per-solve prologue hook (temperature calibration, fresh criterion, ...).

        Args:
            best_routes: Best routes after initialization.
            current_profit: Profit of the initial solution.
        """
        return None

    def _accept_move(
        self,
        current_profit: float,
        new_profit: float,
        best_profit: float,
        iteration: int,
        **kwargs: Any,
    ) -> bool:
        """Decide whether the candidate solution is accepted.

        Args:
            current_profit: Profit of the current incumbent.
            new_profit: Profit of the candidate solution.
            best_profit: Best profit found so far.
            iteration: Current iteration index.
            kwargs: Extra context for acceptance (e.g. trace, routes).

        Returns:
            True if the candidate is accepted.
        """
        raise NotImplementedError

    def _extra_iteration_end(self, **kwargs: Any) -> None:
        """Per-iteration epilogue hook (segment end, criterion cooling, ...)."""
        return None

    def _record_iteration(self, **kwargs: Any) -> None:
        """Per-iteration telemetry hook."""
        return None

    def _get_noise(self) -> float:
        """Generate noise for noisy repair slots (0.0 when unused).

        Returns:
            Noise value; engines without noisy slots never call this.
        """
        return 0.0

    # ------------------------------------------------------------------
    # Core solver loop helpers (copied once, verbatim semantics)
    # ------------------------------------------------------------------

    def select_operator(self, weights: List[float]) -> int:
        """Roulette-wheel operator selection (Eq. 20).

        Args:
            weights: List of operator weights.

        Returns:
            Index of the selected operator.
        """
        total = sum(weights)
        r = self.random.uniform(0, total)
        curr = 0.0
        for i, w in enumerate(weights):
            curr += w
            if curr >= r:
                return i
        return len(weights) - 1

    def calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculates the total routing cost (distance) for the given routes.

        Args:
            routes: List of routes to evaluate.

        Returns:
            Total calculated routing cost.
        """
        total_dist = 0.0
        for route in routes:
            if not route:
                continue
            dist = self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                dist += self.dist_matrix[route[i]][route[i + 1]]
            dist += self.dist_matrix[route[-1]][0]
            total_dist += dist
        return total_dist * self.C

    def build_initial_solution(self) -> List[List[int]]:
        """Build an initial solution using the shared greedy profit-aware heuristic.

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

    def _initialize_solve(
        self, initial_solution: Optional[List[List[int]]]
    ) -> Tuple[List[List[int]], List[List[int]], float, float]:
        """Initialize the solver state with an initial solution.

        Args:
            initial_solution: Optional starting solution.

        Returns:
            Tuple containing (current_routes, best_routes, best_profit, best_cost).
        """
        current_routes = initial_solution or self.build_initial_solution()
        best_routes = self._snapshot_routes(current_routes)
        best_cost = self.calculate_cost(best_routes)
        best_rev = sum(self.wastes.get(node_idx, 0) * self.R for route in best_routes for node_idx in route)
        best_profit = best_rev - best_cost
        return current_routes, best_routes, best_profit, best_cost

    def _select_and_apply_operators(
        self,
        current_routes: List[List[int]],
        trace: Optional[Any] = None,
        stage_id: Optional[str] = None,
        iteration: int = 0,
        current_profit: float = 0.0,
    ) -> Tuple[List[List[int]], int, int]:
        """Select and apply one destroy + one repair operator.

        Repair operators are selected via roulette-wheel (Eq. 20); clean and
        noisy variants are independent slots where the engine uses them.

        Args:
            current_routes: The current solution routes.
            trace: Optional execution trace to record operator events.
            stage_id: Parent stage event id on the trace.
            iteration: Current iteration index.
            current_profit: Objective profit before operator application.

        Returns:
            Tuple containing (new_routes, destroy_idx, repair_idx).
        """
        d_idx = self.select_operator(self.destroy_weights)
        r_idx = self.select_operator(self.repair_weights)
        destroy_op = self.destroy_ops[d_idx]
        repair_op = self.repair_ops[r_idx]

        current_n_nodes = sum(len(route) for route in current_routes)
        if current_n_nodes == 0:
            n_remove = 0
        else:
            lower_bound, upper_bound = self._removal_bounds(current_n_nodes)
            n_remove = self.random.randint(lower_bound, upper_bound)

        # Noise value is only computed when a noisy repair slot was selected.
        # The adaptive weight mechanism controls how often noisy slots are chosen.
        noise_val = self._get_noise() if self.repair_noisy[r_idx] else 0.0

        ruined_routes, removed = destroy_op(self._snapshot_routes(current_routes), n_remove)
        if trace is not None:
            destroy_name = (
                self.destroy_names[d_idx]
                if hasattr(self, "destroy_names") and d_idx < len(self.destroy_names)
                else getattr(destroy_op, "__name__", f"destroy_{d_idx}")
            )
            ruined_cost = self.calculate_cost(ruined_routes)
            ruined_rev = sum(self.wastes.get(node_idx, 0) * self.R for route in ruined_routes for node_idx in route)
            ruined_profit = ruined_rev - ruined_cost
            trace.add(
                name=destroy_name,
                kind="operator",
                parent_id=stage_id,
                changed_solution=bool(removed),
                arguments={"iteration": iteration, "n_remove": n_remove, "removed": [int(x) for x in removed]},
                before=solution_record(current_routes, objective=current_profit),
                after=solution_record(ruined_routes, objective=ruined_profit),
            )
        else:
            ruined_profit = 0.0

        new_routes = repair_op(ruined_routes, removed, noise_val)
        if trace is not None:
            repair_name = (
                self.repair_names[r_idx]
                if hasattr(self, "repair_names") and r_idx < len(self.repair_names)
                else getattr(repair_op, "__name__", f"repair_{r_idx}")
            )
            new_cost = self.calculate_cost(new_routes)
            new_rev = sum(self.wastes.get(node_idx, 0) * self.R for route in new_routes for node_idx in route)
            new_profit = new_rev - new_cost
            trace.add(
                name=repair_name,
                kind="operator",
                parent_id=stage_id,
                changed_solution=(new_routes != ruined_routes),
                arguments={"iteration": iteration, "noise": float(noise_val)},
                before=solution_record(ruined_routes, objective=ruined_profit),
                after=solution_record(new_routes, objective=new_profit),
            )

        return new_routes, d_idx, r_idx

    def _update_weights(self, d_idx: int, r_idx: int, score: float) -> None:
        """Update operator weights via the engine's weight-update strategy.

        Args:
            d_idx: Index of the destroy operator used.
            r_idx: Index of the repair operator used.
            score: Score earned by this iteration's operators.
        """
        self._weights.update(d_idx, r_idx, score)

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(
        self,
        initial_solution: Optional[List[List[int]]] = None,
        trace: Optional[Any] = None,
        parent_id: Optional[str] = None,
    ) -> Tuple[List[List[int]], float, float]:
        """Run the LNS loop.

        Args:
            initial_solution: Optional starting solution. If None, the constructive
                heuristic builds one.
            trace: Optional ExecutionTrace to record events into.
            parent_id: Optional parent event id for trace events.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        effective_trace = (
            trace
            if trace is not None
            else getattr(self, "trace", None) or getattr(getattr(self, "problem", None), "trace", None)
        )
        effective_parent = (
            parent_id
            if parent_id is not None
            else getattr(self, "trace_parent_id", None) or getattr(getattr(self, "problem", None), "trace_parent_id", None)
        )

        stage_id: Optional[str] = None
        if effective_trace is not None:
            if effective_parent is None and effective_trace.events:
                effective_parent = effective_trace.events[0]["id"]
            if effective_parent is not None:
                matches = [e for e in effective_trace.events if e["id"] == effective_parent]
                if matches and matches[0]["kind"] == "stage":
                    stage_id = effective_parent
                else:
                    policy_name = getattr(self, "policy_name", "alns")
                    stage_id = effective_trace.add(name=policy_name, kind="stage", parent_id=effective_parent)
            else:
                input_id = effective_trace.add(name="instance", kind="input")
                policy_name = getattr(self, "policy_name", "alns")
                stage_id = effective_trace.add(name=policy_name, kind="stage", parent_id=input_id)

        start_time = time.perf_counter()
        current_routes, best_routes, best_profit, best_cost = self._initialize_solve(initial_solution)
        current_profit = best_profit

        if effective_trace is not None:
            effective_trace.add(
                name="initial_solution",
                kind="operator",
                parent_id=stage_id,
                changed_solution=True,
                arguments={},
                before=solution_record([], objective=0.0),
                after=solution_record(best_routes, objective=best_profit),
            )
            effective_trace.add(
                name="incumbent",
                kind="incumbent",
                parent_id=stage_id,
                changed_solution=True,
                arguments={"profit": float(best_profit), "cost": float(best_cost)},
                before=solution_record([], objective=0.0),
                after=solution_record(best_routes, objective=best_profit),
            )

        self._prepare_search(best_routes, current_profit)

        for _it in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.perf_counter() - start_time > self.params.time_limit:
                break

            new_routes, d_idx, r_idx = self._select_and_apply_operators(
                current_routes,
                trace=effective_trace,
                stage_id=stage_id,
                iteration=_it,
                current_profit=current_profit,
            )
            new_cost = self.calculate_cost(new_routes)
            new_rev = sum(self.wastes.get(node_idx, 0) * self.R for route in new_routes for node_idx in route)
            new_profit = new_rev - new_cost

            new_hash = self._hash_solution(new_routes)
            is_new_solution = self._track_and_check_new(new_hash)

            events_len_before = len(effective_trace.events) if effective_trace is not None else 0
            accepted = self._accept_move(
                current_profit,
                new_profit,
                best_profit,
                _it,
                trace=effective_trace,
                parent_id=stage_id,
                current_routes=current_routes,
                candidate_routes=new_routes,
            )
            if effective_trace is not None and len(effective_trace.events) == events_len_before:
                crit_name = "acceptance"
                if hasattr(self, "cfg") and hasattr(self.cfg, "acceptance_criterion"):
                    ac = self.cfg.acceptance_criterion
                    crit_name = ac if isinstance(ac, str) else getattr(ac, "method", "acceptance")
                effective_trace.add(
                    name=str(crit_name),
                    kind="operator",
                    parent_id=stage_id,
                    changed_solution=accepted,
                    arguments={
                        "accepted": accepted,
                        "iteration": _it,
                        "current_profit": float(current_profit),
                        "candidate_profit": float(new_profit),
                        "delta": float(new_profit - current_profit),
                    },
                    before=solution_record(current_routes, objective=current_profit),
                    after=solution_record(
                        new_routes if accepted else current_routes,
                        objective=new_profit if accepted else current_profit,
                    ),
                )

            score = self._scoring(new_profit, best_profit, current_profit, accepted, is_new_solution)

            if new_profit > best_profit + 1e-6:
                if effective_trace is not None:
                    effective_trace.add(
                        name="incumbent",
                        kind="incumbent",
                        parent_id=stage_id,
                        changed_solution=True,
                        arguments={"iteration": _it, "profit": float(new_profit), "cost": float(new_cost)},
                        before=solution_record(best_routes, objective=best_profit),
                        after=solution_record(new_routes, objective=new_profit),
                    )
                best_routes = self._snapshot_routes(new_routes)
                best_profit = new_profit
                best_cost = new_cost

            if accepted:
                self._track_accepted(new_hash)
                current_routes = new_routes
                current_profit = new_profit

            self._update_weights(d_idx, r_idx, score)

            self._extra_iteration_end(
                iteration=_it,
                d_idx=d_idx,
                r_idx=r_idx,
                score=score,
                new_profit=new_profit,
                current_profit=current_profit,
                best_profit=best_profit,
                accepted=accepted,
            )

            self._record_iteration(
                iteration=_it,
                d_idx=d_idx,
                r_idx=r_idx,
                best_profit=best_profit,
                current_profit=current_profit,
                accepted=accepted,
                score=score,
            )

        if effective_trace is not None:
            effective_trace.set_final(best_routes, objective=best_profit)

        return best_routes, best_profit, best_cost
