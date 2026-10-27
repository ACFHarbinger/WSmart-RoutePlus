"""Search-phase mixin for the ILS-RVND-SP solver.

Holds the perturbation operator, the ILS-RVND iteration loop, the two
Subramanian et al. (2013) strategies and the RVND operator factory. These
methods were split out of ``ILSRVNDSPSolver`` in ``ils_rvnd_sp`` and are
combined back into it by inheritance.
"""

import copy
import time
from random import Random
from typing import TYPE_CHECKING, Callable, List, Optional, Set, Tuple

from logic.src.configs.policies.routing.matheuristics.ils_rvnd_sp import ILSRVNDSPConfig
from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory
from logic.src.policies.helpers.local_search.local_search_general import GeneralLocalSearch
from logic.src.policies.routing.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.rvnd import (
    RVND,
)


class ILSRVNDSPSearchMixin:
    """ILS-RVND search loop, perturbation, strategies and RVND operator creation."""

    # Instance state and helpers are assigned on ILSRVNDSPSolver; declared here for the mixin.
    params: ILSRVNDSPConfig
    random: Random
    route_pool: Set[Tuple[int, ...]]
    n_nodes: int
    ls_manager: GeneralLocalSearch
    if TYPE_CHECKING:
        # Implemented on ILSRVNDSPSolver; declared so this mixin can call them.
        def calculate_profit(self, routes: List[List[int]]) -> float: ...
        def build_initial_solution(self) -> List[List[int]]: ...
        def solve_set_partitioning(
            self, pool: Optional[Set[Tuple[int, ...]]] = None
        ) -> Tuple[List[List[int]], float, float]: ...
        def _add_to_pool(self, routes: List[List[int]], pool: Optional[Set[Tuple[int, ...]]] = None) -> None: ...

    def _perturb(self, routes: List[List[int]]) -> List[List[int]]:
        """Random perturbation for ILS as specified in Subramanian et al. (2013).

        Performs 2-5 random moves. Each move is either:
        - Swap(1,1): Swap one random node from r1 with one random node from r2.
        - Shift(1,1): Relocate one random node from r1 to a random position in r2.

        Args:
            routes: Current solution routes to perturb.

        Returns:
            List[List[int]]: New perturbed solution routes.
        """
        new_routes = [r[:] for r in routes]
        n_moves = self.random.randint(2, 5)

        for _ in range(n_moves):
            # Only perturb if we have at least 2 non-empty routes for inter-route moves
            active_routes = [i for i, r in enumerate(new_routes) if len(r) > 0]
            if len(active_routes) < 2:
                # Fallback: if only one route, just do a random intra-swap if possible
                if len(active_routes) == 1:
                    ridx = active_routes[0]
                    if len(new_routes[ridx]) >= 2:
                        i, j = self.random.sample(range(len(new_routes[ridx])), 2)
                        new_routes[ridx][i], new_routes[ridx][j] = new_routes[ridx][j], new_routes[ridx][i]
                continue

            if self.random.random() < 0.5:
                # Swap(1,1)
                r1_idx, r2_idx = self.random.sample(active_routes, 2)
                p1 = self.random.randint(0, len(new_routes[r1_idx]) - 1)
                p2 = self.random.randint(0, len(new_routes[r2_idx]) - 1)
                new_routes[r1_idx][p1], new_routes[r2_idx][p2] = new_routes[r2_idx][p2], new_routes[r1_idx][p1]
            else:
                # Shift(1,1) (Relocate)
                # Note: Shift(1,1) in paper terminology usually means a relocate move
                r1_idx = self.random.choice(active_routes)
                r2_idx = self.random.choice(range(len(new_routes)))  # Can shift to any route
                p1 = self.random.randint(0, len(new_routes[r1_idx]) - 1)
                node = new_routes[r1_idx].pop(p1)
                p2 = self.random.randint(0, len(new_routes[r2_idx])) if new_routes[r2_idx] else 0
                new_routes[r2_idx].insert(p2, node)

        return new_routes

    def run_ils_rvnd(
        self,
        initial_routes: List[List[int]],
        max_iterations: int,
        max_ils_iterations: int,
        target_pool: Set[Tuple[int, ...]],
        tolerance: float,
        start_time: float,
        rvnd: RVND,
    ) -> Tuple[List[List[int]], float]:
        """Execute ILS-RVND metaheuristic iterations.

        Args:
            initial_routes: Starting routes for the ILS loop.
            max_iterations: Outer ILS iteration count.
            max_ils_iterations: Inner ILS iteration count per outer iteration.
            target_pool: Route pool to populate during search.
            tolerance: Relative deviation tolerance for adding routes to pool.
            start_time: Process time at search start, used for time limit checks.
            rvnd: RVND instance providing local search operators.

        Returns:
            Tuple[List[List[int]], float]: Best routes and best profit found.
        """
        best_routes = copy.deepcopy(initial_routes)
        best_profit = self.calculate_profit(best_routes)
        current_profit = best_profit

        # Setup modular acceptance criterion
        self.acceptance = AcceptanceCriterionFactory.build(self.params.acceptance_criterion)
        self.acceptance.setup(best_profit)

        self._add_to_pool(best_routes, target_pool)

        for iteration in range(max_iterations):
            if self.params.time_limit > 0 and (time.perf_counter() - start_time) > self.params.time_limit:
                break

            iter_routes = copy.deepcopy(best_routes)

            for ils_idx in range(max_ils_iterations):
                if self.params.time_limit > 0 and (time.perf_counter() - start_time) > self.params.time_limit:
                    break

                # Perturb current solution and apply RVND intensification
                perturbed = self._perturb(iter_routes)
                ls_routes = rvnd.apply(perturbed)
                ls_profit = self.calculate_profit(ls_routes)

                # Dynamic tolerance pool addition
                if len(ls_routes) > 0 and (best_profit - ls_profit) / abs(best_profit + 1e-9) <= tolerance:
                    self._add_to_pool(ls_routes, target_pool)

                # ILS Acceptance Criterion:
                # Delegate decision (improve only, SA, GD, etc. based on injection)
                is_accepted, _ = self.acceptance.accept(
                    current_obj=current_profit,
                    candidate_obj=ls_profit,
                    iteration=iteration * max_ils_iterations + ils_idx,
                    max_iterations=max_iterations * max_ils_iterations,
                )

                if is_accepted:
                    iter_routes = copy.deepcopy(ls_routes)
                    current_profit = ls_profit

                # Global best update: strict improvement only (always)
                if ls_profit > best_profit + 1e-6:
                    best_routes = copy.deepcopy(ls_routes)
                    best_profit = ls_profit

                # Step criterion
                self.acceptance.step(
                    current_obj=current_profit,
                    candidate_obj=ls_profit,
                    accepted=is_accepted,
                    iteration=iteration * max_ils_iterations + ils_idx,
                )

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                current_profit=current_profit,
                score=3 if best_profit > current_profit else 0,
            )

        return best_routes, best_profit

    def _run_strategy_a(
        self,
        initial_solution: Optional[List[List[int]]],
        start_time: float,
        rvnd: RVND,
    ) -> Tuple[List[List[int]], float]:
        """Execute Strategy A for smaller instances (n <= N).

        Args:
            initial_solution: Optional pre-built starting routes.
            start_time: Process time at search start.
            rvnd: RVND instance providing local search operators.

        Returns:
            Tuple[List[List[int]], float]: Best routes and best profit found.
        """
        global_best_routes = []
        global_best_profit = -float("inf")

        max_restarts = getattr(self.params, "max_restarts", 10)
        max_iter_a = getattr(self.params, "MaxIter_a", 50)
        iter_count = max_iter_a if max_iter_a > 0 else max_restarts

        # MaxIterILS-a = n + 0.5 * v (Subramanian 2013)
        init_routes_for_v = initial_solution if initial_solution else self.build_initial_solution()
        v_approx = max(1, len(init_routes_for_v))
        ils_count = int(self.n_nodes + 0.5 * v_approx)

        tolerance = getattr(self.params, "TDev_a", 0.05)

        for _ in range(iter_count):
            if self.params.time_limit > 0 and (time.perf_counter() - start_time) > self.params.time_limit:
                break
            init_routes = copy.deepcopy(initial_solution) if initial_solution else self.build_initial_solution()
            iter_best_routes, iter_best_profit = self.run_ils_rvnd(
                init_routes, 1, ils_count, self.route_pool, tolerance, start_time, rvnd
            )
            if iter_best_profit > global_best_profit:
                global_best_profit = iter_best_profit
                global_best_routes = iter_best_routes

        if getattr(self.params, "use_set_partitioning", True) and self.route_pool:
            sp_routes, sp_profit, _ = self.solve_set_partitioning(self.route_pool)
            if sp_routes and sp_profit > global_best_profit:
                global_best_routes = sp_routes
                global_best_profit = sp_profit

        return global_best_routes, global_best_profit

    def _run_strategy_b(
        self,
        initial_solution: Optional[List[List[int]]],
        start_time: float,
        rvnd: RVND,
    ) -> Tuple[List[List[int]], float]:
        """Execute Strategy B for larger instances (n > N).

        Args:
            initial_solution: Optional pre-built starting routes.
            start_time: Process time at search start.
            rvnd: RVND instance providing local search operators.

        Returns:
            Tuple[List[List[int]], float]: Best routes and best profit found.
        """
        global_best_routes = []
        global_best_profit = -float("inf")

        max_restarts = getattr(self.params, "max_restarts", 10)
        max_iter_b = getattr(self.params, "MaxIter_b", 100)
        max_iter_ils = getattr(self.params, "max_iter_ils", 50)
        max_ils_b = getattr(self.params, "MaxIterILS_b", 2000)
        a_ratio = getattr(self.params, "A", 11.0)
        tdev_b = getattr(self.params, "TDev_b", 0.005)
        mip_time_limit = getattr(self.params, "mip_time_limit", 60.0)

        iter_count = max_iter_b if max_iter_b > 0 else max_restarts
        ils_count = max_ils_b if max_ils_b > 0 else max_iter_ils

        for _ in range(iter_count):
            if self.params.time_limit > 0 and (time.perf_counter() - start_time) > self.params.time_limit:
                break

            init_routes = copy.deepcopy(initial_solution) if initial_solution else self.build_initial_solution()
            num_vehicles_approx = len(init_routes) if len(init_routes) > 0 else 1

            tolerance = tdev_b if (self.n_nodes / num_vehicles_approx) < a_ratio else 1.0

            temp_pool: Set[Tuple[int, ...]] = set()
            iter_best_routes, iter_best_profit = self.run_ils_rvnd(
                init_routes, 1, ils_count, temp_pool, tolerance, start_time, rvnd
            )

            combined_pool = self.route_pool.union(temp_pool)

            if getattr(self.params, "use_set_partitioning", True) and combined_pool:
                sp_time = time.perf_counter()
                sp_routes, sp_profit, _ = self.solve_set_partitioning(combined_pool)
                sp_duration = time.perf_counter() - sp_time

                if sp_routes and sp_profit > iter_best_profit:
                    iter_best_routes = sp_routes
                    iter_best_profit = sp_profit

                if not (sp_routes and sp_profit > iter_best_profit):
                    temp_pool.clear()

                if (self.n_nodes / num_vehicles_approx) < a_ratio:
                    if sp_duration > mip_time_limit:
                        tolerance -= 0.1 * tdev_b
                    elif sp_duration < 1.0:
                        tolerance += 0.1 * tdev_b

            if iter_best_profit > global_best_profit:
                global_best_profit = iter_best_profit
                global_best_routes = iter_best_routes
                self.route_pool.update(temp_pool)

        return global_best_routes, global_best_profit

    def _create_rvnd_operators(self) -> List[Callable]:
        """
        Create atomic RVND operator wrappers for distinct local search neighborhoods.

        This is critical for proper RVND functionality. Each operator must target
        a specific, distinct neighborhood type. RVND's randomized neighborhood list
        mechanics only work when operators are truly independent.

        Returns:
            List of callable operators, each targeting a specific neighborhood.
        """
        operators = []

        # Define the atomic neighborhoods to explore
        # Each operator will run local search restricted to ONE specific move type
        # Exact 10 neighborhoods specified in Subramanian et al. (2013)
        neighborhoods = [
            "intra_swap",  # Exchange (intra)
            "intra_2opt",  # 2-opt (intra)
            "intra_or_opt",  # Or-opt2 + Or-opt3 (intra)
            "inter_relocate",  # Shift(1,0) (inter)
            "shift_2_0",  # Shift(2,0) (inter) [NEW]
            "inter_swap",  # Swap(1,1) (inter)
            "swap_2_1",  # Swap(2,1) (inter) [NEW]
            "swap_2_2",  # Swap(2,2) (inter) [NEW]
            "cross",  # Cross / Suffix exchange (inter) [NEW]
            "unrouted_insert",  # VRPP specific insertion (extension)
        ]

        # Create a wrapper for each neighborhood
        for neighborhood_type in neighborhoods:

            def make_wrapper(nh_type=neighborhood_type):
                """Factory to capture neighborhood_type in closure."""

                def operator_wrapper(routes: List[List[int]]) -> Tuple[List[List[int]], bool]:
                    """
                    RVND operator that applies ONLY the specified neighborhood.

                    Args:
                        routes: Current solution routes.

                    Returns:
                        (optimized_routes, improvement_found)
                    """
                    if not routes:
                        return routes, False

                    # Calculate initial profit
                    initial_profit = self.calculate_profit(routes)

                    # Run local search with ONLY the specific neighborhood
                    optimized_routes = self.ls_manager.optimize(routes, target_neighborhood=nh_type)

                    # Calculate final profit
                    final_profit = self.calculate_profit(optimized_routes)

                    # Check if this neighborhood found an improvement
                    improved = final_profit > initial_profit + 1e-6

                    return optimized_routes, improved

                return operator_wrapper

            operators.append(make_wrapper())

        return operators
