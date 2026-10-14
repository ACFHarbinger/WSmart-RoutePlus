r"""Progressive Hedging (PH) engine for stochastic VRPP.

Progressive Hedging (Rockafellar and Wets, 1991) is a horizontal
decomposition algorithm for stochastic programs with non-anticipativity
constraints. It decomposes the problem into scenario-specific subproblems,
solving each with an augmented objective function that penalizes deviations
from the consensus (mean) solution.

Algorithm (minimization form):
----------------------------
1.  Initialize consensus $\bar{z}^0 = 0$ and duals $w_{\omega}^0 = 0$.
2.  In each iteration $k$:
    a. For each scenario $\omega$:
       Solve subproblem: $\min \,\, f_{\omega}(z_{\omega}) + (w_{\omega}^k)^T z_{\omega} + \frac{\rho}{2} \|z_{\omega} - \bar{z}^k\|^2$
    b. Update consensus: $\bar{z}^{k+1} = \sum_{\omega} P(\omega) z_{\omega}^{k+1}$
    c. Update duals: $w_{\omega}^{k+1} = w_{\omega}^k + \rho (z_{\omega}^{k+1} - \bar{z}^{k+1})$

Linearization for Binary Variables:
----------------------------------
Since $y_{i, \omega} \in \{0, 1\}$, the quadratic term $\|z_{\omega} - \bar{z}\|^2$ is linearized:
    $(y_{i, \omega} - \bar{y}_i)^2 = y_{i, \omega}(1 - 2\bar{y}_i) + \bar{y}_i^2$.
The constant $\bar{y}_i^2$ is ignored in the subproblem optimization.

The augmented cost for node $i$ in scenario $\omega$ is:
    $\Delta Cost_{i, \omega} = w_{i, \omega} + \frac{\rho}{2}(1 - 2\bar{y}_i)$

Attributes:
    ProgressiveHedgingEngine: Core iterative engine for PH decomposition.

Example:
    >>> engine = ProgressiveHedgingEngine(config)
    >>> plan, profit, stats = engine.solve(dist, tree, cap, rev, cost, mand)
"""

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from logic.src.configs.policies import PHConfig
from logic.src.policies.routing.base.factory import RoutingAlgorithmFactory

logger = logging.getLogger(__name__)


class ProgressiveHedgingEngine:
    r"""Core iterative engine for Progressive Hedging decomposition.

    Manages scenario subproblems, performs consensus aggregation, and updates
    dual multipliers based on non-anticipativity residuals.

    Attributes:
        config (PHConfig): Progressive Hedging configuration.
        sub_solver_name (str): Name of the subproblem solver.
        y_consensus (Dict[int, Dict[int, float]]): Consensus solution map.
        w_duals_mp (List[Dict[int, Dict[int, float]]]): Dual multipliers map.
        history (List[Dict[str, float]]): Iteration convergence history.
    """

    def __init__(self, config: PHConfig) -> None:
        """Initialise the PH engine.

        Args:
            config (PHConfig): Progressive Hedging configuration.
        """
        self.config = config
        self.sub_solver_name = config.sub_solver

        # Consensus and dual state
        # y_consensus[day][node]
        self.y_consensus: Dict[int, Dict[int, float]] = {}
        # w_duals[scenario][day][node]
        self.w_duals_mp: List[Dict[int, Dict[int, float]]] = []
        self.history: List[Dict[str, float]] = []

    @staticmethod
    def ensure_route_list(routes: Union[List[int], List[List[int]]]) -> List[List[int]]:
        """Ensures that routes are represented as a list of lists (tours).

        Args:
            routes (Union[List[int], List[List[int]]]): Input routes (flat or nested).

        Returns:
            List[List[int]]: Nested list of tours.
        """
        # 1. Check if already a list of lists
        if routes and isinstance(routes[0], list):
            return routes  # type: ignore[return-value]

        # 2. Handle flattened list [0, 1, 3, 0, 4, 0...]
        # Find all indices where the value is 0
        depot_indices = [i for i, val in enumerate(routes) if val == 0]

        # Create pairs of indices: (0, 3), (3, 5), (5, 7)
        # This ensures the 0 is both the end of one route and start of the next
        nested_routes = [routes[depot_indices[i] : depot_indices[i + 1] + 1] for i in range(len(depot_indices) - 1)]
        return nested_routes  # type: ignore[return-value]

    def solve(  # noqa: C901
        self,
        sub_dist_matrix: np.ndarray,
        scenario_tree: Any,
        capacity: float,
        revenue: float,
        cost_unit: float,
        mandatory_nodes: List[int],
        current_wastes: Optional[Dict[int, float]] = None,
        **kwargs: Any,
    ) -> Tuple[List[List[List[int]]], float, Dict[str, Any]]:
        """Run the Multi-Period Progressive Hedging iterative algorithm.

        Args:
            sub_dist_matrix (np.ndarray): Localised N×N distance matrix.
            scenario_tree (Any): A ScenarioTree object (providing S paths/branches).
            capacity (float): Vehicle capacity.
            revenue (float): Revenue per unit of waste.
            cost_unit (float): Travel cost per distance unit.
            mandatory_nodes (List[int]): Nodes that must be visited on Day 0.
            current_wastes (Optional[Dict[int, float]]): Current waste levels.
            kwargs (Any): Additional parameters for sub-solvers.

        Returns:
            Tuple[List[List[List[int]]], float, Dict[str, Any]]: A tuple containing:
                - full_plan: Collection plan [day][route][node].
                - expected_profit: Expected profit from the solution.
                - stats: Execution statistics and convergence history.
        """
        # 1. Prepare Scenarios from Tree
        # For PH, we typically decompose by full scenario paths.
        # We flatten the tree into S distinct T-period paths.
        from logic.src.pipeline.simulations.bins.prediction import ScenarioTreeNode

        def get_all_paths(node: ScenarioTreeNode, current_path: List[np.ndarray]) -> List[List[np.ndarray]]:
            """Recursively collect all scenarios from a scenario tree."""
            new_path = current_path + [node.wastes]
            if not node.children:
                return [new_path]
            paths = []
            for child in node.children:
                paths.extend(get_all_paths(child, new_path))
            return paths

        all_paths = get_all_paths(scenario_tree.root, [])
        n_scenarios = len(all_paths)
        horizon = scenario_tree.horizon
        num_bins = scenario_tree.num_bins
        node_ids = list(range(1, num_bins + 1))

        if n_scenarios == 0:
            return [], 0.0, {"error": "No scenarios provided"}

        # Determine consensus scope
        scope = getattr(self.config, "consensus_scope", "day_0")
        t_consensus = [0] if scope == "day_0" else list(range(horizon + 1))

        # 2. Initialize State
        # y_consensus[t][node]
        self.y_consensus = {t: {i: 0.0 for i in node_ids} for t in t_consensus}
        # w_duals[s][t][node]
        self.w_duals_mp = [{t: {i: 0.0 for i in node_ids} for t in t_consensus} for _ in range(n_scenarios)]
        probabilities = [1.0 / n_scenarios] * n_scenarios

        # 3. Iterative Loop
        for k in range(self.config.max_iterations):
            scenario_results = []

            # Step 2a: Solve Subproblems
            for s_idx in range(n_scenarios):
                # Calculate augmented node prizes (linearized PH objective)
                # MP prize: node_prizes[day][node]
                node_prizes: Dict[int, Dict[int, float]] = {t: {} for t in range(horizon + 1)}

                for t in range(horizon + 1):
                    wastes_t = all_paths[s_idx][t]
                    for i in node_ids:
                        if t == 0 and current_wastes is not None:
                            base_profit = current_wastes.get(i, 0.0) * revenue
                        else:
                            base_profit = wastes_t[i - 1] * revenue
                        node_prizes[t][i] = base_profit

                        # Apply PH penalties only for days within consensus scope
                        if t in t_consensus:
                            dual = self.w_duals_mp[s_idx][t][i]
                            penalty = (self.config.rho / 2.0) * (1.0 - 2.0 * self.y_consensus[t][i])
                            node_prizes[t][i] -= dual + penalty

                # Dispatch to sub-solver via _run_solver to bypass the execute()
                # interface which requires bins/area context unavailable here.
                # Prizes are already in monetary units, so revenue=1.0 prevents double-scaling.
                sub_solver = RoutingAlgorithmFactory.get_adapter(self.sub_solver_name)
                values: Dict[str, Any] = (
                    asdict(sub_solver.config)
                    if sub_solver.config is not None and is_dataclass(sub_solver.config)
                    else {}
                )
                routes, solver_profit, _dist_cost = sub_solver._run_solver(  # type: ignore[union-attr]
                    sub_dist_matrix=sub_dist_matrix,
                    sub_wastes=node_prizes[0],
                    capacity=capacity,
                    revenue=1.0,
                    cost_unit=cost_unit,
                    values=values,
                    mandatory_nodes=mandatory_nodes,
                )
                full_plan_raw = routes

                # Unflatten and nested-wrap if sub-solver returned a single-day flat tour
                day_0_routes = self.ensure_route_list(full_plan_raw) if isinstance(full_plan_raw, list) else []
                full_plan: List[List[List[int]]] = [day_0_routes] + [[] for _ in range(horizon)]

                # Extract y_hat[t][node] from plan
                y_hat = {t: {i: 0.0 for i in node_ids} for t in t_consensus}
                for t in t_consensus:
                    # Update consensus state y_hat[t][i]
                    day_t_routes: List[List[int]] = full_plan[t]
                    flat_nodes: List[int] = [node for r in day_t_routes for node in r]
                    for i in node_ids:
                        y_hat[t][i] = 1.0 if i in flat_nodes else 0.0

                scenario_results.append((full_plan, solver_profit, y_hat))

            # Step 2b: Update Consensus
            new_consensus = {t: {i: 0.0 for i in node_ids} for t in t_consensus}
            for s_idx in range(n_scenarios):
                _, _, y_hat = scenario_results[s_idx]
                for t in t_consensus:
                    for i in node_ids:
                        new_consensus[t][i] += probabilities[s_idx] * y_hat[t][i]

            # Step 2c: Compute Convergence & Update Duals
            primal_residual = 0.0
            for s_idx in range(n_scenarios):
                _, _, y_hat = scenario_results[s_idx]
                for t in t_consensus:
                    for i in node_ids:
                        residual = y_hat[t][i] - new_consensus[t][i]
                        self.w_duals_mp[s_idx][t][i] += self.config.rho * residual
                        primal_residual += probabilities[s_idx] * (residual**2)

            primal_residual = np.sqrt(primal_residual)
            self.y_consensus = new_consensus

            # Tracking
            avg_profit = sum(res[1] for res in scenario_results) / n_scenarios
            if self.config.verbose:
                logger.info(f"PH MP Iteration {k}: Residual = {primal_residual:.4f}, Avg Profit = {avg_profit:.2f}")

            self.history.append({"iteration": k, "residual": primal_residual, "avg_profit": avg_profit})

            if primal_residual < self.config.convergence_tol:
                break

        # 3. Final Plan Selection
        best_plan: List[List[List[int]]] = scenario_results[0][0]
        expected_profit = sum(res[1] for res in scenario_results) / n_scenarios

        return (
            best_plan,
            expected_profit,
            {"iterations": len(self.history), "final_residual": primal_residual, "history": self.history},
        )
