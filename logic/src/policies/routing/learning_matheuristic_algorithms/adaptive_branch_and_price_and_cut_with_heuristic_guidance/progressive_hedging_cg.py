"""
Progressive Scenario Hedging within Column Generation.

Attributes:
    ProgressiveHedgingCGLoop: The Progressive Hedging algorithm implementation.

Examples:
    >>> from .progressive_hedging_cg import ProgressiveHedgingCGLoop
    >>> progressive_hedging_cg = ProgressiveHedgingCGLoop()
    >>> progressive_hedging_cg.update_x_bar()
    >>> progressive_hedging_cg.compute_dynamic_penalty()
    >>> progressive_hedging_cg.calculate_augmented_reduced_cost()
"""

from typing import Dict, List, Set

import numpy as np


class ProgressiveHedgingCGLoop:
    """
    Non-anticipativity enforcement mechanism for the root-node Column Generation.

    Role in the Pipeline
    --------------------
    Maintains a distinct Restricted Master Problem (RMP) for each scenario ξ
    in the scenario tree and drives all per-scenario solutions toward a common
    non-anticipative consensus by injecting a quadratic proximal penalty into
    each scenario's reduced-cost calculation.

    The update rule follows the standard Progressive Hedging (PH) algorithm
    of Rockafellar & Wets (1991), adapted here to the column-generation
    pricing subproblem:

        c'_{k,ξ}  +=  −ρ_eff · (x_k − x̄_k)²

    where x̄_k is the probability-weighted average of the current scenario
    solutions and ρ_eff is a per-variable dynamic penalty.

    Note on the standard PH form: the full augmented-Lagrangian also includes
    a linear term w_k · (x_k − x̄_k) with a Lagrange multiplier w_k that is
    updated each iteration.  This implementation omits the linear term for
    simplicity; it can be incorporated by maintaining a dual multiplier
    vector updated as w_k ← w_k + ρ · (x_k − x̄_k) after each consensus step.

    Attributes:
    ----------
    num_scenarios : int
        Total number of scenarios |Ξ|; denominator of the x̄ average.
    base_rho : float
        Base penalty coefficient ρ₀.  The effective penalty is scaled
        dynamically per variable: ρ_eff = ρ₀ · (1 + Var_ξ[π_i^ξ]).
    x_bar : Dict[int, float]
        Current consensus variable values x̄_k, keyed by variable (route) id.
        Initialised to an empty dict; populated on the first ``update_x_bar``
        call.
    """

    def __init__(self, num_scenarios: int, base_rho: float = 1.0):
        """
        Args:
            num_scenarios: Total number of scenarios |Ξ| in the scenario tree.
                Used as the denominator when computing the consensus average x̄.
            base_rho: Base augmented-Lagrangian penalty coefficient ρ₀ > 0.
                The effective per-variable penalty is ρ₀ · (1 + Var_ξ[π_i^ξ]).
                Larger values enforce stronger non-anticipativity but may slow
                per-scenario RMP convergence.
        """
        self.num_scenarios = num_scenarios
        self.base_rho = base_rho

        # Consensus variable: x̄_k = (1/|Ξ|) Σ_ξ x_k^ξ
        self.x_bar: Dict[int, float] = {}

    def update_x_bar(self, scenario_solutions: Dict[int, Dict[int, float]]) -> None:
        """
        Recompute the consensus variable x̄ from the current per-scenario solutions.

        For each variable k that appears in at least one scenario solution:

            x̄_k  =  (1/|Ξ|) · Σ_ξ  x_k^ξ

        Variables absent from a scenario solution contribute 0.0 to the sum,
        meaning that |Ξ| in the denominator always equals ``self.num_scenarios``
        regardless of sparsity.

        Args:
            scenario_solutions: Mapping from scenario id to a sparse solution
                dict {var_id: value}.  Values represent the current fractional
                or integral assignment of route k in scenario ξ's RMP.
        """
        all_vars: Set[int] = set()
        for sol in scenario_solutions.values():
            all_vars.update(sol.keys())

        for var_id in all_vars:
            total = sum(sol.get(var_id, 0.0) for sol in scenario_solutions.values())
            self.x_bar[var_id] = total / self.num_scenarios

    def compute_dynamic_penalty(self, var_id: int, scenario_prizes: Dict[int, float]) -> float:
        """
        Compute the dynamic penalty coefficient ρ_eff for a given variable.

        The effective penalty scales with the cross-scenario variance of the
        node prizes associated with the variable's route:

            ρ_eff = ρ₀ · (1 + Var_ξ[π_var_id^ξ])

        Rationale: variables (routes) whose profitability is highly uncertain
        across scenarios receive steeper proximal penalties, incentivising the
        per-scenario RMPs to agree early on these contested columns.

        Args:
            var_id: Variable (route) identifier used as key into
                ``scenario_prizes``.
            scenario_prizes: Mapping {scenario_id: prize value} for the
                variable in question across all scenarios.

        Returns:
            ρ_eff ≥ ρ₀: effective penalty coefficient for this variable.
        """
        if not scenario_prizes:
            return self.base_rho

        prizes = list(scenario_prizes.values())
        variance = np.var(prizes) if len(prizes) > 1 else 0.0

        return float(self.base_rho * (1.0 + variance))

    def calculate_augmented_reduced_cost(
        self,
        route_nodes: List[int],
        dist_matrix: np.ndarray,
        scenario_prizes: Dict[int, float],
        dual_values: Dict[int, float],
        scenario_id: int,
        route_id: int,
        current_x_k: float,
    ) -> float:
        """
        Compute the PH-augmented reduced cost for route k under scenario ξ.

        Standard reduced cost (VRPP maximisation):

            c'_k = −dist(k) + Σ_{i ∈ k} (π_i^{scenario,ξ} − π_i^{dual,ξ})

        PH-augmented form (proximal penalty subtracts from attractiveness):

            c'_{k,ξ} = −dist(k)
                       + Σ_i (π_i^{scenario,ξ} − π_i^{dual,ξ})
                       − ρ_eff · (x_k^ξ − x̄_k)²

        The penalty term discourages scenario ξ from selecting route k when
        its current assignment x_k^ξ deviates significantly from the consensus
        x̄_k, driving convergence across per-scenario RMPs.

        Args:
            route_nodes: Ordered node list including depot (node 0) at both
                ends, e.g. [0, i, j, ..., 0].
            dist_matrix: Square distance matrix; entry [a, b] is the arc
                cost from node a to node b.
            scenario_prizes: Node prizes {node_id: π_i^{scenario,ξ}} for the
                current scenario, as computed by ``ScenarioPrizeEngine``.
            dual_values: RMP dual variables {node_id: π_i^{dual,ξ}} for the
                current scenario's covering constraints.
            scenario_id: Identifier of the current scenario ξ (unused here
                but available for logging / extended implementations).
            route_id: Route identifier k; used as the key into ``x_bar`` and
                as ``var_id`` when computing the dynamic penalty.
            current_x_k: Value of the route-k variable x_k^ξ in the current
                scenario ξ RMP solution (typically fractional in [0,1]).

        Returns:
            Augmented reduced cost c'_{k,ξ}.  A value > 0 indicates the route
            is a candidate improving column for scenario ξ's RMP.
        """
        dist_cost = 0.0
        prize_sum = 0.0

        for i in range(len(route_nodes) - 1):
            dist_cost += dist_matrix[route_nodes[i], route_nodes[i + 1]]

        for n in route_nodes:
            if n != 0:
                prize_sum += scenario_prizes.get(n, 0.0) - dual_values.get(n, 0.0)

        rho = self.compute_dynamic_penalty(route_id, scenario_prizes)
        x_bar_k = self.x_bar.get(route_id, 0.0)
        ph_penalty = rho * ((current_x_k - x_bar_k) ** 2)

        return prize_sum - dist_cost - ph_penalty
