"""
Dive-and-Price Primal Heuristic

Attributes:
    DiveAndPricePrimalHeuristic: A class that implements the primal heuristic for the Benders decomposition.

Example:
    >>> from dive_and_price import DiveAndPricePrimalHeuristic
    >>> heuristic = DiveAndPricePrimalHeuristic()
    >>> heuristic.execute()
"""

from typing import Any, List, Tuple


class DiveAndPricePrimalHeuristic:
    """
    Establishes a strong primal upper bound by greedily fixing the most
    scenario-consistent fractional columns after root-node CG.

    Role in the Pipeline
    --------------------
    After the root-node Column Generation (with or without Progressive
    Hedging) produces a fractional LP solution, this heuristic performs a
    single diving pass to recover a scenario-consistent integer solution.
    The resulting bound acts as the initial upper bound UB for the subsequent
    Benders master–subproblem loop.

    Diving Strategy
    ---------------
    At each step, the column k with the highest composite score is fixed to 1:

        score(k) = λ_k · scenario_consensus(k)

    where λ_k ∈ (0,1) is the current fractional LP value and
    scenario_consensus(k) is the fraction of high-probability leaf scenarios
    in which all bins on route k are simultaneously projected to overflow
    (fill > 90 %).  This jointly prioritises columns that are:

    - Nearly integer (λ_k close to 1), reducing integrality gap exposure.
    - Strongly urged by the scenario tree, ensuring the fixing is robust
      across realisations.

    Infeasibility Recovery
    ----------------------
    The procedure is backtrack-free: if fixing a column causes the RMP to
    become infeasible, the column is unfixed and instead penalised by adding
    a large cost M to its objective coefficient.  This soft penalty makes the
    column unattractive without removing it from the feasible region.
    The penalised column is then excluded from further consideration in the
    current diving pass.

    Attributes:
    ----------
    penalty_M : float
        Big-M soft penalty applied to infeasibility-inducing columns.
        Should be large relative to the expected route profit to effectively
        exclude the column while preserving RMP feasibility.
    """

    def __init__(self, penalty_M: float = 10_000.0):
        """
        Args:
            penalty_M: Soft penalty magnitude M.  Applied to a column's
                objective coefficient when fixing it causes infeasibility.
                Defaults to 10,000 — well above typical route revenues on
                standard VRPP instances.
        """
        self.penalty_M = penalty_M

    def evaluate_scenario_consensus(self, route_nodes: List[int], scenario_tree: Any) -> float:
        """
        Compute the scenario consensus for route k.

        Consensus is defined as the probability-weighted fraction of leaf
        scenarios in which *all* non-depot bins on route k are simultaneously
        near overflow (fill > 90 % of capacity):

            consensus(k) = Σ_{ℓ: all bins near overflow} p_ℓ  /  Σ_ℓ p_ℓ

        A high consensus value indicates that route k is urgently needed in
        most scenario realisations, making it a high-confidence fixing choice.

        Args:
            route_nodes: Ordered node list of route k (depot 0 may appear at
                either end).
            scenario_tree: ScenarioTree used to retrieve leaf scenario nodes
                and their fill-level vectors.

        Returns:
            Consensus value in [0, 1].  Returns 1.0 if the scenario tree has
            no leaves (degenerate / deterministic case).
        """
        leaves = scenario_tree.get_leaves()
        if not leaves:
            return 1.0

        consensus_count = 0.0
        total_prob = 0.0

        for leaf in leaves:
            total_prob += leaf.probability
            # Check if all non-depot bins on the route are near overflow
            all_near_overflow = all(leaf.wastes[n - 1] > 90.0 for n in route_nodes if n != 0)
            if all_near_overflow:
                consensus_count += leaf.probability

        return consensus_count / total_prob if total_prob > 0 else 1.0

    def select_column_to_fix(
        self,
        fractional_columns: List[Tuple[int, float, List[int]]],
        scenario_tree: Any,
    ) -> int:
        """
        Select the fractional column to fix next based on the composite score.

            score(k) = λ_k · scenario_consensus(k)

        Only columns with 0 < λ_k < 1 are eligible.  Fully fixed (λ_k = 1)
        or excluded (λ_k = 0) columns are skipped.

        Args:
            fractional_columns: List of (col_idx, lambda_k, route_nodes)
                tuples representing currently fractional master LP columns.
            scenario_tree: ScenarioTree for consensus evaluation.

        Returns:
            Index (into the RMP) of the column to fix, or −1 if no fractional
            column is eligible.
        """
        best_idx = -1
        best_score = -float("inf")

        for col_idx, lambda_k, route_nodes in fractional_columns:
            if 0.0 < lambda_k < 1.0:
                consensus = self.evaluate_scenario_consensus(route_nodes, scenario_tree)
                score = lambda_k * consensus

                if score > best_score:
                    best_score = score
                    best_idx = col_idx

        return best_idx

    def execute(
        self,
        rmp: Any,
        fractional_columns: List[Tuple[int, float, List[int]]],
        scenario_tree: Any,
    ) -> None:
        """
        Execute the backtrack-free diving procedure over fractional columns.

        At each step:
        1. Select the column with the highest composite score via
           ``select_column_to_fix``.
        2. Attempt to fix λ_k = 1 in the RMP and re-solve.
        3. If feasible: accept the fix and terminate (one column per pass).
        4. If infeasible: unfix the column, apply the big-M soft penalty,
           and continue to the next-best candidate.

        The loop exits when either a feasible fix is found or no remaining
        fractional column is eligible (all have been penalised).

        Args:
            rmp: Restricted Master Problem object exposing:
                - ``fix_column(col_idx, value)``
                - ``unfix_column(col_idx)``
                - ``penalize_column(col_idx, M)``
                - ``resolve() -> str`` (returns ``"FEASIBLE"`` or
                  ``"INFEASIBLE"``).
            fractional_columns: List of (col_idx, lambda_k, route_nodes)
                tuples for currently fractional columns.  Modified in-place:
                infeasibility-inducing columns are removed from the candidate
                list.
            scenario_tree: ScenarioTree passed to ``evaluate_scenario_consensus``.
        """
        while True:
            col_to_fix = self.select_column_to_fix(fractional_columns, scenario_tree)
            if col_to_fix == -1:
                break  # No eligible fractional column remains

            rmp.fix_column(col_to_fix, 1.0)
            status = rmp.resolve()

            if status == "INFEASIBLE":
                # Soft-penalty recovery: penalise and exclude, no backtrack
                rmp.unfix_column(col_to_fix)
                rmp.penalize_column(col_to_fix, self.penalty_M)
                fractional_columns = [c for c in fractional_columns if c[0] != col_to_fix]
            else:
                break  # Feasible fix accepted
