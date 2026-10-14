"""
Scenario-Consistent Branching.

Attributes:
    ScenarioConsistentBranching: The Scenario Consistent Branching algorithm implementation.

Examples:
    >>> from .scenario_branching import ScenarioConsistentBranching
    >>> scenario_branching = ScenarioConsistentBranching()
    >>> scenario_branching.calculate_consensus()
    >>> scenario_branching.select_branching_variable()
"""

from typing import Any, Dict, List


class ScenarioConsistentBranching:
    """
    Dynamic branching rule that leverages the optimal solutions of individual
    scenario subproblems to select variables with the strongest cross-scenario
    consensus.

    Motivation
    ----------
    In stochastic programming B&B, a natural branching heuristic is to branch
    on the fractional variable that most scenarios "agree on" when each is
    solved deterministically.  A variable with consensus near 1 is almost
    always set to 1 across scenarios; near 0 it is almost never selected.
    Either extreme implies a high-confidence branch.  A consensus near 0.5
    signals genuine ambiguity and is branched on last.

    Branching Score
    ---------------
    For variable g_i:

        consensus(g_i) = |{ξ : g_i^ξ = 1}| / |Ξ|

        score(g_i) = |consensus(g_i) − 0.5|

    The variable maximising this score is selected; it corresponds to the
    variable with the least ambiguity across the scenario tree.

    Horizon Adaptivity
    ------------------
    The commented-out threshold logic in ``select_branching_variable`` sketches
    an adaptive variant where the minimum acceptable consensus is decayed as
    the horizon shortens:

        threshold(d) = base_threshold · (days_remaining / initial_horizon)

    This allows looser consensus requirements near the end of the horizon,
    where the stochastic structure collapses and all scenarios converge.
    The current implementation uses the pure score-maximisation variant,
    which is equivalent to setting threshold = 0.

    Attributes:
    ----------
    base_threshold : float
        Reference consensus threshold used as the anchor for horizon-adaptive
        decay.  Currently serves as a documentation reference; the active
        branching rule is threshold-free.
    """

    def __init__(self, base_threshold: float = 0.95):
        """
        Args:
            base_threshold: Reference threshold for horizon-adaptive decay.
                A variable with |consensus − 0.5| ≥ (base_threshold − 0.5)
                is considered strongly agreed upon.  Used in the adaptive
                variant; the current implementation is threshold-free.
        """
        self.base_threshold = base_threshold

    def calculate_consensus(
        self,
        var_id: int,
        scenario_tree: Any,
        deterministic_scenario_solutions: Dict[int, Dict[int, int]],
    ) -> float:
        """
        Compute the fraction of scenarios in which variable ``var_id`` is set to 1.

            consensus(g_i) = |{ξ : g_i^ξ = 1}| / |Ξ|

        A value of 1.0 means all scenarios include this variable (strong
        positive consensus); 0.0 means none do (strong negative consensus);
        0.5 is returned as the uninformative default when no solutions are
        available.

        Args:
            var_id: Identifier of the binary variable (e.g. route or bin
                assignment) whose consensus is to be measured.
            scenario_tree: ScenarioTree providing the scenario structure.
                Currently unused but available for probability-weighted
                extensions (replace uniform count with Σ_ξ p_ξ · g_i^ξ).
            deterministic_scenario_solutions: Mapping {scenario_id: {var_id:
                binary_value}} of per-scenario optimal solutions obtained by
                solving each scenario independently (with its deterministic
                prize realization).

        Returns:
            Consensus value in [0, 1].  Returns 0.5 if
            ``deterministic_scenario_solutions`` is empty.
        """
        if not deterministic_scenario_solutions:
            return 0.5

        total_scenarios = len(deterministic_scenario_solutions)
        visited_count = sum(1 for sol in deterministic_scenario_solutions.values() if sol.get(var_id, 0) == 1)

        return visited_count / total_scenarios if total_scenarios > 0 else 0.5

    def select_branching_variable(
        self,
        fractional_vars: List[Any],
        scenario_tree: Any,
        deterministic_solutions: Dict[int, Dict[int, int]],
        days_remaining: int,
        initial_horizon: int,
    ) -> Any:
        """
        Select the fractional variable to branch on by maximising consensus score.

        Iterates over all fractional variables in the current master LP and
        returns the one for which |consensus(g_i) − 0.5| is largest, i.e. the
        variable whose value is most agreed upon (in either direction) by the
        individual-scenario optimal solutions.

        Ties are broken implicitly in favour of the variable encountered first
        in ``fractional_vars``.

        Args:
            fractional_vars: List of variable objects with a ``.id`` attribute,
                representing variables that are fractional (0 < λ < 1) in the
                current master LP relaxation.
            scenario_tree: ScenarioTree; passed through to ``calculate_consensus``
                for potential probability-weighted extensions.
            deterministic_solutions: Per-scenario optimal solutions
                {scenario_id: {var_id: 0 or 1}} used to compute consensus.
            days_remaining: Number of planning days remaining in the horizon.
                Available for horizon-adaptive threshold tuning (currently
                unused in the active threshold-free variant).
            initial_horizon: Full planning horizon length T.  Used as the
                normalising denominator in horizon-adaptive threshold decay
                (currently unused).

        Returns:
            The variable object from ``fractional_vars`` with the highest
            |consensus − 0.5| score, or ``None`` if the list is empty.
        """
        best_var = None
        best_score = -1.0

        for var in fractional_vars:
            consensus = self.calculate_consensus(var.id, scenario_tree, deterministic_solutions)
            score = abs(consensus - 0.5)

            if score > best_score:
                best_score = score
                best_var = var

        return best_var
