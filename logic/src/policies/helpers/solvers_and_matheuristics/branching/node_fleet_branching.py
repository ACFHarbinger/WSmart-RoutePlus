"""node fleet branching definitions relocated from strategies."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import (
    FleetSizeBranchingConstraint,
    NodeVisitationBranchingConstraint,
    RyanFosterBranchingConstraint,
)
from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode


class RyanFosterBranching:
    """
    Ryan-Foster branching strategy for Set Partitioning Problems.

    Identifies a node pair (r, s) that are "fractionally co-occurring" in
    the current LP solution and partitions the search space to eliminate
    this fractional state.

    Mathematical Basis:
    -------------------
    Based on Proposition 1 of Ryan and Foster (1981):
    If a Set Partitioning Problem has a fractional solution λ, there must
    exist two nodes r and s such that the set of routes visiting both
    r and s has a fractional sum:
        0 < ∑_{k: r ∈ route_k, s ∈ route_k} λ_k < 1

    Branching Rule:
    ---------------
    - Left Branch: r and s MUST be in the same route (∑_{k: r,s ∈ route_k} λ_k = 1).
      Any route visiting only one of the pair is disabled.
    - Right Branch: r and s MUST NOT be in the same route (∑_{k: r,s ∈ route_k} λ_k = 0).
      Any route visiting both nodes is disabled.

    Theoretical Rationale for VRPP:
    -------------------------------
    Ryan-Foster branching is polyhedrally stronger than simple edge-based
    branching for VRPs with identical vehicles (anonymous fleet). By
    constraining node pairs, the search space is partitioned without
    inducing the massive symmetry issues of arc-fixing.

    Attributes:
        find_branching_pair: Static method to identify the best node pair to branch on.
        create_child_nodes: Static method to generate child B&B nodes.

    Reference: Ryan & Foster (1981), Proposition 1.
    """

    @staticmethod
    def find_branching_pair(
        routes: List[Route],
        route_values: Dict[int, float],
        mandatory_nodes: Set[int],
        tol: float = 1e-5,
    ) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        Find the node pair (r, s) whose fractional co-occurrence is closest to 0.5.

        Ryan & Foster (1981) Proposition 1 guarantees that for any fractional SPP
        solution there exists such a pair.  Among all fractional pairs we select the
        one with ``|together_sum − round(together_sum)|`` maximised (i.e. closest to
        0.5), which produces the strongest symmetry-breaking branch (equal LP bound
        reduction on both children) and leads to fewer B&B nodes overall.

        The previous implementation returned the *first* fractional pair encountered
        (dependent on arbitrary dict-iteration order), which could produce very
        lopsided branches.  This version scores all candidates and returns the best.

        Args:
            routes: All routes in the master problem.
            route_values: Current LP solution {route_index: λ_k}.
            mandatory_nodes: Set of mandatory nodes needing exact partitioning.
            tol: Integrality tolerance.

        Returns:
            ((node_r, node_s), together_sum) to branch on, or None if integer.
        """
        best_pair: Optional[Tuple[int, int]] = None
        best_together_sum: float = 0.0
        best_frac: float = -1.0

        candidate_nodes: Set[int] = set()
        for idx, val in route_values.items():
            if abs(val - round(val)) > tol:
                candidate_nodes.update(routes[idx].node_coverage)

        candidate_list = sorted(candidate_nodes)

        for i, r in enumerate(candidate_list):
            for s in candidate_list[i + 1 :]:
                together_sum = sum(
                    v
                    for idx, v in route_values.items()
                    if r in routes[idx].node_coverage and s in routes[idx].node_coverage
                )
                frac = abs(together_sum - round(together_sum))
                if tol < frac < 1.0 - tol and frac > best_frac:
                    best_frac = frac
                    best_pair = (r, s)
                    best_together_sum = together_sum

        if best_pair is None:
            return None
        return best_pair, best_together_sum

    @staticmethod
    def create_child_nodes(
        parent: "BranchNode",
        node_r: int,
        node_s: int,
        together_sum: float = 0.5,
    ) -> Tuple["BranchNode", "BranchNode"]:
        """
        Create together (left) and separate (right) child nodes.

        Args:
            parent: The node being branched.
            node_r: First node in the pair.
            node_s: Second node in the pair.
            together_sum: Fractional co-occurrence value for tie-breaking.

        Returns:
            (left_child, right_child)
        """
        from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode

        hint = parent.lp_bound if parent.lp_bound is not None else 0.0
        # Separation branch gets small penalty
        right_hint = hint - (together_sum * 1e-4)

        left = BranchNode(
            constraints=[RyanFosterBranchingConstraint(node_r, node_s, together=True)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=hint,
        )
        right = BranchNode(
            constraints=[RyanFosterBranchingConstraint(node_r, node_s, together=False)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=right_hint,
        )

        return left, right


class FleetSizeBranching:
    """Logic for branching on the total fleet size.

    Attributes:
        find_fleet_branching: Static method to check if fleet usage is fractional.
        create_child_nodes: Static method to generate child B&B nodes.
    """

    @staticmethod
    def find_fleet_branching(
        route_values: Dict[int, float],
        tol: float = 1e-4,
    ) -> Optional[float]:
        """Checks if the total fleet usage (sum of lambdas) is fractional.

        Args:
            route_values: Current LP solution {route_index: λ_k}.
            tol: Integrality tolerance.

        Returns:
            Optional[float]: Total fleet usage if fractional, else None.
        """
        fleet_usage = sum(route_values.values())
        if abs(fleet_usage - round(fleet_usage)) > tol:
            return fleet_usage
        return None

    @staticmethod
    def create_child_nodes(parent: "BranchNode", fleet_usage: float) -> Tuple["BranchNode", "BranchNode"]:
        """Create floor (lower branch) and ceiling (upper branch) child nodes.

        Args:
            parent: The node being branched.
            fleet_usage: Current fractional fleet usage.

        Returns:
            (lower_branch, upper_branch) child nodes.
        """

        from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode

        floor = math.floor(fleet_usage + 1e-6)
        ceil = math.ceil(fleet_usage - 1e-6)
        lower_branch = BranchNode(
            constraints=[FleetSizeBranchingConstraint(floor, is_upper=True)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
        )
        upper_branch = BranchNode(
            constraints=[FleetSizeBranchingConstraint(ceil, is_upper=False)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
        )
        return lower_branch, upper_branch


class NodeVisitationBranching:
    """Logic for branching on optional node visitation.

    Attributes:
        find_node_branching: Static method to find fractional node visitation.
        create_child_nodes: Static method to generate child B&B nodes.
    """

    @staticmethod
    def find_node_branching(
        routes: List[Route],
        route_values: Dict[int, float],
        optional_nodes: Set[int],
        tol: float = 1e-4,
    ) -> Optional[Tuple[int, float]]:
        """Finds an optional node with fractional visitation.

        Args:
            routes: Column pool.
            route_values: Current LP solution {route_index: λ_k}.
            optional_nodes: Indices of optional customers.
            tol: Integrality tolerance.

        Returns:
            Optional[Tuple[int, float]]: (node_index, visitation_value) if fractional, else None.
        """
        for node in sorted(optional_nodes):
            visitation = sum(val for idx, val in route_values.items() if node in routes[idx].node_coverage)
            if tol < visitation < 1.0 - tol:
                return node, visitation
        return None

    @staticmethod
    def create_child_nodes(parent: "BranchNode", node: int, visitation: float) -> Tuple["BranchNode", "BranchNode"]:
        """Create v_i = 0 and v_i = 1 child nodes.

        Args:
            parent: The node being branched.
            node: Index of the optional customer node.
            visitation: Current fractional visitation value.

        Returns:
            (left_child, right_child) child nodes.
        """

        from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode

        left = BranchNode(
            constraints=[NodeVisitationBranchingConstraint(node, forced=False)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
        )
        right = BranchNode(
            constraints=[NodeVisitationBranchingConstraint(node, forced=True)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
        )
        return left, right
