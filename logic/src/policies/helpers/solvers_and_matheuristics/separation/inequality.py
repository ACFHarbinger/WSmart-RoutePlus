"""VRPP Valid Inequality Definitions.

Defines the mathematical structures and facet forms for cutting planes used
to strengthen the LP relaxation of the Vehicle Routing Problem with Profits.

Attributes:
    Inequality: Base class for all valid inequalities in VRPP.
    PCSubtourEliminationCut: SEC specialized for profitable VRP.
    CapacityCut: Rounded Capacity Cut (RCC) for subset coverage.
    CombInequality: Classical comb inequality representation.

Example:
    >>> sec = PCSubtourEliminationCut(node_set={1, 2, 3}, violation=0.5)
    >>> cap = CapacityCut(node_set={1, 2}, total_demand=150, capacity=100, violation=0.2)
"""

from __future__ import annotations

from typing import List, Set

import numpy as np


class Inequality:
    """Base class for valid inequalities.

    Attributes:
        type (str): Identifier for the cut type (e.g., 'SEC', 'COMB').
        violation (float): How much the LP solution violates this cut.
        rhs (float): Right-hand side of inequality.
        node_set (Set[int]): Set of nodes involved in the inequality.

    Example:
        >>> ineq = Inequality("SEC", 0.5)
    """

    def __init__(self, inequality_type: str, degree_of_violation: float):
        """Initialize the base inequality.

        Args:
            inequality_type: Identifier for the cut type (e.g., 'SEC', 'COMB').
            degree_of_violation: How much the LP solution violates this cut.
        """
        self.type = inequality_type
        self.violation = degree_of_violation
        self.rhs = 0.0  # Right-hand side of inequality
        self.node_set: Set[int] = set()  # Set of nodes involved in the inequality

    def __lt__(self, other: Inequality) -> bool:
        """Sort by violation (descending).

        Args:
            other: Other inequality to compare.

        Returns:
            bool: True if this inequality has a greater violation than the other.
        """
        return self.violation > other.violation


class PCSubtourEliminationCut(Inequality):
    r"""Prize-Collecting Subtour Elimination Constraint (PC-SEC).

    For S ⊂ N \ {0} with 2 ≤ |S| ≤ n - 2:
    ∑_{i,j ∈ S} x[i,j] ≤ |S| - 1

    Equivalently (in cut form):
    ∑_{(i,j) ∈ δ(S)} x[i,j] ≥ 2 if S is visited (strengthened for VRPP).

    Attributes:
        node_set (Set[int]): The set of nodes forming the subtour.
        facet_form (str): The specific mathematical form of the facet.
        node_i (int): First endpoint node (if applicable).
        node_j (int): Second endpoint node (if applicable).
        local_only (bool): Whether the cut is only valid in the current branch.

    Example:
        >>> cut = PCSubtourEliminationCut(node_set={1, 2}, violation=0.5)
    """

    def __init__(
        self, node_set: Set[int], violation: float, facet_form: str = "2.1", node_i: int = -1, node_j: int = -1
    ):
        """Initialize a Subtour Elimination Cut.

        Args:
            node_set: The set of nodes forming the subtour.
            violation: The degree of LP violation.
            facet_form: The specific mathematical form of the facet.
            node_i: First endpoint node (if applicable).
            node_j: Second endpoint node (if applicable).
        """
        super().__init__("SEC", violation)
        self.node_set = node_set
        self.facet_form = facet_form
        self.node_i = node_i
        self.node_j = node_j
        self.local_only = False  # Task 7: Default to global, mark True for Form 2.3
        self.rhs = 2.0  # Default for Form 2.1


class CapacityCut(Inequality):
    r"""Capacity Inequality (Rounded Capacity Cut).

    For S ⊂ N \ {0}:
    ∑_{(i,j) ∈ δ(S)} x[i,j] ≥ 2 ⌈demand(S) / Q⌉

    This ensures that enough vehicles enter the set S to serve all demand.

    Attributes:
        total_demand (float): Sum of demands for the node set.
        min_vehicles (int): Minimum vehicles required (lower bound).

    Example:
        >>> cut = CapacityCut(node_set={1, 2}, total_demand=150, capacity=100, violation=0.5)
    """

    def __init__(self, node_set: Set[int], total_demand: float, capacity: float, violation: float):
        """Initialize a Capacity Cut.

        Args:
            node_set: The set of nodes in the component.
            total_demand: Sum of demands for the node set.
            capacity: Vehicle capacity.
            violation: The degree of LP violation.
        """
        super().__init__("CAPACITY", violation)
        self.node_set = node_set
        self.total_demand = total_demand
        self.min_vehicles = int(np.ceil(total_demand / capacity))
        self.rhs = 2.0 * self.min_vehicles


class CombInequality(Inequality):
    """Comb Inequality for TSP/VRP.

    For a handle H and teeth T_1, ..., T_s where |T_j ∩ H| = 1 for all j:
    ∑_{e ∈ E(H)} x_e + ∑_{j=1}^{s} ∑_{e ∈ E(T_j)} x_e ≤ |H| + ∑_{j=1}^{s} |T_j| - (s+1)/2

    Attributes:
        handle (Set[int]): The handle set of the comb.
        teeth (List[Set[int]]): The list of tooth sets.

    Example:
        >>> handle = {1, 2, 3}
        >>> teeth = [{1, 4}, {2, 5}, {3, 6}]
        >>> comb = CombInequality(handle, teeth, violation=0.5)
    """

    def __init__(self, handle: Set[int], teeth: List[Set[int]], violation: float):
        """Initialize a Comb Inequality.

        Args:
            handle: The handle set of the comb.
            teeth: The list of tooth sets.
            violation: The degree of LP violation.
        """
        super().__init__("COMB", violation)
        self.handle = handle
        self.teeth = teeth
        self.node_set = set(handle).union(*(set(t) for t in teeth))
        self.rhs = len(handle) + sum(len(t) for t in teeth) - (len(teeth) + 1) / 2.0
