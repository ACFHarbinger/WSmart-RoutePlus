"""mtz branching methods relocated from mtz."""

from typing import Any, List, Optional, Tuple

from gurobipy import GRB

from logic.src.policies.helpers.solvers_and_matheuristics.common.node import Node


class MTZBranchingMixin:
    """Relocated mtz branching methods."""

    def _evaluate_node(self, node: Node) -> float:
        """
        Evaluate a search node by updating variable bounds and solving the LP relaxation.

        This method reuses the persistent self.model by:
        1. Resetting all variable bounds to [0.0, 1.0]
        2. Applying the fixed variable values from the node
        3. Solving the LP relaxation
        4. Returning the objective value

        Args:
            node: The current search node containing fixed variable states.

        Returns:
            The LP relaxation objective value, or -inf if infeasible.
        """
        # Step 1: Reset all variable bounds to [0.0, 1.0]
        for (_i, _j), var in self.x.items():
            var.LB = 0.0
            var.UB = 1.0

        for i, var in self.y.items():
            lb = 1.0 if i in self.mandatory_indices else 0.0
            var.LB = lb
            var.UB = 1.0

        # Reset u (MTZ load) variable bounds to initialization values
        for i, var in self.u.items():
            var.LB = self.wastes.get(i, 0.0)
            var.UB = self.capacity

        # Step 2: Apply fixed variable values from the node
        for (i, j), val in node.fixed_x.items():
            self.x[i, j].LB = float(val)
            self.x[i, j].UB = float(val)

        for i, val in node.fixed_y.items():
            self.y[i].LB = float(val)
            self.y[i].UB = float(val)

        # Step 3: Solve the LP relaxation
        self.model.optimize()  # type: ignore[union-attr]

        # Step 4: Return the objective value
        if self.model.Status == GRB.OPTIMAL:  # type: ignore[union-attr]
            return self.model.ObjVal  # type: ignore[union-attr]
        else:
            return -float("inf")

    def _is_integer(self, val: float, tol: float = 1e-6) -> bool:
        """Heuristic check for integer feasibility of a continuous variable.

        Args:
            val: The continuous variable value to check.
            tol: Absolute tolerance for integer proximity.

        Returns:
            True if the value is within tolerance of its nearest integer.
        """
        return abs(val - round(val)) < tol

    def _get_branching_variable(self) -> Optional[Tuple[str, Any]]:
        """
        Identify the most suitable fractional variable to branch upon.

        Supported strategies:
        - 'strong': True Strong Branching using LP solves.
        - 'most_fractional': Maximize the uncertainty (closest to 0.5).
        - 'least_fractional': Minimize the uncertainty (closest to integer bounds).

        Returns:
            A tuple identifying the variable ('x'|'y', Index) or None if feasible.
        """
        fractional_vars = []

        for i_j, var in self.x.items():
            if not self._is_integer(var.X):
                fractional_vars.append(("x", i_j, var.X))

        # Note: We only collect fractional x variables. Branching on y is redundant
        # because the flow constraints sum_j x[i,j] = y[i] and sum_j x[j,i] = y[i]
        # ensure that if y[i] is fractional, at least one incident x variable
        # must also be fractional. Branching on x directly is more efficient.

        if not fractional_vars:
            return None

        if self.branching_strategy == "strong":
            return self._strong_branching(fractional_vars)

        if self.branching_strategy == "most_fractional":
            return sorted(fractional_vars, key=lambda v: abs(v[2] - 0.5))[0][:2]

        return sorted(fractional_vars, key=lambda v: min(v[2], 1 - v[2]))[0][:2]

    def _strong_branching(self, fractional_vars: List[Tuple]) -> Tuple[str, Any]:
        """
        True Strong Branching: choose variable that causes maximum bound degradation.

        For each candidate variable:
        1. Fix the variable to 0.0, solve LP, record objective z_down
        2. Fix the variable to 1.0, solve LP, record objective z_up
        3. Restore the variable bounds to [0.0, 1.0]

        The score is computed as: min(root_obj - z_down, root_obj - z_up)
        This measures the worst-case degradation in the objective bound.

        Args:
            fractional_vars: Candidate variables (type, index, value).

        Returns:
            The variable (type, index) that yielded the highest score.
        """
        # Limit the number of candidates for performance
        candidates = sorted(fractional_vars, key=lambda v: abs(v[2] - 0.5))[: self.strong_branching_limit]

        # Store current objective as the baseline
        root_obj = self.model.ObjVal  # type: ignore[union-attr]

        best_var = candidates[0][:2]
        best_score = -float("inf")

        for var_type, var_idx, _val in candidates:
            # Get the variable reference
            var = self.x[var_idx] if var_type == "x" else self.y[var_idx]

            # Save original bounds
            orig_lb = var.LB
            orig_ub = var.UB

            # Test 1: Fix variable to 0
            var.LB = 0.0
            var.UB = 0.0
            self.model.optimize()  # type: ignore[union-attr]
            z_down = self.model.ObjVal if self.model.Status == GRB.OPTIMAL else -float("inf")  # type: ignore[union-attr]

            # Test 2: Fix variable to 1
            var.LB = 1.0
            var.UB = 1.0
            self.model.optimize()  # type: ignore[union-attr]
            z_up = self.model.ObjVal if self.model.Status == GRB.OPTIMAL else -float("inf")  # type: ignore[union-attr]

            # Restore original bounds
            var.LB = orig_lb
            var.UB = orig_ub

            # Score: degradation from root_obj (for maximization, degradation = decrease)
            # We want the variable that causes the largest degradation in the worst direction
            degradation_down = root_obj - z_down if z_down > -float("inf") else float("inf")
            degradation_up = root_obj - z_up if z_up > -float("inf") else float("inf")
            score = min(degradation_down, degradation_up)

            if score > best_score:
                best_score = score
                best_var = (var_type, var_idx)

        # Restore the node's LP state by re-solving after final bound restoration.
        # This re-solve is correct because all candidates had their bounds
        # restored to orig_lb/orig_ub (the node's bounds) within the loop or
        # immediately after their probes, so the model state now correctly
        # reflects the original node subproblem.
        self.model.optimize()  # type: ignore[union-attr]

        return best_var
