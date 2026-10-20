"""Set-partitioning cuts: subset-row (SRI), limited-memory rank-1 and edge clique.

Attributes:
    SubsetRowCutEngine: 3-SRI separation (Jepsen et al. 2008).
    LimitedMemoryRank1CutEngine: Rank-1 cuts over subsets up to size 5 (Pecin et al. 2017).
    EdgeCliqueCutEngine: Edge clique incompatibility cuts.

Example:
    >>> SubsetRowCutEngine(v_model).separate_and_add_cuts(master, max_cuts=10)

Parts are relocated to cuts_clique_rank.
"""

import hashlib
import itertools
from typing import Dict, FrozenSet, List, Set

import gurobipy as gp
import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import VRPPMasterProblem
from logic.src.policies.helpers.solvers_and_matheuristics.search.cut_engine_base import CuttingPlaneEngine
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel


class SubsetRowCutEngine(CuttingPlaneEngine):
    """Subset-Row Inequality (SRI) separation engine.

    The Subset-Row Inequalities are a class of Chvátal-Gomory rank-1
    inequalities derived from the Set Partitioning Problem (SPP) formulation.

    They are essential for tightening the LP relaxation when solving VRPs
    with column generation.

    Mathematical Formulation (3-SRI):
    ---------------------------------
    For a subset S of exactly |S| = 3 customer nodes, the 3-SRI (with p=2)
    is defined as:
        ∑_{k} ⌊ 1/2 * |S ∩ route_k| ⌋ * λ_k ≤ 1

    Logic:
    If three routes each visit exactly two nodes of the triplet S, the sum
    of their fractional variables λ_k cannot exceed 1.0. If it does (e.g.,
    three routes with λ_k = 0.5), the solution is fractional and violated.

    Theoretical Context:
    --------------------
    SRIs handle the "entanglement" of fractional routes. By penalizing
    routes that share too many nodes in a specified triplet, they force
    the branch-and-bound tree toward integer solutions. This implementation
    follows the separation logic established by Jepsen et al. (2008).

    Separation Methodology:
    -----------------------
    Heuristic search identifies candidate triplets where the sum of visitation
    variables y_i = ∑_{k: i ∈ route_k} λ_k is high. We specifically target
    triplets forming triangles in the fractional support graph.

    Integration with Pricing:
    --------------------------
    Specifically for 3-SRIs, the dual variables π_SRI modify the RCSPP label
    transition. A labeling state must track how many nodes of S it has
    visited. When the count transitions from 1 to 2, the dual value π_SRI
    is subtracted from the path profit (or added to reduced cost).

    Attributes:
        v_model (VRPPModel): Model for node indexing and route tracking.
    """

    def __init__(self, v_model: VRPPModel):
        """Initializes the Subset-Row Cut separation engine.

        Args:
            v_model: VRPP model for node indexing and route tracking.
        """
        self.v_model = v_model

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """Identifies violated 3-Subset-Row Inequalities (SRIs) using structural heuristics.

        Args:
            master: Master problem instance with current fractional solution.
            max_cuts: Maximum number of SRI cuts to add in this separation round.
            kwargs: Additional parameters, such as cut_orthogonality_threshold.

        Returns:
            int: The number of SRI cuts successfully added to the master problem.
        """
        if master.model is None or len(master.routes) < 2:
            return 0

        # Structural SRI Separation (Academic Standard):
        # Instead of a myopic visitation-based heuristic, we identify candidate
        # triplets (i, j, k) that "co-occur" in active fractional columns.
        # This focuses separation on subsets most likely to be violated by
        # the current RMP solution.

        # 1. Identify "fractional" routes (0 < λ_k < 1)
        fractional_routes = []
        for idx, var in enumerate(master.lambda_vars):
            try:
                val = var.X
            except (AttributeError, gp.GurobiError):
                continue
            if 0.01 < val < 0.99:
                fractional_routes.append((master.routes[idx], val))

        if not fractional_routes:
            return 0

        # 2. Build co-occurrence connectivity
        # We look for nodes that appear together in fractional routes.
        # This forms the "structural entanglement" suggested by Jepsen et al. (2008).
        adjacency: Dict[int, Set[int]] = {}
        for route, _ in fractional_routes:
            nodes = list(route.node_coverage)
            for i, j in itertools.combinations(nodes, 2):
                adjacency.setdefault(i, set()).add(j)
                adjacency.setdefault(j, set()).add(i)

        # 3. Candidate Triplets: Nodes forming a triangle or near-triangle in the support graph.
        candidate_subsets: Set[FrozenSet[int]] = set()
        for i, neighbors in adjacency.items():
            if len(neighbors) < 2:
                continue
            for j, k in itertools.combinations(list(neighbors), 2):
                # Triplets that share at least two fractional support edges
                candidate_subsets.add(frozenset([i, j, k]))

        if not candidate_subsets:
            return 0

        added = 0
        # 4. Evaluate violation for structural candidates
        for node_fset in candidate_subsets:
            if added >= max_cuts:
                break
            if self._evaluate_and_add_sri(master, set(node_fset), **kwargs):
                added += 1

        return added

    def _evaluate_and_add_sri(self, master: VRPPMasterProblem, node_set: Set[int], **kwargs) -> bool:
        """Calculate SRI violation and add to master if violated.

        Args:
            master: Master problem instance.
            node_set: Set of nodes in the candidate triplet.
            kwargs: Threshold parameters.

        Returns:
            bool: True if cut was added, False otherwise.
        """
        val = 0.0
        coeff_dict: Dict[str, float] = {}
        threshold = kwargs.get("cut_orthogonality_threshold", 0.8)
        for idx, route in enumerate(master.routes):
            try:
                lam = master.lambda_vars[idx].X
            except (AttributeError, gp.GurobiError):
                continue

            if lam < 1e-6:
                continue

            # Fix 13: Key by content hash, not index, to remain robust against column purging.
            content = ",".join(map(str, route.nodes))
            route_h = hashlib.md5(content.encode()).hexdigest()

            count = len(node_set.intersection(route.node_coverage))
            coeff = count // 2
            if coeff > 0:
                coeff_dict[route_h] = float(coeff)
                val += float(coeff) * lam

        if val > 1.0 + 1e-4:
            # Task 9: Check similarity using content-mapped vectors
            active_sri_vecs = list(master.global_cut_pool.active_sri_vectors.values())
            if not self._is_orthogonal_content(coeff_dict, active_sri_vecs, threshold):
                return False

            return master.add_subset_row_cut(list(node_set))
        return False

    def _is_orthogonal_content(
        self, candidate_dict: Dict[str, float], active_vecs: List[Dict[str, float]], threshold: float = 0.8
    ) -> bool:
        """Content-keyed orthogonality check.

        Args:
            candidate_dict: Coefficients of candidate cut.
            active_vecs: List of existing cut coefficient dicts.
            threshold: Orthogonality threshold.

        Returns:
            bool: True if orthogonal, False otherwise.
        """
        if not active_vecs:
            return True

        norm_c = np.sqrt(sum(v * v for v in candidate_dict.values()))
        if norm_c < 1e-6:
            return False

        for a_dict in active_vecs:
            dot = 0.0
            for h, v in candidate_dict.items():
                if h in a_dict:
                    dot += v * a_dict[h]
            norm_a = np.sqrt(sum(v * v for v in a_dict.values()))
            if norm_a < 1e-6:
                continue
            cos_sim = abs(dot) / (norm_c * norm_a)
            if cos_sim > threshold:
                return False
        return True

    def get_name(self) -> str:
        """Returns the identifier for this engine.

        Returns:
            str: The engine name 'sri'.
        """
        return "sri"
