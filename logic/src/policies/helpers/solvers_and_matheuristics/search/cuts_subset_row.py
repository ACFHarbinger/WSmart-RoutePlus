"""Set-partitioning cuts: subset-row (SRI), limited-memory rank-1 and edge clique.

Attributes:
    SubsetRowCutEngine: 3-SRI separation (Jepsen et al. 2008).
    LimitedMemoryRank1CutEngine: Rank-1 cuts over subsets up to size 5 (Pecin et al. 2017).
    EdgeCliqueCutEngine: Edge clique incompatibility cuts.

Example:
    >>> SubsetRowCutEngine(v_model).separate_and_add_cuts(master, max_cuts=10)
"""

import hashlib
import itertools
from typing import Dict, FrozenSet, List, Set, Tuple

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


class EdgeCliqueCutEngine(CuttingPlaneEngine):
    """Fleet Cover Inequality (Edge-Clique) separation engine.

    This engine separates valid inequalities derived from conflict graphs
    on the route-selection variables. Specifically, it identifies cliques
    in the conflict graph of routes that compete for a shared resource
    (in this case, single-use of a directed arc or a total vehicle count).

    Mathematical Formulation:
    -------------------------
    For a physical edge e = (i, j), let C(e) be the set of all routes in
    the RMP that traverse edge e. The Edge-Clique inequality is:
        ∑_{k ∈ C(e)} λ_k ≤ 1

    Theoretical Context:
    --------------------
    Standard Set Partitioning restricts each node to one visit, but
    Edge-Clique inequalities extend this to the "edges" of the graph.
    They are derived by lifting minimal covers on the arc capacity knapsack,
    forming facets of the resulting knapsack polytope. This implementation
    adapts the Lifting and Cover Inequality (LCI) logic of Balas (1975)
    and Barnhart et al. (2000).

    Heuristic Separation:
    --------------------
    The engine computes the aggregated flow x_{ij} for every edge in the
    current solution. Edges where x_{ij} > 1.0 + ε are candidates for a
    clique cut. For anonymous fleets where route weights are uniform (=1),
    the lifting coefficient for all routes in the clique simplifies to 1.0.

    Operational Logic:
    ------------------
    1. Scan the Restricted Master Problem columns (λ_k).
    2. Build an edge-indexed map of crossing routes.
    3. Identify "oversaturated" edges where fractional routes sum to > 1.0.
    4. Register the clique constraint to prune the fractional space.

    Attributes:
        v_model (VRPPModel): Model for edge indexing.
        capacity (float): Physical capacity of the arc.
        epsilon (float): Violation threshold.
    """

    def __init__(self, v_model: VRPPModel, capacity: float = 1.0, epsilon: float = 0.01) -> None:
        """Initialize the EdgeClique separation engine.

        Args:
            v_model: VRPP model for edge indexing.
            capacity: Physical capacity of the arc.
            epsilon: Violation threshold.
        """
        self.v_model = v_model
        self.capacity = capacity
        self.epsilon = epsilon

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:  # noqa: C901
        """Identify violated LCIs by lifting minimal covers.

        Args:
            master: Master problem instance.
            max_cuts: Maximum number of cuts to add.
            kwargs: Unused (for interface consistency).

        Returns:
            int: Number of cuts added.
        """
        if master.model is None or not master.lambda_vars:
            return 0

        # Edge-clique cuts are globally valid: they depend only on the edge
        # capacity structure, not on branching decisions. Safe for GlobalCutPool.

        # 1. Aggregate edge flow: x_e = Σ_{k} w_{ek} λ_k
        try:
            route_values = {i: var.X for i, var in enumerate(master.lambda_vars) if var.X > 1e-6}
        except (AttributeError, gp.GurobiError):
            return 0
        edge_to_routes: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}  # arc -> list of (route_idx, weight)

        for ridx in route_values:
            route = master.routes[ridx]
            nodes = [0] + route.nodes + [0]
            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i + 1]
                e = tuple(sorted((u, v)))
                # Track weights (crossings) for the knapsack formulation
                # In standard VRP, weight is 1. If route repeats arc, weight > 1.
                found = False
                if e in edge_to_routes:
                    for j, (ridx_existing, w) in enumerate(edge_to_routes[e]):  # type: ignore[index]
                        if ridx_existing == ridx:
                            edge_to_routes[e][j] = (ridx_existing, w + 1)  # type: ignore[index]
                            found = True
                            break
                if not found:
                    edge_to_routes.setdefault(e, []).append((ridx, 1))  # type: ignore[arg-type]

        # 2. Identify saturated edges
        saturated_edges = []
        for e, routes_info in edge_to_routes.items():
            try:
                flow = sum(master.lambda_vars[ridx].X * w for ridx, w in routes_info)
            except (AttributeError, gp.GurobiError):
                flow = 0.0
            if flow > self.capacity + self.epsilon:
                saturated_edges.append((e, routes_info, flow))

        # Sort by flow violation
        saturated_edges.sort(key=lambda x: x[2], reverse=True)

        added_cuts = 0
        for edge_tuple, routes_info, _ in saturated_edges:
            if added_cuts >= max_cuts:
                break

            # 3. Find minimal cover C
            # Sort routes by λ_k descending
            try:
                routes_info.sort(key=lambda x: master.lambda_vars[x[0]].X, reverse=True)
            except (AttributeError, gp.GurobiError):
                continue

            cover_indices = []
            weight_sum = 0
            for idx, w in routes_info:
                cover_indices.append((idx, w))
                weight_sum += w
                if weight_sum > self.capacity:
                    break

            if weight_sum <= self.capacity:
                continue  # Should not happen for saturated arc if all w >= 1

            # 4. O(1) Clique Lifting
            # For Set Partitioning, capacity is 1. Any pair of routes traversing
            # this edge form a clique. The lifting coefficient is uniformly 1.0.
            rhs = 1.0
            coefficients = {ridx: 1.0 for ridx, _ in routes_info}

            # 5. Add to master
            if master.add_edge_clique_cut(edge_tuple[0], edge_tuple[1], coefficients=coefficients, rhs=rhs):
                added_cuts += 1

        return added_cuts

    def get_name(self) -> str:
        """Returns the identifier for this engine.

        Returns:
            str: The engine name 'edge_clique'.
        """
        return "edge_clique"


class LimitedMemoryRank1CutEngine(CuttingPlaneEngine):
    r"""Limited-Memory Rank-1 Cuts for VRPP (Pecin et al. 2017).

    Generalises Subset-Row Inequalities (Jepsen et al. 2008) to subsets of
    up to 5 rows with optimal multipliers and a *memory set* M ⊆ V that
    controls label-dimension explosion in the RCSPP.

    For a subset C ⊆ V with multiplier vector α ∈ (0,1)^|C| and memory set
    M ⊆ V, the limited-memory rank-1 cut is:

        Σ_r ⌊ Σ_{v ∈ C} α_v · a_{vr} ⌋  λ_r  ≤  ⌊ Σ_{v ∈ C} α_v ⌋

    For the standard 3-SRI (|C|=3, α=(1/2,1/2,1/2)):
        Σ_r ⌊(a_{ir} + a_{jr} + a_{kr}) / 2⌋ λ_r  ≤  1

    The memory set M ⊆ V restricts which visits "count" for the DP state in
    the pricing subproblem: the parity bit s_C resets to 0 whenever the path
    leaves M, bounding the label proliferation.

    This implementation enumerates violated 3-SRI and 5-row cuts with the
    (1/2,…,1/2) multiplier, which is optimal for odd-sized C.

    References:
        Pecin, Pessoa, Poggi, Uchoa (2017) Math. Prog. Comput. 9(1):61-100.
        Pecin et al. (2017) Operations Research Letters 45(3):206-209.
        Jepsen, Petersen, Spoorendonk, Pisinger (2008) OR 56(2):497-511.

    Attributes:
        v_model: The VRPP model.
        max_subset_size: The maximum size of the subset.
    """

    def __init__(self, v_model: "VRPPModel", max_subset_size: int = 5):
        """
        Args:
            v_model: The VRPP model.
            max_subset_size: The maximum size of the subset.
        """
        self.v_model = v_model
        self.max_subset_size = min(max_subset_size, 5)

    def separate_and_add_cuts(self, master: "VRPPMasterProblem", max_cuts: int, **kwargs) -> int:
        """
        Add limited memory rank 1 cuts to the master problem.

        Args:
            master: The master problem.
            max_cuts: The maximum number of cuts to add.
            kwargs: Additional arguments.

        Returns:
            int: The number of cuts added.
        """
        if master.model is None or not master.lambda_vars:
            return 0

        routes = master.routes
        try:
            sol = {idx: var.X for idx, var in enumerate(master.lambda_vars) if var.X > 1e-9}
        except (AttributeError, gp.GurobiError):
            return 0

        tol = 1e-4
        added = 0

        # Build node-visit indicator: a[r][v] = 1 if route r visits v
        # For efficiency, only consider nodes with fractional LP visitation
        node_visits = master.get_node_visitation()
        fractional_nodes = sorted(
            [v for v, y in node_visits.items() if tol < y < 1.0 - tol], key=lambda v: abs(node_visits[v] - 0.5)
        )

        # Enumerate 3-subsets (most impactful)
        for size in [3, 5]:
            if added >= max_cuts:
                break
            if size > self.max_subset_size:
                break
            candidates = fractional_nodes[: min(15, len(fractional_nodes))]
            if len(candidates) < size:
                continue

            for subset in itertools.combinations(candidates, size):
                # Compute LHS = Σ_r ⌊(Σ_{v∈C} a_{vr}) / 2⌋ · λ_r
                lhs = 0.0
                for idx, lam in sol.items():
                    if lam < 1e-9:
                        continue
                    visit_count = sum(1 for v in subset if v in routes[idx].node_coverage)
                    lhs += (visit_count // 2) * lam

                rhs = len(subset) // 2
                violation = lhs - rhs
                if violation < tol:
                    continue

                # Add the cut with memory set M = subset (self-referential, most common)
                memory_set = set(subset)
                if hasattr(master, "add_limited_memory_rank1_cut"):
                    success = master.add_limited_memory_rank1_cut(
                        list(subset), rhs, memory_set=memory_set, violation=violation
                    )
                elif hasattr(master, "add_subset_row_cut"):
                    success = master.add_subset_row_cut(frozenset(subset))
                else:
                    success = False

                if success:
                    added += 1
                    if added >= max_cuts:
                        return added

        return added

    def get_name(self) -> str:
        """
        Return the name of the cut engine.

        Args:
            None

        Returns:
            str: The name of the cut engine.
        """
        return "limited_memory_rank1"
