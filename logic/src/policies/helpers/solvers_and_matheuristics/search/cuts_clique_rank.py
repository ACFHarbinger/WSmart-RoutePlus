"""cuts clique rank definitions relocated from cuts_subset_row."""

import itertools
from typing import Dict, List, Tuple

import gurobipy as gp

from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import VRPPMasterProblem
from logic.src.policies.helpers.solvers_and_matheuristics.search.cut_engine_base import CuttingPlaneEngine
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel


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
