"""Fleet-cover and lifted cover inequalities (LCI).

Attributes:
    BasicFleetCoverEngine: Fleet-size cover cuts.
    PhysicalCapacityLCIEngine: Lifted covers on physical capacity.
    SaturatedArcLCIEngine: Lifted covers on saturated arcs.

Example:
    >>> BasicFleetCoverEngine(v_model).separate_and_add_cuts(master, max_cuts=10)
"""

from typing import Dict, List, Tuple

import gurobipy as gp

from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import VRPPMasterProblem
from logic.src.policies.helpers.solvers_and_matheuristics.search.cut_engine_base import CuttingPlaneEngine
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel


class BasicFleetCoverEngine(CuttingPlaneEngine):
    """Separates basic cover inequalities for the fleet-size knapsack.

    The fleet-size constraint Σλₖ ≤ K is a 0-1 knapsack where every route
    has weight 1.

    Attributes:
        v_model (VRPPModel): Model for fleet limits.
        epsilon (float): Violation threshold for cut generation.
    """

    def __init__(self, v_model: VRPPModel, epsilon: float = 0.01) -> None:
        """Initializes the Basic Fleet Cover engine.

        Args:
            v_model: VRPP model for fleet limits.
            epsilon: Violation threshold for cut generation.
        """
        self.v_model = v_model
        self.epsilon = epsilon

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """Separate and add basic fleet cover inequalities.

        Identifies a violated cover C (a set of routes with |C| > K whose
        fractional values sum to more than K) and adds the basic cover
        inequality Σ_{k∈C} λ_k ≤ K to the master problem.

        Args:
            master: Master problem instance.
            max_cuts: Maximum number of cuts to add.
            kwargs: Unused (for interface consistency).

        Returns:
            int: Number of cuts added.
        """
        if master.vehicle_limit is None or not master.lambda_vars:
            return 0

        K = float(master.vehicle_limit)

        # 1. Identify fractional routes sorted by λ_k descending.
        active_routes = []
        for i, var in enumerate(master.lambda_vars):
            try:
                val = var.X
                if val > 1e-4:
                    active_routes.append((i, val))
            except (AttributeError, gp.GurobiError):
                continue
        active_routes.sort(key=lambda x: x[1], reverse=True)

        if len(active_routes) <= K:
            return 0

        # 2. Find a minimal cover C: the top K+1 routes by fractional value.
        cover_indices = [idx for idx, _ in active_routes[: int(K) + 1]]
        try:
            sum_val = sum(master.lambda_vars[i].X for i in cover_indices)
        except (AttributeError, gp.GurobiError):
            return 0

        if sum_val <= K + self.epsilon:
            return 0  # No violation — cut would not be useful.

        # 3. Basic uniform lifting
        # Because we have uniform route weights (=1), the exact lifting
        # coefficients alpha_k simplify back to 1.0 across the board.
        # We explicitly calculate it just as alpha_k = 1.0.
        coefficients = {i: 1.0 for i in cover_indices}
        for i in range(len(master.lambda_vars)):
            if i not in cover_indices:
                coefficients[i] = 1.0

        # Represent the cover by its union of visited customer nodes.
        # add_lci_cut keys by node_set; two covers with the same node union
        # are treated as equivalent.
        cover_node_set: List[int] = []
        seen: set = set()
        for i in cover_indices:
            for n in master.routes[i].node_coverage:
                if n not in seen:
                    cover_node_set.append(n)
                    seen.add(n)

        if not cover_node_set:
            return 0

        added = 0
        if master.add_lci_cut(cover_node_set, K, coefficients):
            added += 1
        return added

    def get_name(self) -> str:
        """Returns the identifier for this engine.

        Returns:
            str: The engine name 'fleet_cover'.
        """
        return "fleet_cover"


class PhysicalCapacityLCIEngine(CuttingPlaneEngine):
    """Separates Lifted Cover Inequalities (LCI) over node-visit knapsacks.

    Targets the heterogeneous physical capacity knapsack (Σ_{i ∈ S} q_i y_i ≤ Q),
    where q_i is the demand of customer i and Q is the vehicle capacity.
    These cuts target the visitation variables (y_i) rather than the route
    selection variables (λ_k).

    Mathematical Formulation:
    -------------------------
    1. Identify a minimal cover S such that ∑_{i ∈ S} q_i > Q.
    2. The basic cover inequality is ∑_{i ∈ S} y_i ≤ |S| - 1.
    3. Strengthen the inequality via sequence-independent lifting:
       ∑_{i ∈ S} α_i y_i + ∑_{j ∉ S} α_j y_j ≤ α_0

    Lifting Procedure (Balas 1975):
    -------------------------------
    Calculates exact lifting coefficients (α_i) for nodes based on their
    fractional demands q_i. These node-level coefficients are then aggregated
    into the route variables λ_k for the RMP:
        ∑_{k} (∑_{i ∈ node_coverage_k} α_i) λ_k ≤ α_0

    Operational Logic:
    ------------------
    - Sort nodes by visitation probability y_i to find heuristic covers.
    - Compute lifting coefficients using prefix sums of sorted demands.
    - Map node-alphas to route-coefficients and register the cut in the RMP.

    Attributes:
        v_model (VRPPModel): Model for node demands.
        epsilon (float): Violation threshold for cut generation.
    """

    def __init__(self, v_model: VRPPModel, epsilon: float = 0.01) -> None:
        """Initializes the node-visit LCI engine.

        Args:
            v_model: VRPP model for node demands.
            epsilon: Violation threshold for cut generation.
        """
        self.v_model = v_model
        self.epsilon = epsilon

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:  # noqa: C901
        """Separate and add cuts based on node visitation frequencies.

        Args:
            master: Master problem instance.
            max_cuts: Maximum number of cuts to add.
            kwargs: Unused (for interface consistency).

        Returns:
            int: Number of cuts added.
        """
        if master.model is None or not master.lambda_vars:
            return 0
        # These lifted cover inequalities treat the vehicle capacity Q as one global
        # knapsack, which is valid only for a single vehicle. With several routes a
        # cover can legitimately be visited in full, so the cut would remove optimal
        # solutions (3-node case: 17 instead of 25).
        if getattr(master, "vehicle_limit", None) != 1:
            return 0

        Q = self.v_model.capacity
        y_vals = master.get_node_visitation()

        added = 0
        already_covered: List[List[int]] = []

        # Try multiple cover orderings to find distinct violated LCIs.
        # Paper §6: Gu et al. (1995a) heuristic generates one LCI per saturated structure.
        # We diversify by sorting by three different keys so non-dominated covers are found.
        customer_nodes = [i for i in range(1, self.v_model.n_nodes) if y_vals.get(i, 0.0) > 1e-4]
        orderings = [
            sorted(customer_nodes, key=lambda i: y_vals.get(i, 0.0), reverse=True),  # high visitation first
            sorted(customer_nodes, key=lambda i: self.v_model.wastes.get(i, 0.0), reverse=True),  # heavy demand first
            sorted(
                customer_nodes, key=lambda i: y_vals.get(i, 0.0) * self.v_model.wastes.get(i, 0.0), reverse=True
            ),  # weighted first
        ]

        for nodes_sorted in orderings:
            if added >= max_cuts:
                break

            cover: List[int] = []
            weight_sum = 0.0

            for i in nodes_sorted:
                cover.append(i)
                weight_sum += self.v_model.wastes.get(i, 0.0)
                if weight_sum > Q:
                    break

            if weight_sum <= Q or not cover:
                continue

            # Deduplicate: skip if this cover is too similar to an already-added one
            cover_set = frozenset(cover)
            if any(frozenset(prev) == cover_set for prev in already_covered):
                continue

            cover_y_sum = sum(y_vals.get(i, 0.0) for i in cover)
            if cover_y_sum <= len(cover) - 1 + self.epsilon:
                continue

            # Sequence-independent lifting coefficients (Gu, Nemhauser, Savelsbergh 1995a)
            cover_weights_sorted = sorted([self.v_model.wastes.get(i, 0.0) for i in cover], reverse=True)

            prefix_sums = [0.0] * (len(cover) + 1)
            for idx, w in enumerate(cover_weights_sorted):
                prefix_sums[idx + 1] = prefix_sums[idx] + w

            def compute_lifting_coeff(
                wj: float,
                _ps: List[float] = prefix_sums,
                _cover_len: int = len(cover),
            ) -> float:
                """Exact lifting function α_j = max p s.t. w_j + Σ_{top p} ≥ Q.

                Args:
                    wj: Weight of the node being lifted.
                    _ps: Prefix sums of cover weights.
                    _cover_len: Number of nodes in the cover.

                Returns:
                    float: The computed lifting coefficient.
                """
                # Gu et al. (1995a) exact lifting: α_j = max p such that w_j + Σ_{top p} ≥ Q+ε
                for p in range(_cover_len, -1, -1):
                    if wj + _ps[p] > Q:
                        return float(p)
                return 0.0

            # Node-level lifting coefficients α_i
            alpha_nodes: Dict[int, float] = {}
            for j in range(1, self.v_model.n_nodes):
                if j in cover:
                    alpha_nodes[j] = 1.0
                else:
                    alpha_nodes[j] = compute_lifting_coeff(self.v_model.wastes.get(j, 0.0))

            # Route-level lambda coefficients Σ_{i ∈ route ∩ domain} α_i
            # All routes must be included regardless of current LP value; a route
            # with var.X ≈ 0 now may become active in future LP iterations, and
            # omitting it leaves the LCI constraint under-constrained from the start.
            coefficients: Dict[int, float] = {}
            for k, route in enumerate(master.routes):
                alpha_k = sum(alpha_nodes.get(n, 0.0) for n in route.node_coverage if n != 0)
                if alpha_k > 1e-6:
                    coefficients[k] = alpha_k

            # Pass both route coefficients AND node alphas for pricing dual integration.
            if master.add_lci_cut(cover, float(len(cover) - 1), coefficients, node_alphas=alpha_nodes):
                already_covered.append(cover)
                added += 1

        return added

    def get_name(self) -> str:
        """Returns the identifier for this engine.

        Returns:
            str: The engine name 'physical_lci'.
        """
        return "physical_lci"


class SaturatedArcLCIEngine(CuttingPlaneEngine):
    r"""
    Saturated-Arc Lifted Cover Inequality engine — the primary LCI mechanism
    described in Barnhart, Hane, and Vance (2000) §6.

    Algorithm (paper §6, Gu et al. 1995a):
    -----------------------------------------
    Finding the most violated Lifted Cover Inequality is theoretically NP-hard.
    Consequently, this module employs a heuristic mechanism to identify minimal
    covers derived from saturated arcs in the fractional support graph.

    For each ARC (i, j) that is saturated in the current LP solution
    (i.e., fractional arc flow x_{ij} > capacity − ε):

    1. Identify C(i,j) = {routes k : arc (i,j) ∈ route k}.
    2. If C(i,j) is a cover (Σ_{k ∈ C} demand_k > arc_capacity), form the
       basic cover inequality:
           Σ_{k ∈ C} λ_k ≤ |C| − 1
    3. Lift coefficients for routes NOT in C using the Gu et al. (1995a)
       exact sequence-independent procedure:
           α_k = max p s.t. p routes from \bar{C} can fit with remaining capacity.
    4. Translate to the path-based formulation via delta_{ij}^p indicators.
    5. Add violated LCI to the master.

    Pricing Integration (§4.2):
    In the VRPP RCSPP, when the DP traverses arc (i,j), the reduced cost is
    decreased by α_{ij}^k · γ_{ij}, exactly as in the paper.  Since arcs are
    unit-capacity in VRPP, α_{ij}^k = 1 for all routes in C, and the lifting
    coefficients for routes outside C are computed via the Gu heuristic.

    Arc Capacity in VRPP:
    ---------------------
    In Set Partitioning, each arc can appear in at most 1 selected route in any
    integer solution.  A fractional LP may assign more than 1.0 total flow to an
    arc across multiple fractional routes; this constitutes arc saturation.
    The corresponding capacity knapsack is:
        Σ_{k : arc ∈ route_k} λ_k ≤ 1   (arc capacity = 1 in unit-flow VRP)

    Attributes:
        v_model (VRPPModel): Model for problem data.
        epsilon (float): Violation threshold for cut generation.
    """

    def __init__(self, v_model: VRPPModel, epsilon: float = 0.01) -> None:
        """Initializes the saturated-arc LCI engine.

        Args:
            v_model: VRPP model for problem data.
            epsilon: Violation threshold for cut generation.
        """
        self.v_model = v_model
        self.epsilon = epsilon

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:  # noqa: C901
        """Separate saturated-arc LCIs and add to master.

        Args:
            master: Master problem instance.
            max_cuts: Maximum number of cuts to add.
            kwargs: Unused (for interface consistency).

        Returns:
            int: Number of cuts added.
        """
        if master.model is None or not master.lambda_vars:
            return 0

        # 1. Aggregate arc flows x_{ij} = Σ_k λ_k · δ_{ij}^k
        arc_to_routes: Dict[Tuple[int, int], List[int]] = {}
        route_values: Dict[int, float] = {}

        for ridx, var in enumerate(master.lambda_vars):
            try:
                val = var.X
            except Exception:
                continue
            if val < 1e-6:
                continue
            route_values[ridx] = val
            route = master.routes[ridx]
            full_path = [0] + route.nodes + [0]
            for pos in range(len(full_path) - 1):
                arc = (full_path[pos], full_path[pos + 1])
                arc_to_routes.setdefault(arc, []).append(ridx)

        # 2. Find saturated arcs: Σ_{k : arc ∈ route_k} λ_k > 1.0 − ε
        arc_capacity = 1.0  # unit capacity in VRPP SP
        saturated: List[Tuple[float, Tuple[int, int]]] = []
        for arc, route_indices in arc_to_routes.items():
            flow = sum(route_values.get(k, 0.0) for k in route_indices)
            if flow > arc_capacity - self.epsilon:
                violation = flow - arc_capacity
                saturated.append((violation, arc))

        # Sort by most violated first — focus separation on the tightest constraints
        saturated.sort(reverse=True)

        added = 0
        for _violation, arc in saturated:
            if added >= max_cuts:
                break

            cover_indices = arc_to_routes[arc]
            # With unit arc weights and capacity 1, a cover needs at least two routes
            # (sum of weights > capacity). A single route at full flow is feasible, and
            # the "cover" cut for it would read lambda_k <= 0 and forbid the route.
            if len(cover_indices) < 2:
                continue

            # 3. Basic cover: all routes using this arc form a cover (sum > capacity = 1)
            cover_flow = sum(route_values.get(k, 0.0) for k in cover_indices)
            if cover_flow <= arc_capacity - self.epsilon:
                continue  # Not actually violated

            # 4. Sequence-independent lifting (Gu et al. 1995a)
            # For arc capacity = 1, weights are all 1.0 (each route contributes 1).
            # α_k = 1 for routes in C; for routes NOT in C, α_k = 1 if adding that
            # route saturates the arc with the current cover.
            cover_set_idx = set(cover_indices)
            n_cover = len(cover_indices)
            rhs = float(n_cover - 1)

            # Coefficient for routes IN cover: 1.0
            coefficients: Dict[int, float] = {k: 1.0 for k in cover_indices}

            # Coefficient for routes NOT in cover: α_k ∈ {0, 1}
            # Since arc capacity = 1 and cover already saturates it, any additional route
            # also using this arc has lifting coefficient α_k = 1 (extends the cover).
            for ridx_other in route_values:
                if ridx_other in cover_set_idx:
                    continue
                route_other = master.routes[ridx_other]
                full_path = [0] + route_other.nodes + [0]
                if any(full_path[pos] == arc[0] and full_path[pos + 1] == arc[1] for pos in range(len(full_path) - 1)):
                    coefficients[ridx_other] = 1.0

            # Node-level alphas: α_i = 1 for nodes in routes that use this arc.
            # These allow pricing to penalise arc-covering routes at the node level.
            alpha_nodes: Dict[int, float] = {}
            for k in cover_indices:
                for node in master.routes[k].node_coverage:
                    alpha_nodes[node] = max(alpha_nodes.get(node, 0.0), 1.0)

            # 5. Check current violation before adding
            lhs_val = sum(coefficients.get(k, 0.0) * route_values.get(k, 0.0) for k in coefficients)
            if lhs_val <= rhs + 1e-4:
                continue  # Not violated in fractional solution

            # Use the covered node set as the LCI dedup key.
            # This is semantically correct: the cut constrains routes that share
            # this node cover, and two arcs with identical covered-node sets
            # would generate the same constraint (so skipping is safe).
            cover_node_list = sorted(alpha_nodes.keys())
            if not cover_node_list:
                continue

            # Pass the saturated arc so the pricing DP gates the dual on exact arc
            # traversal (i→j) rather than on any visit to a node in the cover set.
            # This implements Barnhart et al. (2000) §4.2 c'_lm^k = c_lm^k + α·γ
            # without the over-penalisation caused by a node-visit approximation.
            if master.add_lci_cut(
                cover_node_list,
                rhs,
                coefficients,
                node_alphas=alpha_nodes,
                arc=arc,
            ):
                added += 1

        return added

    def get_name(self) -> str:
        """Returns the identifier for this engine.

        Returns:
            str: The engine name 'saturated_arc_lci'.
        """
        return "saturated_arc_lci"
