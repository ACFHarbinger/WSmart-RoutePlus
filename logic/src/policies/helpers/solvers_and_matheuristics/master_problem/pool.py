"""
Global Cut Pool for Branch-and-Price-and-Cut.

Attributes:
    GlobalCutPool: Centralized repository for globally valid inequalities.

Cuts are archived centrally as they are discovered; persistence across B&B
nodes is provided by the single shared master object, which is never rebuilt
mid-search. There is deliberately no replay path: re-injecting pooled cuts into
a rebuilt master was planned but never wired, and the archival half above is
what the engines actually use (e.g. SRI coefficient vectors at separation).

Example:
    >>> pool = GlobalCutPool()
    >>> pool.add_cut("rcc", (frozenset({1, 2}), 2.0))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional, Set, Tuple

if TYPE_CHECKING:
    pass


class GlobalCutPool:
    """
    Centralized repository for globally valid inequalities across B&B nodes.

    Philosophy:
    In BPC, separation is expensive. By pooling valid inequalities (RCC, SRI, SEC 2.1)
    globally, we ensure that a cut discovered in one branch tightens the LP bound
    in sibling and child branches immediately, avoiding redundant separation and
    reducing the total number of B&B nodes explored.

    RCC storage note:
        RCC cuts are stored as (node_set, rhs) pairs so that the original RHS
        (= 2*⌈demand(S)/Q⌉, computed at discovery) is faithfully replayed when
        the cut is re-injected at descendant nodes. Storing only the node set
        and hard-coding rhs=1.0 would produce trivially weak cuts.

        Assumption: Customer demands and vehicle capacities are static throughout
        the B&B tree. If these were dynamic (e.g., stochastic demands handled at
        internal nodes), the RHS of purely node-set-based cuts could change,
        invalidating the global mathematical integrity of this archive.

    Attributes:
        rcc_cuts: Dictionary mapping node sets to their RHS values.
        sri_cuts: Set of node sets with SRI cuts.
        active_sri_vectors: Active SRI coefficient vectors.
        sec_cuts: Set of node sets with SEC Form 2.1 cuts.
        edge_clique_cuts: Set of edges with Edge Clique cuts.
        lci_cuts: Dictionary mapping node sets to LCI cut data.
        lci_arcs: Arcs associated with LCI cuts.
    """

    def __init__(self) -> None:
        """Initializes empty global cut registries.

        Sets up dictionaries and sets for RCC, SRI, SEC, Edge Clique, and LCI cuts.

        Args:
            None

        Returns:
            None
        """
        # RCC: maps node_set -> original rhs (2*ceil(demand/Q))
        self.rcc_cuts: Dict[FrozenSet[int], float] = {}
        self.sri_cuts: Set[FrozenSet[int]] = set()
        self.active_sri_vectors: Dict[FrozenSet[int], Dict[str, float]] = {}
        self.sec_cuts: Set[FrozenSet[int]] = set()  # Form 2.1 (Global)
        self.edge_clique_cuts: Set[Tuple[int, int]] = set()
        # LCI: maps node_set -> (rhs, route_coefficients, node_alphas)
        # node_alphas: per-node lifting coefficients for pricing (Barnhart et al. 2000 §4.2)
        self.lci_cuts: Dict[FrozenSet[int], Tuple[float, Dict[int, float], Dict[int, float]]] = {}
        # Optional source arc (i, j) for arc-saturation LCI (SaturatedArcLCIEngine).
        # When set, the pricing dual fires ONLY when the DP traverses that specific arc,
        # not on any visit to a node in the cover set.  None for node/capacity LCI.
        self.lci_arcs: Dict[FrozenSet[int], Optional[Tuple[int, int]]] = {}
        # Multistar: maps node_set -> route_coefficients {route_idx: -a_k}
        # Duals γ_S are applied per-arc in the RCSPP via multistar_duals.
        # (Letchford, Eglese, Lysgaard 2002 — Generalized Multistar Inequalities)
        self.multistar_cuts: Dict[FrozenSet[int], Dict[int, float]] = {}

    def add_cut(self, cut_type: str, data: Any) -> None:
        """Archive a globally valid cut in the pool.

        Only archives cuts that are valid at EVERY node in the B&B tree.
        Node-local cuts must not be added here.

        Args:
            cut_type: Type of cut ("rcc", "sri", "sec_2.1", "edge_clique", "lci").
            data: Cut-specific data (node-sets, coefficients, etc.).

        Returns:
            None
        """

        if cut_type == "rcc":
            node_set, rhs = data
            # Only archive if better (tighter) than any existing cut on this set.
            existing = self.rcc_cuts.get(node_set, 0.0)
            if rhs > existing:
                self.rcc_cuts[node_set] = rhs
        elif cut_type == "sri":
            node_set, coeff_vec = data
            self.sri_cuts.add(node_set)
            self.active_sri_vectors[node_set] = coeff_vec
        elif cut_type == "sec_2.1":
            self.sec_cuts.add(data)
        elif cut_type == "edge_clique":
            self.edge_clique_cuts.add(data)
        elif cut_type == "lci":
            # Accept 3-tuple, 4-tuple (+ node_alphas), or 5-tuple (+ arc) variants.
            # 5-tuple: (node_set, rhs, coefficients, node_alphas, arc)
            # 4-tuple: (node_set, rhs, coefficients, node_alphas)
            # 3-tuple: (node_set, rhs, coefficients)
            if len(data) == 5:
                node_set, rhs, coefficients, node_alphas, arc = data
            elif len(data) == 4:
                node_set, rhs, coefficients, node_alphas = data
                arc = None
            else:
                node_set, rhs, coefficients = data
                node_alphas = {}
                arc = None
            self.lci_cuts[node_set] = (rhs, coefficients, node_alphas)
            self.lci_arcs[node_set] = arc
        elif cut_type == "multistar":
            # data = (node_set, coefficients) where coefficients = {route_idx: -a_k}
            node_set, coefficients = data
            # Only keep/update if this is a new or more restrictive cut for the same S.
            self.multistar_cuts[node_set] = coefficients
