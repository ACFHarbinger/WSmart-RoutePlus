"""Shared problem preparation for the SWC-TCF backends.

Every backend (native Gurobi, OR-Tools, Pyomo) used to derive the same
quantities from the raw inputs: node indexing, percent fills, the mandatory
(criticos) map, the arc cutoff, and the vehicle fallback. This module builds
them once so the three model builders cannot drift apart.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .params import MAX_ARC_DISTANCE_KM


@dataclass
class TCFData:
    """Common inputs derived once for all SWC-TCF model builders.

    Attributes:
        Q: Vehicle capacity in percent-of-one-bin units.
        R: Revenue per percent point (€).
        C: Transport cost per kilometre (€/km).
        Omega: Per-vehicle charge (€).
        psi: Force-visit threshold, fraction of bin capacity.
        n_bins: Number of bins in the sub-problem (depot excluded).
        nodes: Node indices, depot first (0).
        nodes_real: Node indices without the depot.
        S_dict: Fill level in percent for every node (depot 0).
        pure_binsids: Global bin ids without the depot marker.
        criticos_dict: True where a bin is mandatory or already critical.
        valid_arcs: Directed arcs within the distance cutoff.
        id_map: Local node index -> global bin id (depot maps to 0).
        max_trucks: Vehicle bound (explicit limit, or n_bins when unbounded).
    """

    Q: float
    R: float
    C: float
    Omega: float
    psi: float
    n_bins: int
    nodes: List[int]
    nodes_real: List[int]
    S_dict: Dict[int, float]
    pure_binsids: List[int]
    criticos_dict: Dict[int, bool]
    valid_arcs: List[Tuple[int, int]]
    id_map: Dict[int, int]
    max_trucks: int


def build_tcf_data(
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    values: Dict[str, float],
    binsids: List[int],
    mandatory: List[int],
    number_vehicles: int,
) -> TCFData:
    """Derive the shared SWC-TCF model inputs from the raw adapter data.

    Args:
        bins: Fill levels in percent of bin capacity (depot excluded).
        distance_matrix: Full (n_bins + 1) x (n_bins + 1) kilometre matrix.
        values: Adapter parameters (Q, R, C, Omega, psi; percent units).
        binsids: Global bin ids, optionally depot-first (length n_bins + 1).
        mandatory: Global ids of bins that must be collected.
        number_vehicles: Fleet bound; non-positive means unbounded (n_bins).

    Returns:
        A TCFData consumed by every backend model builder.
    """
    Omega, psi = values["Omega"], values["psi"]
    Q, R, C = values["Q"], values["R"], values["C"]

    n_bins = len(bins)
    nodes = list(range(n_bins + 1))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]

    enchimentos = np.insert(bins, 0, 0.0)
    S_dict = {i: float(enchimentos[i]) for i in nodes}

    pure_binsids = binsids[1:] if len(binsids) == n_bins + 1 else binsids
    criticos_dict = {0: False}
    for i, bin_id in enumerate(pure_binsids, 1):
        criticos_dict[i] = bin_id in mandatory

    valid_arcs = [
        (i, j) for i in nodes for j in nodes if i != j and distance_matrix[i][j] <= MAX_ARC_DISTANCE_KM
    ]

    id_map = {0: 0}
    for i, bin_id in enumerate(pure_binsids, 1):
        id_map[i] = bin_id

    max_trucks = number_vehicles if number_vehicles > 0 else n_bins

    return TCFData(
        Q=Q,
        R=R,
        C=C,
        Omega=Omega,
        psi=psi,
        n_bins=n_bins,
        nodes=nodes,
        nodes_real=nodes_real,
        S_dict=S_dict,
        pure_binsids=pure_binsids,
        criticos_dict=criticos_dict,
        valid_arcs=valid_arcs,
        id_map=id_map,
        max_trucks=max_trucks,
    )
