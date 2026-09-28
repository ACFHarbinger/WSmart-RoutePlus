"""Shared tour extraction for the SWC-TCF backends.

All three solver wrappers (native Gurobi, OR-Tools, Pyomo) receive the same
solution shape --- a set of active arcs over local node indices --- and walk
it into a flat, depot-delimited route list. This module hosts that single
implementation so the backends cannot drift apart.
"""

from typing import Dict, List, Tuple

Arc = Tuple[int, int]


def extract_depot_delimited_route(active_arcs: List[Arc], id_map: Dict[int, int]) -> List[int]:
    """Walk active arcs from the depot into a flat, depot-delimited route list.

    Each route starts and ends at the depot; consecutive routes are separated
    by a single depot entry, so ``[0, a, b, 0, c, 0]`` encodes two routes
    ``[a, b]`` and ``[c]``. An empty plan returns ``[0, 0]`` (the shared
    empty-day shape).

    Args:
        active_arcs: Directed arcs with value above the integrality threshold,
            in the solver's local node indexing (0 = depot).
        id_map: Mapping from local node index to the global bin id (0 maps to 0).

    Returns:
        Flat depot-delimited route list.
    """
    remaining = list(active_arcs)
    visited: set = set()
    collected: List[int] = []

    while True:
        rota: List[Arc] = []
        atual = 0
        while True:
            prox = [j for (i, j) in remaining if i == atual and (i, j) not in visited]
            if not prox:
                break
            j = prox[0]
            visited.add((atual, j))
            rota.append((atual, j))
            atual = j
            if j == 0:
                break
        if rota:
            collected.extend(id_map[j] for (i, j) in rota if j != 0)
            collected.append(0)
        else:
            break

    if not collected:
        return [0, 0]
    return [0] + collected
