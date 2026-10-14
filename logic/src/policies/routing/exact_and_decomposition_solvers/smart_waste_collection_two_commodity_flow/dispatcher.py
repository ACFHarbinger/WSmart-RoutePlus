r"""SWC-TCF (Smart Waste Collection - Two-Commodity Flow) Optimizer Interface.

Reference:
    Ramos, T. R. P., Morais, C. S., & Barbosa-Povoa, A. P.
    "The smart waste collection routing problem:
    Alternative operational management approaches", 2018.

Attributes:
    run_swc_tcf_optimizer: Solves TCF using OR-Tools or Pyomo wrappers.

Example:
    >>> res = run_swc_tcf_optimizer(bins, dist, values, ids, mandatory)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .gurobi import _run_gurobi_optimizer


def run_swc_tcf_optimizer(
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    values: Dict[str, float],
    binsids: List[int],
    mandatory_nodes: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
    framework: str = "ortools",
    optimizer: str = "gurobi",
    seed: int = 42,
    dual_values: Optional[Dict[int, float]] = None,
):
    """Solve SWC-TCF using either Google OR-Tools or Pyomo wrappers.

    Args:
        bins (NDArray[np.float64]): Array of bin fill levels.
        distance_matrix (List[List[float]]): Distance matrix between nodes.
        values (Dict[str, float]): Problem parameters (Q, R, B, C, V, Omega, etc.).
        binsids (List[int]): Global identifiers for bins.
        mandatory_nodes (List[int]): IDs of bins that must be collected.
        number_vehicles (int): Number of available vehicles.
        time_limit (int): Solver time limit in seconds.
        framework (str): Optimization framework ('ortools', 'pyomo', or 'native').
        optimizer (str): Backend solver ('gurobi', 'scip', 'highs').
        seed (int): Random seed for reproducibility.
        dual_values (Optional[Dict[int, float]]): Dual values for pricing.

    Returns:
        Tuple[List[int], float, float]: (route, profit, cost)
    """
    # Map solver names to the exact casing required by the respective frameworks
    if framework == "ortools":
        # OR-Tools requires uppercase identifiers (e.g., 'GUROBI', 'SCIP', 'HIGHS')
        ortools_backend = optimizer.upper()
        if ortools_backend not in ["GUROBI", "SCIP", "HIGHS", "CPLEX"]:
            raise ValueError(f"Unsupported OR-Tools backend: '{ortools_backend}'")
        # Imported here so a build that ships only some frameworks can drop the others' modules.
        from .ortools_wrapper import _run_ortools_tcf_optimizer


        result = _run_ortools_tcf_optimizer(
            bins=bins,
            distance_matrix=distance_matrix,
            values=values,
            binsids=binsids,
            mandatory_nodes=mandatory_nodes,
            number_vehicles=number_vehicles,
            time_limit=time_limit,
            solver_id=ortools_backend,
            seed=seed,
            dual_values=dual_values,
        )
        # Fallback to native Gurobi if OR-Tools wrapper failed (likely due to missing linkage/license)
        if ortools_backend == "GUROBI" and result[0] == [0, 0] and result[1] == 0.0:
            return _run_gurobi_optimizer(
                bins=bins,
                distance_matrix=distance_matrix,
                env=None,
                values={k: float(v) if isinstance(v, (int, float)) else v for k, v in values.items()},
                binsids=binsids,
                mandatory=mandatory_nodes,
                number_vehicles=number_vehicles,
                time_limit=time_limit,
                seed=seed,
                dual_values=dual_values,
            )
        return result

    elif framework == "pyomo":
        # Pyomo requires lowercase identifiers (e.g., 'gurobi', 'scip', 'appsi_highs')
        pyomo_backend = optimizer.lower()
        if pyomo_backend not in ["gurobi", "scip", "appsi_highs", "highs"]:
            raise ValueError(f"Unsupported Pyomo backend: '{pyomo_backend}'")
        from .pyomo_wrapper import _run_pyomo_tcf_optimizer


        return _run_pyomo_tcf_optimizer(
            bins=bins,
            distance_matrix=distance_matrix,
            values=values,
            binsids=binsids,
            mandatory_nodes=mandatory_nodes,
            number_vehicles=number_vehicles,
            time_limit=time_limit,
            solver_id="appsi_highs" if pyomo_backend == "highs" else pyomo_backend,
            seed=seed,
            dual_values=dual_values,
        )

    elif framework in ["gurobi", "native"]:
        # Direct Gurobi delegation (bypass OR-Tools/Pyomo abstractions)
        # Note: This requires gurobipy to be installed and license available.
        return _run_gurobi_optimizer(
            bins=bins,
            distance_matrix=distance_matrix,
            env=None,  # Use default environment
            values={
                k: (
                    float(v)
                    if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit())
                    else v
                )
                for k, v in values.items()
            },
            binsids=binsids,
            mandatory=mandatory_nodes,
            number_vehicles=number_vehicles,
            time_limit=time_limit,
            seed=seed,
            dual_values=dual_values,
        )

    else:
        raise ValueError(
            f"Unknown optimization framework: '{framework}'. Use 'ortools:<solver>', 'pyomo:<solver>', or 'gurobi'/'native'."
        )
