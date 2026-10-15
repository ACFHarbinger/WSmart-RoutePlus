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

from typing import Any, Dict, Optional, Union

import numpy as np

from logic.src.configs.policies import SWCTCFConfig
from logic.src.policies.routing.base.base_routing_policy import _structured_load, _unwrap_engine_section
from logic.src.policies.routing.base.sub_problem import SubProblem

from .gurobi import _run_gurobi_optimizer


def run_swc_tcf_optimizer(
    problem: SubProblem,
    cfg: Optional[Union[SWCTCFConfig, Dict[str, Any]]] = None,
):
    """Solve SWC-TCF using either Google OR-Tools or Pyomo wrappers.

    P-30: takes the consolidated sub-problem value object plus the typed
    config; the adapter-side input conversions (wastes dict → amounts array,
    matrix → list-of-lists, fleet limit, seed) live here now.

    Args:
        problem (SubProblem): Routing sub-problem. ``wastes`` is the
            {local_idx: fill} mapping, ``dist_matrix`` the symmetric distance
            matrix, ``n_vehicles`` the fleet limit (0/None = unbounded) and
            ``dual_values`` optional pricing duals.
        cfg (Optional[Union[SWCTCFConfig, Dict[str, Any]]]): Typed policy
            config (framework/engine/time_limit/seed); a raw dict is coerced
            leniently into SWCTCFConfig (WP-04 pattern).

    Returns:
        Tuple[List[int], float, float]: (flat depot-delimited route, profit, cost)
    """
    config = cfg if isinstance(cfg, SWCTCFConfig) else _structured_load(SWCTCFConfig, _unwrap_engine_section(cfg))

    # Adapter-side conversions, moved here from the policy (P-30).
    n_nodes = len(problem.wastes)
    amounts = np.zeros(n_nodes)
    for i, fill in problem.wastes.items():
        amounts[i - 1] = fill  # Solver uses 1-based binsids internally for depot mapping
    binsids = list(range(1, n_nodes + 1))
    distance_matrix = problem.dist_matrix.tolist()
    values = {**dict(problem.values or {}), "Q": problem.capacity, "R": problem.revenue, "C": problem.cost_unit}
    mandatory_nodes = problem.mandatory_nodes
    number_vehicles = int(problem.n_vehicles) if problem.n_vehicles and int(problem.n_vehicles) > 0 else 0
    time_limit = float(config.time_limit)  # int() turned sub-second budgets into "no limit"
    framework = config.framework
    optimizer = config.engine
    seed = int(config.seed) if config.seed is not None else 42
    dual_values = problem.dual_values

    # Map solver names to the exact casing required by the respective frameworks
    if framework == "ortools":
        # OR-Tools requires uppercase identifiers (e.g., 'GUROBI', 'SCIP', 'HIGHS')
        ortools_backend = optimizer.upper()
        if ortools_backend not in ["GUROBI", "SCIP", "HIGHS", "CPLEX"]:
            raise ValueError(f"Unsupported OR-Tools backend: '{ortools_backend}'")
        # Imported here so a build that ships only some frameworks can drop the others' modules.
        from .ortools_wrapper import _run_ortools_tcf_optimizer


        result = _run_ortools_tcf_optimizer(
            bins=amounts,
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
                bins=amounts,
                distance_matrix=distance_matrix,
                env=None,
                values={k: float(v) if isinstance(v, (int, float)) else v for k, v in values.items()},
                binsids=binsids,
                mandatory=mandatory_nodes,
                number_vehicles=number_vehicles,
                time_limit=time_limit,
                seed=seed,
                dual_values=dual_values,
                cfg=cfg,
            )
        return result

    elif framework == "pyomo":
        # Pyomo requires lowercase identifiers (e.g., 'gurobi', 'scip', 'appsi_highs')
        pyomo_backend = optimizer.lower()
        if pyomo_backend not in ["gurobi", "scip", "appsi_highs", "highs"]:
            raise ValueError(f"Unsupported Pyomo backend: '{pyomo_backend}'")
        from .pyomo_wrapper import _run_pyomo_tcf_optimizer


        return _run_pyomo_tcf_optimizer(
            bins=amounts,
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
            bins=amounts,
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
            cfg=cfg,
        )

    else:
        raise ValueError(
            f"Unknown optimization framework: '{framework}'. Use 'ortools:<solver>', 'pyomo:<solver>', or 'gurobi'/'native'."
        )
