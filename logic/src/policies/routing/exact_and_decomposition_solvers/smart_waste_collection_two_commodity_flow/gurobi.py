r"""Gurobi solver for the Smart Waste Collection Routing model (SWC-TCF).

The default formulation is the published model: Ramos, Morais & Barbosa-Póvoa
(2018), "The smart waste collection routing problem: Alternative operational
management approaches", Expert Systems with Applications 103, Section 3.2.2,
eqs. (6), (8), (11)-(15) and (17)-(21), as implemented in
``notebooks/collectioncompare.ipynb`` (``policy_gurobi``): an undirected
two-commodity flow with a copy depot n+1.

Everything that is not in those sources is optional and off by default:

- ``formulation: directed``: the earlier directed two-commodity flow (separate
  waste / empty-space flows on directed arcs, no copy depot).
- ``depot_inflow: le``: the notebook's ``sum y_i0 <= Q k`` instead of eq. (15).
- ``solver_tuning``: MIPFocus/heuristics/cut/presolve settings and a 1% MIP gap.
- ``relax_forced_on_infeasible``: re-solve without forced visits when infeasible.
- ``link_depot_arcs``: ``x[0, j] <= g[j]``.
- ``max_arc_distance_km``: drop arcs longer than this.
- ``warm_start``: Clarke-Wright MIP starts and a no-incumbent fallback (#41).
- ``asymmetric``: directed arcs at their own cost (``_build_asymmetric``) instead of the
  undirected edges of the published model; requires ``formulation: paper``.

Attributes:
    _run_gurobi_optimizer: Solves one day's SWC-TCF model with native Gurobi.

Example:
    >>> res = _run_gurobi_optimizer(bins, dist, env, values, ids, mandatory)
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import NDArray

from logic.src.configs.policies import SWCTCFConfig
from logic.src.constants.routing import HEURISTICS_RATIO, MIP_GAP, NODEFILE_START_GB
from logic.src.policies.routing.base.base_routing_policy import _structured_load, _unwrap_engine_section
from logic.src.policies.routing.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.formulations import (
    _build_asymmetric,
    _build_directed,
    _build_paper,
)

from ._route_extraction import extract_depot_delimited_route
from ._tcf_data import TCFData, build_tcf_data

_CONFIG_FIELDS = {f.name for f in fields(SWCTCFConfig)}


def _config_from_values(values: Dict[str, Any]) -> SWCTCFConfig:
    """Fold the solver switches a caller placed in ``values`` into the typed config.

    ``values`` also carries the non-config runtime scalars (Q, R, C, B, V, ...),
    which are filtered out before the strict structured load.
    """
    return _structured_load(SWCTCFConfig, {key: value for key, value in values.items() if key in _CONFIG_FIELDS})


def _clarke_wright_trips(d: TCFData, distance_matrix: List[List[float]], visit: List[int]) -> Optional[List[List[int]]]:
    """Build Clarke-Wright trips over ``visit`` that are feasible in the TCF model.

    Args:
        d: Shared TCF model data (local node indices, depot 0).
        distance_matrix: Full kilometre matrix in local indexing.
        visit: Local indices of the bins to collect.

    Returns:
        Trips as lists of local node indices, or None when the trips would not
        be feasible in the model (a bin above capacity, a missing arc, or more
        trips than the fleet bound).
    """
    from logic.src.utils.routing.clark_wright import clarke_wright_solve

    if not visit or any(d.S_dict[i] > d.Q for i in visit):
        return None
    tour = clarke_wright_solve(np.asarray(distance_matrix, dtype=float), {i: d.S_dict[i] for i in visit}, d.Q, visit)
    trips: List[List[int]] = []
    current: List[int] = []
    for node in tour[1:]:
        if node == 0:
            if current:
                trips.append(current)
            current = []
        else:
            current.append(node)
    arcs = set(d.valid_arcs)
    for trip in trips:
        path = [0, *trip, 0]
        if any((a, b) not in arcs for a, b in zip(path, path[1:], strict=False)):
            return None
    if len(trips) > d.max_trucks:
        return None
    return trips


def _warm_starts(d: TCFData, distance_matrix: List[List[float]], forced_nodes: List[int]) -> List[List[List[int]]]:
    """Feasible MIP starts: Clarke-Wright trips over the forced bins and over every bin with waste.

    Used only when ``warm_start`` is enabled; the published model (Ramos et al.,
    2018) has no MIP start. At 350 bins and a 60 s limit Gurobi can end without
    any incumbent when only part of the bins is forced, and the day then
    collected nothing (#41). A forced-only start alone fixes that but anchors
    the search on a small plan; the collect-everything start supplies the
    multi-trip plan Gurobi finds on its good days. Gurobi keeps the better one
    and improves it.

    Args:
        d: Shared TCF model data.
        distance_matrix: Full kilometre matrix in local indexing.
        forced_nodes: Local indices whose visit is forced.

    Returns:
        Distinct feasible starts, each a list of trips.
    """
    starts: List[List[List[int]]] = []
    candidates = [forced_nodes, [i for i in d.nodes_real if d.S_dict[i] > 0]]
    for visit in candidates:
        if visit and all(i in visit for i in forced_nodes):
            trips = _clarke_wright_trips(d, distance_matrix, visit)
            if trips is not None and trips not in starts:
                starts.append(trips)
    return starts


def _plan_objective(d: TCFData, distance_matrix: List[List[float]], trips: List[List[int]]) -> Tuple[float, float]:
    """Return (profit, travel cost) of a trip plan under the standard objective."""
    cost = float(sum(distance_matrix[a][b] for trip in trips for a, b in zip([0, *trip], [*trip, 0], strict=False)))
    profit = d.R * sum(d.S_dict[i] for trip in trips for i in trip) - d.C * cost - d.Omega * len(trips)
    return profit, cost


def _flag(value: object) -> bool:
    """Read a yaml/OmegaConf boolean that may arrive as a string."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _optional_float(value: object) -> Optional[float]:
    """Read an optional numeric option (None, '', 'null' or 'none' mean unset)."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "null", "none")):
        return None
    return float(value)  # type: ignore[arg-type]


def _options(cfg: Union[SWCTCFConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """Read and validate the optional, off-by-default model switches.

    Args:
        cfg: Typed SWCTCFConfig, or a raw dict (flat values mapping or an
            engine-wrapped config section), standardized before reading.

    Returns:
        Dict[str, Any]: Validated model switches keyed as the builders expect.
    """
    if isinstance(cfg, dict):
        cfg = _structured_load(SWCTCFConfig, _unwrap_engine_section(cfg))
    formulation = str(cfg.formulation).strip().lower()
    if formulation not in ("paper", "directed"):
        raise ValueError("formulation must be 'paper' or 'directed'")
    depot_inflow = str(cfg.depot_inflow).strip().lower()
    if depot_inflow not in ("equal", "le"):
        raise ValueError("depot_inflow must be 'equal' (eq. 15) or 'le' (notebook)")
    cutoff = _optional_float(cfg.max_arc_distance_km)
    if cutoff is not None and not (math.isfinite(cutoff) and cutoff > 0):
        raise ValueError("max_arc_distance_km must be positive and finite, or null")
    no_solution_factor = float(cfg.no_solution_time_factor)
    if not (math.isfinite(no_solution_factor) and no_solution_factor >= 1):
        raise ValueError("no_solution_time_factor must be a finite number at least 1")
    return {
        "formulation": formulation,
        "depot_inflow": depot_inflow,
        "max_arc_distance_km": cutoff,
        "solver_tuning": _flag(cfg.solver_tuning),
        "relax_forced_on_infeasible": _flag(cfg.relax_forced_on_infeasible),
        "link_depot_arcs": _flag(cfg.link_depot_arcs),
        "warm_start": _flag(cfg.warm_start),
        "asymmetric": _flag(cfg.asymmetric),
        "no_solution_time_factor": no_solution_factor,
    }


def _run_gurobi_optimizer(  # noqa: C901
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    env: Optional[gp.Env],
    values: Dict[str, Any],
    binsids: List[int],
    mandatory: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
    seed: int = 42,
    dual_values: Optional[Dict[int, float]] = None,
    default_feastol: bool = False,
    allow_escalation: bool = True,
    retry: bool = False,
    cfg: Optional[Union[SWCTCFConfig, Dict[str, Any]]] = None,
) -> Tuple[List[int], float, float]:
    """Solve one day of the Smart Waste Collection Routing model.

    Args:
        bins: Bin fill levels in percent (depot excluded).
        distance_matrix: Distance matrix between nodes (km).
        env: Gurobi environment.
        values: Problem parameters (Q, R, C, V runtime scalars).
        binsids: Global identifiers for bins.
        mandatory: IDs of bins that must be collected.
        number_vehicles: Maximum number of vehicles; non-positive means unbounded.
        time_limit: Solver time limit in seconds.
        seed: Random seed for reproducibility.
        dual_values: Dual values for pricing subproblems.
        cfg: Typed policy config (P-10:
            the single source of truth for the solver switches). Direct callers that
            predate the typed config may omit it and pass the switches in ``values``;
            they are folded into the config before reading.

    Returns:
        Tuple[List[int], float, float]: A tuple containing:
            - route: Sequence of node IDs starting and ending at depot (0).
            - profit: The total profit of the solution.
            - cost: The total travel cost of the solution.
    """
    # Standardize cfg to SWCTCFConfig (P-10: the typed config is the only parameter layer).
    if cfg is None:
        cfg = _config_from_values(values)
    elif isinstance(cfg, dict):
        cfg = _structured_load(SWCTCFConfig, _unwrap_engine_section(cfg))

    # Validate before allocating a native model. sim.cpu_cores controls worker
    # processes, not Gurobi's per-solve thread count.
    threads = float(cfg.gurobi_threads)
    memory_gb = float(cfg.gurobi_soft_mem_limit_gb)
    if not math.isfinite(threads) or threads < 1 or not threads.is_integer():
        raise ValueError("gurobi_threads must be a positive integer")
    if not math.isfinite(memory_gb) or memory_gb <= 0:
        raise ValueError("gurobi_soft_mem_limit_gb must be positive and finite")
    options = _options(cfg)

    # Shared preparation (identical across all three backends).
    d = build_tcf_data(bins, distance_matrix, values, binsids, mandatory, number_vehicles)
    forced_nodes = [i for i in d.nodes_real if d.criticos_dict[i] or d.S_dict[i] >= d.psi * 100]

    mdl = gp.Model("VRPP", env=env) if env else gp.Model("VRPP")
    try:
        # Do not loosen limits on a caller-owned environment. SoftMemLimit allows
        # solution extraction on MEM_LIMIT, unlike a hard allocation failure.
        inherited_threads = mdl.Params.Threads
        mdl.Params.Threads = min(int(threads), inherited_threads) if inherited_threads > 0 else int(threads)
        mdl.Params.SoftMemLimit = min(memory_gb, mdl.Params.SoftMemLimit)
        print(f"[INFO][VRPP-Gurobi] Threads={mdl.Params.Threads}, SoftMemLimit={mdl.Params.SoftMemLimit} GB")
        mdl.Params.Seed = seed

        if options["asymmetric"] and options["formulation"] != "paper":
            raise ValueError("asymmetric requires formulation: paper")
        if options["asymmetric"] and options["warm_start"]:
            raise ValueError("warm_start is not defined for the asymmetric model")
        if options["formulation"] != "paper":
            build = _build_directed
        else:
            build = _build_asymmetric if options["asymmetric"] else _build_paper
        built = build(mdl, d, distance_matrix, forced_nodes, number_vehicles, dual_values, options)

        # Optional, off by default: not part of the published model (#41).
        starts = _warm_starts(d, distance_matrix, forced_nodes) if options["warm_start"] else []
        if starts:
            built.set_starts(starts)

        tight_tol = False
        if not env:
            mdl.Params.LogToConsole = 0
            mdl.Params.OutputFlag = 0
            if options["formulation"] == "paper" and not default_feastol:
                mdl.Params.FeasibilityTol = 1e-09  # as in collectioncompare's policy_gurobi
                tight_tol = True
            if options["solver_tuning"]:
                mdl.Params.MIPFocus = 1
                mdl.Params.Heuristics = HEURISTICS_RATIO
                mdl.Params.Cuts = 3
                mdl.Params.CliqueCuts = 2
                mdl.Params.CoverCuts = 2
                mdl.Params.FlowCoverCuts = 2
                mdl.Params.GUBCoverCuts = 2
                mdl.Params.Presolve = 1
                mdl.Params.NodefileStart = NODEFILE_START_GB
                mdl.setParam("MIPGap", MIP_GAP)

        if time_limit > 0:
            mdl.Params.TimeLimit = float(time_limit)

        from logic.src.pipeline.simulations.reporting import format_backend_status, note_solver_status

        mdl.optimize()
        note_solver_status(format_backend_status("gurobi", mdl.Status), append=retry)
        if mdl.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) and tight_tol:
            # The 1e-9 feasibility tolerance can report a feasible model as infeasible. Re-optimizing this model
            # keeps that verdict, so re-solve on a fresh model at Gurobi's default tolerance.
            print("[WARN][VRPP-Gurobi] Infeasible at FeasibilityTol=1e-09; re-solving at the default tolerance.")
            note_solver_status("fallback:default_feastol", append=True)
            return _run_gurobi_optimizer(
                bins,
                distance_matrix,
                env,
                values,
                binsids,
                mandatory,
                number_vehicles,
                time_limit,
                seed,
                dual_values,
                default_feastol=True,
                allow_escalation=False,
                retry=True,
                cfg=cfg,
            )
        if mdl.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) and built.forced and options["relax_forced_on_infeasible"]:
            # Forcing every mandatory / over-psi bin can exceed the fleet's capacity.
            print(
                f"[WARN][VRPP-Gurobi] {len(built.forced)} forced visits are infeasible together; "
                "re-solving without forcing."
            )
            mdl.remove(built.forced)
            mdl.optimize()
            note_solver_status(format_backend_status("gurobi", mdl.Status), append=True)
        if mdl.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            raise RuntimeError(f"SWC-TCF model is infeasible (Gurobi status {mdl.Status}).")
        if mdl.SolCount == 0 and mdl.Status == GRB.TIME_LIMIT and time_limit > 0 and allow_escalation:
            # Hit the limit before any incumbent: the day would otherwise collect nothing. The retry uses the
            # default tolerance and no further retry, so one day costs at most (1 + factor) x the limit.
            longer = int(time_limit * options["no_solution_time_factor"])
            print(
                f"[WARN][VRPP-Gurobi] No incumbent in {time_limit}s; re-solving at the default tolerance with {longer}s."
            )
            note_solver_status("fallback:time_limit", append=True)
            return _run_gurobi_optimizer(
                bins,
                distance_matrix,
                env,
                values,
                binsids,
                mandatory,
                number_vehicles,
                longer,
                seed,
                dual_values,
                default_feastol=True,
                allow_escalation=False,
                retry=True,
                cfg=cfg,
            )
        if mdl.SolCount == 0:
            best = max(starts, key=lambda t: _plan_objective(d, distance_matrix, t)[0]) if starts else None
            # Without forced bins the empty plan (profit 0) is feasible and wins unless a start beats it.
            if (
                best is not None
                and not dual_values
                and (forced_nodes or _plan_objective(d, distance_matrix, best)[0] > 0)
            ):
                profit, cost = _plan_objective(d, distance_matrix, best)
                arcs = [(a, b) for trip in best for a, b in zip([0, *trip], [*trip, 0], strict=False)]
                route = extract_depot_delimited_route(arcs, d.id_map)
                print(
                    f"[WARN][VRPP-Gurobi] No solution found (status {mdl.Status}); "
                    f"executing the best Clarke-Wright start ({len(best)} trips)."
                )
                note_solver_status("fallback:clarke_wright", append=True)
                return route, profit, cost
            print(f"[WARN][VRPP-Gurobi] No solution found (status {mdl.Status}); the day collects nothing.")
            return [0, 0], 0.0, 0.0

        route = built.extract_route()
        if route == [0, 0]:
            # All-zero incumbent (or nothing collected): the shared empty-day shape.
            return [0, 0], 0.0, 0.0
        profit = float(mdl.ObjVal)
        local = {gid: i for i, gid in d.id_map.items()}
        tour = [local[n] for n in route]
        # Travel cost of the executed tour on the (possibly asymmetric) matrix.
        cost = float(sum(distance_matrix[a][b] for a, b in zip(tour, tour[1:], strict=False)))
        print(
            f"[INFO][VRPP-Gurobi] Profit: {profit}, Cost: {cost}, MIPGap: {mdl.Params.MIPGap}, "
            f"Collected: {sum(1 for n in route if n != 0)}"
        )
        return route, profit, cost
    finally:
        # Native solver memory must be released before the next simulated day,
        # including build/optimization errors and early no-incumbent returns.
        mdl.dispose()
