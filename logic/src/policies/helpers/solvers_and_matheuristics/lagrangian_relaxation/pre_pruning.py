"""Lagrangian Relaxation utilities for Branch-and-Price-and-Cut solvers.

Provides the Lagrangian pre-pruning subgradient pass that can be applied at
any B&B node before column generation to prune nodes whose Lagrangian upper
bound is dominated by the current incumbent.

Attributes:
-----------
compute_lr_bound_at_node  — Subgradient-based LR bound at a single B&B node.

Example:
    None

References
----------
Poggi de Aragão, Viana & Uchoa (2010) ATMOS — LR pre-pruning for TOP/VRPP.
Polyak (1967) — Subgradient step-size rule.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization import (
    run_subgradient,
)
from logic.src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem import (
    solve_uncapacitated_op,
)

logger = logging.getLogger(__name__)


def compute_lr_bound_at_node(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory: Set[int],
    forced_out: Set[int],
    params: Any,
    time_budget: float,
    env: Optional[Any],
    recorder: Optional[Any],
) -> Tuple[float, float, Set[int]]:
    """
    Compute a fast Lagrangian upper bound at a BPC B&B node.

    Runs a lightweight subgradient pass over the *effective* customer set
    (excluding customers already forced out by branching), then returns the
    tightest Lagrangian bound found and λ* for optional CG warm-starting.

    The forced_out set comes from the active branching constraints at this node.
    Customers in forced_out are pre-excluded from the subproblem, making the
    bound tighter and the UOP solve faster as the tree deepens.

    Args:
        dist_matrix:  Full distance matrix (n × n), index 0 = depot.
        wastes:       {customer_id → fill_level}.
        capacity:     Vehicle capacity Q.
        R:            Revenue coefficient.
        C:            Distance cost coefficient.
        mandatory:      Customers forced in by branching (forced_in for the UOP).
        forced_out:   Customers forced out by branching (excluded from UOP).
        params:       BPCParams carrying the lr_* fields.
        time_budget:  Wall-clock seconds available for the subgradient phase.
        env:          Optional shared Gurobi environment.
        recorder:     Optional telemetry recorder.

    Returns:
        (lr_upper_bound, lam_star, op_visited_set) where:
            lr_upper_bound – Tightest Lagrangian bound found: min_k L(λ_k).
            lam_star       – λ that achieved lr_upper_bound.
            op_visited_set – Customer set from the UOP solve at λ*; used for
                             optional column seeding (lr_warm_start_cg).
    """
    # Build a trimmed wastes dict that excludes forced-out customers.
    # run_subgradient internally calls solve_uncapacitated_op, which already
    # accepts forced_in and forced_out. We pass forced_out through the
    # mandatory_indices mechanism by manipulating the wastes dict instead, so
    # that the a-priori elimination step inside solve_uncapacitated_op skips them.
    # The cleaner path is to pass forced_out explicitly. run_subgradient does
    # not currently accept forced_out, so we filter wastes here.
    trimmed_wastes = {k: v for k, v in wastes.items() if k not in forced_out}

    # Create a temporary BBParams-compatible object to call run_subgradient.
    # run_subgradient accepts a params object with these specific fields; we
    # use a SimpleNamespace to avoid a hard dependency on BBParams in bpc_engine.
    lr_params = SimpleNamespace(
        lr_lambda_init=params.lr_lambda_init,
        lr_max_subgradient_iters=params.lr_max_subgradient_iters,
        lr_subgradient_theta=params.lr_subgradient_theta,
        lr_op_time_limit=params.lr_op_time_limit,
        mip_gap=params.optimality_gap,
        seed=params.seed if (hasattr(params, "seed") and params.seed is not None) else 42,
    )

    lam_star, ub_best, _lb, _history = run_subgradient(
        dist_matrix=dist_matrix,
        wastes=trimmed_wastes,
        capacity=capacity,
        R=R,
        C=C,
        mandatory_indices=mandatory,
        params=lr_params,
        time_budget=time_budget,
        env=env,
        recorder=recorder,
    )

    # Resolve UOP at λ* to get the visited set for CG warm-starting.
    # This is a single additional solve (cheap, reuses λ*).
    op_visited: Set[int] = set()
    if params.lr_warm_start_cg:
        op_visited, _, _ = solve_uncapacitated_op(
            dist_matrix=dist_matrix,
            wastes=trimmed_wastes,
            lam=lam_star,
            R=R,
            C=C,
            forced_in=mandatory,
            forced_out=forced_out,
            time_limit=params.lr_op_time_limit,
            seed=lr_params.seed,
            env=env,
            recorder=recorder,
        )

    return ub_best, lam_star, op_visited
