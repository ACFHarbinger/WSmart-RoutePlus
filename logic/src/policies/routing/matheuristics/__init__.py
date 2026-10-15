"""Matheuristic routing policies. Policy modules load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "policy_aks": "logic.src.policies.routing.matheuristics.adaptive_kernel_search.policy_aks",
    "policy_cf_rs": "logic.src.policies.routing.matheuristics.cluster_first_route_second.policy_cf_rs",
    "policy_cgh": "logic.src.policies.routing.matheuristics.column_generation_heuristic.policy_cgh",
    "policy_ils_rvnd_sp": "logic.src.policies.routing.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp",
    "policy_ks": "logic.src.policies.routing.matheuristics.kernel_search.policy_ks",
    "policy_lrh": "logic.src.policies.routing.matheuristics.lagrangian_relaxation_heuristic.policy_lrh",
    "policy_lns_mip": "logic.src.policies.routing.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip",
    "policy_lkh3": "logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3",
    "policy_lb": "logic.src.policies.routing.matheuristics.local_branching.policy_lb",
    "policy_lb_vns": "logic.src.policies.routing.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns",
    "policy_popmusic": "logic.src.policies.routing.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.policy_popmusic",
    "policy_rfo": "logic.src.policies.routing.matheuristics.relax_and_fix_and_optimize.policy_rfo",
    "policy_rens": "logic.src.policies.routing.matheuristics.relaxation_enforced_neighborhood_search.policy_rens",
    "policy_tpks": "logic.src.policies.routing.matheuristics.two_phase_kernel_search.policy_tpks",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
