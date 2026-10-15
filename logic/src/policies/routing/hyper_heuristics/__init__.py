"""Hyper-heuristic routing policies. Policy modules load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "policy_amp_hh": "logic.src.policies.routing.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh",
    "policy_aco_hh": "logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh",
    "policy_gp_hh": "logic.src.policies.routing.hyper_heuristics.genetic_programming_hyper_heuristic.policy_gp_hh",
    "policy_gp_mp_hh": "logic.src.policies.routing.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh",
    "policy_gihh": "logic.src.policies.routing.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh",
    "policy_hmm_gd_hh": "logic.src.policies.routing.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh",
    "policy_hulk": "logic.src.policies.routing.hyper_heuristics.hyper_heuristic_us_lk.policy_hulk",
    "policy_mhh": "logic.src.policies.routing.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh",
    "policy_phh": "logic.src.policies.routing.hyper_heuristics.population_based_hyper_heuristic.policy_phh",
    "policy_shh": "logic.src.policies.routing.hyper_heuristics.selection_hyper_heuristic.policy_shh",
    "policy_ss_hh": "logic.src.policies.routing.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
