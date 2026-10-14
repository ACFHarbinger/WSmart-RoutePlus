"""
Package documentation for Hyper-Heuristics module.

This package implements various Hyper-Heuristics for the Vehicle Routing Problem (VRP).

Attributes:
    - Adaptive Memory Programming (AMPHH): policy_amphh
    - Ant Colony Optimization (ACO): policy_aco_hh
    - Genetic Programming (GP): policy_gp_hh
    - Genetic Programming Multi-Period (GP MP): policy_gp_mp_hh
    - Guided Indicators (GIHH): policy_gihh
    - Hidden Markov Model - Great Deluge (HMM-GD): policy_hmm_gd_hh
    - Hyper-Heuristic Using Local Knowledge (HULK): policy_hulk
    - Matheuristic Hyper-Heuristic (MHH): policy_mhh
    - Population-Based (PHH): policy_phh
    - Selection (SHH): policy_shh
    - Sequence-Based Selection (SS-HH): policy_ss_hh

Examples:
    >>> from logic.src.policies.routing.hyper_heuristics import policy_amp_hh
    >>> solver = policy_amp_hh.AMPHHPolicy()
    >>> solution = solver.solve(problem, multi_day_ctx)
"""

from .adaptive_memory_programming_hyper_heuristic import policy_amphh as policy_amp_hh
from .ant_colony_optimization_hyper_heuristic import policy_aco_hh as policy_aco_hh
from .genetic_programming_hyper_heuristic import policy_gp_hh as policy_gp_hh
from .genetic_programming_multi_period_hyper_heuristic import policy_gp_mp_hh as policy_gp_mp_hh
from .guided_indicators_hyper_heuristic import policy_gihh as policy_gihh
from .hidden_markov_model_great_deluge_hyper_heuristic import policy_hmm_gd_hh as policy_hmm_gd_hh
from .hyper_heuristic_us_lk import policy_hulk as policy_hulk
from .matheuristic_hyper_heuristic import policy_mhh as policy_mhh
from .population_based_hyper_heuristic import policy_phh as policy_phh
from .selection_hyper_heuristic import policy_shh as policy_shh
from .sequence_based_selection_hyper_heuristic import policy_ss_hh as policy_ss_hh

__all__ = [
    "policy_aco_hh",
    "policy_gp_hh",
    "policy_gihh",
    "policy_hmm_gd_hh",
    "policy_hulk",
    "policy_ss_hh",
    "policy_shh",
    "policy_mhh",
    "policy_phh",
    "policy_amp_hh",
    "policy_gp_mp_hh",
]
