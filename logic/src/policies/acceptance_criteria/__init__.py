"""Acceptance criteria. Modules load on first access."""

import importlib
from typing import Any

from .base.factory import AcceptanceCriterionFactory
from .base.registry import AcceptanceCriterionRegistry

_LAZY_MODULES = (
    "adaptive_boltzmann_metropolis",
    "all_moves_accepted",
    "aspiration_criterion",
    "binary_tournament_acceptance",
    "boltzmann_metropolis_criterion",
    "demon_algorithm",
    "ensemble_move_acceptance",
    "epsilon_dominance",
    "exponential_monte_carlo_counter",
    "fitness_proportional",
    "generalized_tsallis_simulated_annealing",
    "great_deluge",
    "improving_and_equal",
    "late_acceptance_hill_climbing",
    "legacy_sa",
    "monte_carlo",
    "non_linear_great_deluge",
    "old_bachelor_acceptance",
    "only_improving",
    "pareto_dominance",
    "power_cooling",
    "probabilistic_transition",
    "record_to_record_travel",
    "sans_geometric",
    "skewed_variable_neighborhood_search",
    "step_counting_hill_climbing",
    "threshold_accepting",
)

__all__ = ["AcceptanceCriterionFactory", "AcceptanceCriterionRegistry", *_LAZY_MODULES]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_MODULES:
        raise AttributeError(f"module has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
