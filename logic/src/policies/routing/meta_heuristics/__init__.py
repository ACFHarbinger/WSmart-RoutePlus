"""Meta-heuristic routing policies. Policy modules load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "policy_alns": "logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.policy_alns",
    "policy_alns_ipo": "logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo",
    "policy_aco_ks": "logic.src.policies.routing.meta_heuristics.ant_colony_optimization_k_sparse.policy_aco_ks",
    "policy_abc": "logic.src.policies.routing.meta_heuristics.artificial_bee_colony.policy_abc",
    "policy_de": "logic.src.policies.routing.meta_heuristics.differential_evolution.policy_de",
    "policy_es_mcl": "logic.src.policies.routing.meta_heuristics.evolution_strategy_mu_comma_lambda.policy_es_mcl",
    "policy_es_mkl": "logic.src.policies.routing.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl",
    "policy_es_mpl": "logic.src.policies.routing.meta_heuristics.evolution_strategy_mu_plus_lambda.policy_es_mpl",
    "policy_filo": "logic.src.policies.routing.meta_heuristics.fast_iterative_localized_optimization.policy_filo",
    "policy_fa": "logic.src.policies.routing.meta_heuristics.firefly_algorithm.policy_fa",
    "policy_ga": "logic.src.policies.routing.meta_heuristics.genetic_algorithm.policy_ga",
    "policy_genius": "logic.src.policies.routing.meta_heuristics.genius.policy_genius",
    "policy_gls": "logic.src.policies.routing.meta_heuristics.guided_local_search.policy_gls",
    "policy_hs": "logic.src.policies.routing.meta_heuristics.harmony_search.policy_hs",
    "policy_hgs": "logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.policy_hgs",
    "policy_hgs_adc": "logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc",
    "policy_hgs_alns": "logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.policy_hgs_alns",
    "policy_hgs_rr": "logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr",
    "policy_hmlns": "logic.src.policies.routing.meta_heuristics.hybrid_memetic_large_neighborhood_search.policy_hmlns",
    "policy_hvpl": "logic.src.policies.routing.meta_heuristics.hybrid_volleyball_premier_league.policy_hvpl",
    "policy_ils": "logic.src.policies.routing.meta_heuristics.iterated_local_search.policy_ils",
    "policy_kgls": "logic.src.policies.routing.meta_heuristics.knowledge_guided_local_search.policy_kgls",
    "policy_lca": "logic.src.policies.routing.meta_heuristics.league_championship_algorithm.policy_lca",
    "policy_ma": "logic.src.policies.routing.meta_heuristics.memetic_algorithm.policy_ma",
    "policy_ma_dp": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_dual_population.policy_ma_dp",
    "policy_ma_im": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_island_model.policy_ma_im",
    "policy_ma_ts": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_tolerance_based_selection.policy_ma_ts",
    "policy_mp_aco": "logic.src.policies.routing.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco",
    "policy_mp_ils": "logic.src.policies.routing.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils",
    "policy_mp_pso": "logic.src.policies.routing.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso",
    "policy_mp_sa": "logic.src.policies.routing.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa",
    "policy_pso": "logic.src.policies.routing.meta_heuristics.particle_swarm_optimization.policy_pso",
    "policy_psoda": "logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.policy_psoda",
    "policy_psoma": "logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_memetic_algorithm.policy_psoma",
    "policy_pg_clns": "logic.src.policies.routing.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.policy_pg_clns",
    "policy_qde": "logic.src.policies.routing.meta_heuristics.quantum_differential_evolution.policy_qde",
    "policy_rts": "logic.src.policies.routing.meta_heuristics.reactive_tabu_search.policy_rts",
    "policy_sa": "logic.src.policies.routing.meta_heuristics.simulated_annealing.policy_sa",
    "policy_sans": "logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.policy_sans",
    "policy_sca": "logic.src.policies.routing.meta_heuristics.sine_cosine_algorithm.policy_sca",
    "policy_sisr": "logic.src.policies.routing.meta_heuristics.slack_induction_by_string_removal.policy_sisr",
    "policy_slc": "logic.src.policies.routing.meta_heuristics.soccer_league_competition.policy_slc",
    "policy_ts": "logic.src.policies.routing.meta_heuristics.tabu_search.policy_ts",
    "policy_vns": "logic.src.policies.routing.meta_heuristics.variable_neighborhood_search.policy_vns",
    "policy_vpl": "logic.src.policies.routing.meta_heuristics.volleyball_premier_league.policy_vpl",
}

__all__ = [
    "policy_alns",
    "policy_hgs",
    "policy_psoma",
    "policy_pg_clns",
    "policy_sans",
]


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
