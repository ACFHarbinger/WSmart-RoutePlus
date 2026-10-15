"""Policies package. Public names load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "IRoutingAlgorithm": "logic.src.interfaces.routing_algorithm:IRoutingAlgorithm",
    "RoutingAlgorithmFactory": "logic.src.policies.routing.base.factory:RoutingAlgorithmFactory",
    "RoutingAlgorithmRegistry": "logic.src.policies.routing.base.registry:RoutingAlgorithmRegistry",
    "run_alns": "logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.dispatcher:run_alns",
    "find_routes": "logic.src.policies.routing.other_algorithms.capacitated_vehicle_routing_problem.cvrp:find_routes",
    "find_routes_ortools": "logic.src.policies.routing.other_algorithms.capacitated_vehicle_routing_problem.cvrp:find_routes_ortools",
    "GIHHSolver": "logic.src.policies.routing.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh:GIHHSolver",
    "GIHHConfig": "logic.src.policies.routing.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh:GIHHConfig",
    "run_hgs": "logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.dispatcher:run_hgs",
    "HGSRRSolver": "logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr:HGSRRSolver",
    "HGSRRConfig": "logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr:HGSRRConfig",
    "NeuralAgent": "logic.src.policies.routing.learning_algorithms.neural_agent.policy_na:NeuralAgent",
    "find_route": "logic.src.policies.routing.other_algorithms.travelling_salesman_problem.tsp:find_route",
    "MuPlusLambdaESSolver": "logic.src.policies.routing.meta_heuristics.evolution_strategy_mu_plus_lambda.solver:MuPlusLambdaESSolver",
    "MuPlusLambdaESConfig": "logic.src.policies.routing.meta_heuristics.evolution_strategy_mu_plus_lambda.solver:MuPlusLambdaESConfig",
    "DistancePSOSolver": "logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.solver:DistancePSOSolver",
    "DistancePSOConfig": "logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.solver:DistancePSOConfig",
    "DESolver": "logic.src.policies.routing.meta_heuristics.differential_evolution.solver:DESolver",
    "DEConfig": "logic.src.policies.routing.meta_heuristics.differential_evolution.params:DEConfig",
    "MuCommaLambdaESSolver": "logic.src.policies.routing.meta_heuristics.evolution_strategy_mu_comma_lambda.solver:MuCommaLambdaESSolver",
    "MuCommaLambdaESConfig": "logic.src.policies.routing.meta_heuristics.evolution_strategy_mu_comma_lambda.solver:MuCommaLambdaESConfig",
    "HybridMemeticLargeNeighborhoodSearchSolver": "logic.src.policies.routing.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver:HybridMemeticLargeNeighborhoodSearchSolver",
    "HybridMemeticLargeNeighborhoodSearchParams": "logic.src.policies.routing.meta_heuristics.hybrid_memetic_large_neighborhood_search.params:HybridMemeticLargeNeighborhoodSearchParams",
    "MemeticAlgorithmIslandModelSolver": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_island_model.solver:MemeticAlgorithmIslandModelSolver",
    "MemeticAlgorithmIslandModelConfig": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_island_model.params:MemeticAlgorithmIslandModelConfig",
    "MemeticAlgorithmToleranceBasedSelectionSolver": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_tolerance_based_selection.solver:MemeticAlgorithmToleranceBasedSelectionSolver",
    "MemeticAlgorithmToleranceBasedSelectionConfig": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_tolerance_based_selection.params:MemeticAlgorithmToleranceBasedSelectionConfig",
    "MemeticAlgorithmDualPopulationSolver": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_dual_population.solver:MemeticAlgorithmDualPopulationSolver",
    "MemeticAlgorithmDualPopulationConfig": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_dual_population.params:MemeticAlgorithmDualPopulationConfig",
    "NDSBRKGAPolicy": "logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga:NDSBRKGAPolicy",
    "NDSBRKGAConfig": "logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga:NDSBRKGAConfig",
    "JointSAPolicy": "logic.src.policies.packing_and_routing.joint_simulated_annealing.policy_jsa:JointSAPolicy",
    "JointSAConfig": "logic.src.policies.packing_and_routing.joint_simulated_annealing.policy_jsa:JointSAConfig",
    "JointGreedyPolicy": "logic.src.policies.packing_and_routing.joint_greedy_orienteering.policy_jgo:JointGreedyPolicy",
    "JointGreedyConfig": "logic.src.policies.packing_and_routing.joint_greedy_orienteering.policy_jgo:JointGreedyConfig",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
