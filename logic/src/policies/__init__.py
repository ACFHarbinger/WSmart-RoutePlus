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
    "GIHHParams": "logic.src.policies.routing.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh:GIHHParams",
    "run_hgs": "logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.dispatcher:run_hgs",
    "HGSRRSolver": "logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr:HGSRRSolver",
    "HGSRRParams": "logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr:HGSRRParams",
    "NeuralAgent": "logic.src.policies.routing.learning_algorithms.neural_agent.policy_na:NeuralAgent",
    "find_route": "logic.src.policies.routing.other_algorithms.travelling_salesman_problem.tsp:find_route",
    "MuPlusLambdaESSolver": "logic.src.policies.routing.meta_heuristics.evolution_strategy_mu_plus_lambda.solver:MuPlusLambdaESSolver",
    "MuPlusLambdaESParams": "logic.src.policies.routing.meta_heuristics.evolution_strategy_mu_plus_lambda.solver:MuPlusLambdaESParams",
    "DistancePSOSolver": "logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.solver:DistancePSOSolver",
    "DistancePSOParams": "logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.solver:DistancePSOParams",
    "DESolver": "logic.src.policies.routing.meta_heuristics.differential_evolution.solver:DESolver",
    "DEParams": "logic.src.policies.routing.meta_heuristics.differential_evolution.params:DEParams",
    "MuCommaLambdaESSolver": "logic.src.policies.routing.meta_heuristics.evolution_strategy_mu_comma_lambda.solver:MuCommaLambdaESSolver",
    "MuCommaLambdaESParams": "logic.src.policies.routing.meta_heuristics.evolution_strategy_mu_comma_lambda.solver:MuCommaLambdaESParams",
    "HybridMemeticLargeNeighborhoodSearchSolver": "logic.src.policies.routing.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver:HybridMemeticLargeNeighborhoodSearchSolver",
    "HybridMemeticLargeNeighborhoodSearchParams": "logic.src.policies.routing.meta_heuristics.hybrid_memetic_large_neighborhood_search.params:HybridMemeticLargeNeighborhoodSearchParams",
    "MemeticAlgorithmIslandModelSolver": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_island_model.solver:MemeticAlgorithmIslandModelSolver",
    "MemeticAlgorithmIslandModelParams": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_island_model.params:MemeticAlgorithmIslandModelParams",
    "MemeticAlgorithmToleranceBasedSelectionSolver": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_tolerance_based_selection.solver:MemeticAlgorithmToleranceBasedSelectionSolver",
    "MemeticAlgorithmToleranceBasedSelectionParams": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_tolerance_based_selection.params:MemeticAlgorithmToleranceBasedSelectionParams",
    "MemeticAlgorithmDualPopulationSolver": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_dual_population.solver:MemeticAlgorithmDualPopulationSolver",
    "MemeticAlgorithmDualPopulationParams": "logic.src.policies.routing.meta_heuristics.memetic_algorithm_dual_population.params:MemeticAlgorithmDualPopulationParams",
    "NDSBRKGAPolicy": "logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga:NDSBRKGAPolicy",
    "NDSBRKGAParams": "logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga:NDSBRKGAParams",
    "JointSAPolicy": "logic.src.policies.packing_and_routing.joint_simulated_annealing.policy_jsa:JointSAPolicy",
    "JointSAParams": "logic.src.policies.packing_and_routing.joint_simulated_annealing.policy_jsa:JointSAParams",
    "JointGreedyPolicy": "logic.src.policies.packing_and_routing.joint_greedy_orienteering.policy_jgo:JointGreedyPolicy",
    "JointGreedyParams": "logic.src.policies.packing_and_routing.joint_greedy_orienteering.policy_jgo:JointGreedyParams",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
