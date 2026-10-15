"""Factory for route improvers. Construction is ``IMPROVEMENT.create``."""

from typing import Any, Dict, List

from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.utils.registry import IMPROVEMENT

from .registry import RouteImproverRegistry


def _improver_kwargs(method: str, config: Any) -> Dict[str, Any]:
    """Read the config subsection for one improver method."""
    method_lower = method.lower()
    if method_lower in ("fast_tsp", "tsp", "ftsp"):
        return {"time_limit": config.fast_tsp.time_limit, "seed": config.fast_tsp.seed}
    if method_lower in ("lkh", "lkh3", "lkh-3", "lin_kernighan_helsgaun"):
        return {
            "max_iterations": config.lkh.max_iterations,
            "time_limit": config.lkh.time_limit,
            "seed": config.lkh.seed,
        }
    if method_lower in ("local_search", "classical_local_search", "cls"):
        return {
            "ls_operator": config.local_search.ls_operator,
            "iterations": config.local_search.iterations,
            "time_limit": config.local_search.time_limit,
            "seed": config.local_search.seed,
        }
    if method_lower in ("random", "random_local_search", "rls"):
        return {
            "iterations": config.random_local_search.iterations,
            "params": config.random_local_search.params,
            "time_limit": config.random_local_search.time_limit,
            "seed": config.random_local_search.seed,
        }
    if method_lower == "path":
        return {"vehicle_capacity": config.path.vehicle_capacity}
    if method_lower == "or_opt":
        return {
            "chain_len": config.or_opt.chain_len,
            "iterations": config.or_opt.iterations,
            "seed": config.or_opt.seed,
        }
    if method_lower == "cross_exchange":
        return {
            "cross_exchange_max_segment_len": config.cross_exchange.cross_exchange_max_segment_len,
            "iterations": config.cross_exchange.iterations,
            "seed": config.cross_exchange.seed,
        }
    if method_lower == "guided_local_search":
        return {
            "gls_iterations": config.guided_local_search.gls_iterations,
            "gls_inner_iterations": config.guided_local_search.gls_inner_iterations,
            "gls_lambda_factor": config.guided_local_search.gls_lambda_factor,
            "gls_base_operator": config.guided_local_search.gls_base_operator,
            "seed": config.guided_local_search.seed,
        }
    if method_lower == "simulated_annealing":
        t_init = 10.0
        alpha = 0.999
        if config.simulated_annealing.acceptance_criterion.method == "bmc":
            bmc_cfg = config.simulated_annealing.acceptance_criterion.params
            if bmc_cfg:
                t_init = getattr(bmc_cfg, "initial_temp", t_init)
                alpha = getattr(bmc_cfg, "alpha", alpha)
        return {
            "sa_iterations": config.simulated_annealing.iterations,
            "sa_t_init": t_init,
            "sa_cooling": alpha,
            "params": config.simulated_annealing.params,
            "seed": config.simulated_annealing.seed,
        }
    if method_lower == "cheapest_insertion":
        return {
            "cost_per_km": config.insertion.cost_per_km,
            "revenue_kg": config.insertion.revenue_kg,
            "n_bins": config.insertion.n_bins,
            "seed": config.insertion.seed,
        }
    if method_lower == "regret_k_insertion":
        return {
            "regret_k": config.insertion.regret_k,
            "cost_per_km": config.insertion.cost_per_km,
            "revenue_kg": config.insertion.revenue_kg,
            "n_bins": config.insertion.n_bins,
            "seed": config.insertion.seed,
        }
    if method_lower == "profitable_detour":
        return {
            "detour_epsilon": config.insertion.detour_epsilon,
            "cost_per_km": config.insertion.cost_per_km,
            "revenue_kg": config.insertion.revenue_kg,
            "n_bins": config.insertion.n_bins,
            "seed": config.insertion.seed,
        }
    if method_lower == "ruin_recreate":
        return {
            "lns_iterations": config.ruin_recreate.lns_iterations,
            "ruin_fraction": config.ruin_recreate.ruin_fraction,
            "lns_acceptance": config.ruin_recreate.acceptance_criterion.method,
            "repair_k": config.ruin_recreate.repair_k,
            "cost_per_km": config.ruin_recreate.cost_per_km,
            "revenue_kg": config.ruin_recreate.revenue_kg,
            "seed": config.ruin_recreate.seed,
        }
    if method_lower == "adaptive_large_neighborhood_search":
        return {
            "alns_iterations": config.adaptive_lns.alns_iterations,
            "ruin_fraction": config.adaptive_lns.ruin_fraction,
            "alns_bandit_warm_start_path": config.adaptive_lns.alns_bandit_warm_start_path,
            "cost_per_km": config.adaptive_lns.cost_per_km,
            "revenue_kg": config.adaptive_lns.revenue_kg,
            "seed": config.adaptive_lns.seed,
        }
    if method_lower == "multi_phase":
        return {"phases": config.multi_phase.phases, "seed": config.multi_phase.seed}
    if method_lower == "branch_and_price":
        bp = config.branch_and_price
        return {
            "bp_max_iterations": bp.bp_max_iterations,
            "bp_max_routes_per_iteration": bp.bp_max_routes_per_iteration,
            "bp_optimality_gap": bp.bp_optimality_gap,
            "bp_branching_strategy": bp.bp_branching_strategy,
            "bp_max_branch_nodes": bp.bp_max_branch_nodes,
            "bp_use_exact_pricing": bp.bp_use_exact_pricing,
            "bp_use_ng_routes": bp.bp_use_ng_routes,
            "bp_ng_neighborhood_size": bp.bp_ng_neighborhood_size,
            "bp_tree_search_strategy": bp.bp_tree_search_strategy,
            "bp_vehicle_limit": bp.bp_vehicle_limit,
            "bp_cleanup_frequency": bp.bp_cleanup_frequency,
            "bp_cleanup_threshold": bp.bp_cleanup_threshold,
            "bp_early_termination_gap": bp.bp_early_termination_gap,
            "bp_allow_heuristic_ryan_foster": bp.bp_allow_heuristic_ryan_foster,
            "bp_time_limit": bp.bp_time_limit,
            "bp_use_cspy": bp.bp_use_cspy,
            "seed": bp.seed,
        }
    if method_lower in ("dp", "dp_route_reopt"):
        return {"dp_max_nodes": config.dp_max_nodes}
    if method_lower == "fix_and_optimize":
        return {
            "fo_n_free": config.fix_and_optimize.fo_n_free,
            "fo_free_fraction": config.fix_and_optimize.fo_free_fraction,
            "fo_time_limit": config.fix_and_optimize.fo_time_limit,
            "seed": config.fix_and_optimize.seed,
        }
    if method_lower == "learned":
        return {
            "learned_weights_path": config.learned.learned_weights_path,
            "learned_max_iter": config.learned.learned_max_iter,
            "learned_min_improvement": config.learned.learned_min_improvement,
            "learned_neighborhood_size": config.learned.learned_neighborhood_size,
            "seed": config.learned.seed,
        }
    if method_lower == "set_partitioning":
        return {
            "sp_n_perturbations": config.set_partitioning.sp_n_perturbations,
            "sp_include_dp": config.set_partitioning.sp_include_dp,
            "sp_time_limit": config.set_partitioning.sp_time_limit,
            "ruin_fraction": config.set_partitioning.ruin_fraction,
            "seed": config.set_partitioning.seed,
        }
    if method_lower == "set_partitioning_polish":
        return {
            "route_pool": config.set_partitioning_polish.route_pool,
            "sp_time_limit": config.set_partitioning_polish.sp_time_limit,
            "seed": config.set_partitioning_polish.seed,
        }
    if method_lower == "neural_selector":
        return {"config": config}
    return {}


class RouteImproverFactory:
    """Create route improvers by registry name."""

    @staticmethod
    def create(name: str, **kwargs: Any) -> IRouteImprovement:
        if RouteImproverRegistry.get(name) is None:
            raise ValueError(f"Unknown route improver: {name}")
        inst = IMPROVEMENT.create(name, **kwargs)
        if kwargs and isinstance(getattr(inst, "config", None), dict):
            inst.config.update(kwargs)
        return inst

    @classmethod
    def create_from_config(cls, config: Any) -> List[IRouteImprovement]:
        processors: List[IRouteImprovement] = []
        if not config.methods:
            return processors
        for method in config.methods:
            processors.append(cls.create(method, **_improver_kwargs(method, config)))
        return processors
