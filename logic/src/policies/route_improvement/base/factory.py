"""Factory for route improvers. Construction is ``IMPROVEMENT.create``."""

from typing import Any, Callable, Dict, List, Tuple

from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.utils.registry import IMPROVEMENT

from .registry import RouteImproverRegistry


def _picked(source: Any, names: Tuple[str, ...]) -> Dict[str, Any]:
    return {name: getattr(source, name) for name in names}


def _from_section(section_name: str, fields: Tuple[str, ...]) -> Callable[[Any], Dict[str, Any]]:
    def build(config: Any) -> Dict[str, Any]:
        return _picked(getattr(config, section_name), fields)

    return build


def _annealing_kwargs(config: Any) -> Dict[str, Any]:
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


def _ruin_recreate_kwargs(config: Any) -> Dict[str, Any]:
    ruin = config.ruin_recreate
    return {
        "lns_iterations": ruin.lns_iterations,
        "ruin_fraction": ruin.ruin_fraction,
        "lns_acceptance": ruin.acceptance_criterion.method,
        "repair_k": ruin.repair_k,
        "cost_per_km": ruin.cost_per_km,
        "revenue_kg": ruin.revenue_kg,
        "seed": ruin.seed,
    }


def _branch_and_price_kwargs(config: Any) -> Dict[str, Any]:
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


def _dp_kwargs(config: Any) -> Dict[str, Any]:
    return {"dp_max_nodes": config.dp_max_nodes}


def _neural_selector_kwargs(config: Any) -> Dict[str, Any]:
    return {"config": config}


_FAST_TSP = _from_section("fast_tsp", ("time_limit", "seed", "directed_two_opt"))
_LKH = _from_section("lkh", ("max_iterations", "time_limit", "seed"))
_LOCAL_SEARCH = _from_section("local_search", ("ls_operator", "iterations", "time_limit", "seed"))
_RANDOM_SEARCH = _from_section("random_local_search", ("iterations", "params", "time_limit", "seed"))
_INSERTION = _from_section("insertion", ("cost_per_km", "revenue_kg", "n_bins", "seed"))
_IMPROVER_BUILDERS: Dict[str, Callable[[Any], Dict[str, Any]]] = {
    "fast_tsp": _FAST_TSP,
    "tsp": _FAST_TSP,
    "ftsp": _FAST_TSP,
    "lkh": _LKH,
    "lkh3": _LKH,
    "lkh-3": _LKH,
    "lin_kernighan_helsgaun": _LKH,
    "local_search": _LOCAL_SEARCH,
    "classical_local_search": _LOCAL_SEARCH,
    "cls": _LOCAL_SEARCH,
    "random": _RANDOM_SEARCH,
    "random_local_search": _RANDOM_SEARCH,
    "rls": _RANDOM_SEARCH,
    "path": _from_section("path", ("vehicle_capacity",)),
    "or_opt": _from_section("or_opt", ("chain_len", "iterations", "seed")),
    "cross_exchange": _from_section(
        "cross_exchange", ("cross_exchange_max_segment_len", "iterations", "seed")
    ),
    "guided_local_search": _from_section(
        "guided_local_search",
        ("gls_iterations", "gls_inner_iterations", "gls_lambda_factor", "gls_base_operator", "seed"),
    ),
    "simulated_annealing": _annealing_kwargs,
    "cheapest_insertion": _INSERTION,
    "regret_k_insertion": _from_section("insertion", ("regret_k", "cost_per_km", "revenue_kg", "n_bins", "seed")),
    "profitable_detour": _from_section(
        "insertion", ("detour_epsilon", "cost_per_km", "revenue_kg", "n_bins", "seed")
    ),
    "ruin_recreate": _ruin_recreate_kwargs,
    "adaptive_large_neighborhood_search": _from_section(
        "adaptive_lns",
        (
            "alns_iterations",
            "ruin_fraction",
            "alns_bandit_warm_start_path",
            "cost_per_km",
            "revenue_kg",
            "seed",
        ),
    ),
    "multi_phase": _from_section("multi_phase", ("phases", "seed")),
    "branch_and_price": _branch_and_price_kwargs,
    "dp": _dp_kwargs,
    "dp_route_reopt": _dp_kwargs,
    "fix_and_optimize": _from_section(
        "fix_and_optimize", ("fo_n_free", "fo_free_fraction", "fo_time_limit", "seed")
    ),
    "learned": _from_section(
        "learned",
        (
            "learned_weights_path",
            "learned_max_iter",
            "learned_min_improvement",
            "learned_neighborhood_size",
            "seed",
        ),
    ),
    "set_partitioning": _from_section(
        "set_partitioning",
        ("sp_n_perturbations", "sp_include_dp", "sp_time_limit", "ruin_fraction", "seed"),
    ),
    "set_partitioning_polish": _from_section(
        "set_partitioning_polish", ("route_pool", "sp_time_limit", "seed")
    ),
    "neural_selector": _neural_selector_kwargs,
}


def _improver_kwargs(method: str, config: Any) -> Dict[str, Any]:
    """Read the config subsection for one improver method."""
    builder = _IMPROVER_BUILDERS.get(method.lower())
    if builder is None:
        return {}
    return builder(config)


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
