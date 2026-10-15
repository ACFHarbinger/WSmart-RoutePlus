"""NDS-BRKGA Joint Policy.

Orchestrates the full NDS-BRKGA algorithm, resolving the joint mandatory-bin
selection and routing problem in a single population-based search.

Attributes:
    NDSBRKGAPolicy: The evolutionary policy class.

Example:
    >>> policy = NDSBRKGAPolicy()
    >>> selection, routes, profit, cost = policy.solve_joint(context)
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.nds_brkga import NDSBRKGAConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.joint_context import JointSelectionConstructionContext
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.packing_and_routing.base.base_joint_policy import BaseJointPolicy
from logic.src.policies.packing_and_routing.base.registry import JointPolicyRegistry
from logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome import (
    compute_adaptive_thresholds,
)
from logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives import (
    compute_overflow_risk,
    evaluate_population,
)
from logic.src.configs.policies.nds_brkga import NDSBRKGAConfig
from logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.population import (
    Population,
)


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.EVOLUTIONARY_ALGORITHM,
    PolicyTag.PACKING,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.STOCHASTIC,
    PolicyTag.SINGLE_PERIOD,
    PolicyTag.ANYTIME,
    PolicyTag.PACKING_AND_ROUTING,
)
@JointPolicyRegistry.register("nds_brkga")
@RoutingAlgorithmRegistry.register("nds_brkga")
class NDSBRKGAPolicy(BaseJointPolicy):
    """Non-Dominated Sorting Biased Random-Key Genetic Algorithm Policy.

    Jointly optimises mandatory-packing and routing in a
    single population-based evolutionary loop, maintaining a Pareto front
    over profit, overflow risk, and distance.

    Attributes:
        config (Optional[Union[NDSBRKGAConfig, Dict[str, Any]]]): Configuration source.
    """

    def __init__(self, config: Optional[Union[NDSBRKGAConfig, Dict[str, Any]]] = None):
        """Initialize NDSBRKGAPolicy.

        Args:
            config: Configuration source.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the dataclass type for config parsing.

        Returns:
            Optional[Type]: The NDSBRKGAConfig type.
        """
        return NDSBRKGAConfig

    def _get_config_key(self) -> str:
        """Return the Hydra configuration key.

        Returns:
            str: The configuration key string 'nds_brkga'.
        """
        return "nds_brkga"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Internal bridge for test compatibility.

        Args:
            sub_dist_matrix: Local distance matrix.
            sub_wastes: Waste demands per node.
            capacity: Vehicle capacity.
            revenue: Revenue per kg.
            cost_unit: Cost per km.
            values: Problem parameters.
            mandatory_nodes: Required node IDs.
            kwargs: Additional simulation context.

        Returns:
            Tuple[List[List[int]], float, float]:
                - routes: Constructed routes.
                - profit: Calculated net profit.
                - cost: Total routing cost.
        """
        # Convert sub_wastes (dict) to current_fill (list)
        # nodes IDs are 1-based in sub_wastes keys
        max_id = max(sub_wastes.keys()) if sub_wastes else 0
        current_fill = np.zeros(max_id)
        for i, f in sub_wastes.items():
            current_fill[i - 1] = f

        context = JointSelectionConstructionContext(
            bin_ids=np.arange(1, max_id + 1, dtype=np.int32),
            current_fill=current_fill,
            distance_matrix=sub_dist_matrix,
            capacity=capacity,
            revenue_kg=revenue,
            cost_per_km=cost_unit,
            bin_density=float(values.get("B", 1.0)),
            bin_volume=float(values.get("V", 1.0)),
            max_fill=float(values.get("max_fill", 100.0)),
            n_vehicles=int(kwargs.get("n_vehicles", 1)),
            scenario_tree=kwargs.get("scenario_tree"),
            mandatory_override=[mandatory_nodes] if mandatory_nodes else None,
        )

        _, routes, profit, cost = self.solve_joint(context)
        return routes, profit, cost

    def solve_joint(
        self,
        context: JointSelectionConstructionContext,
    ) -> Tuple[List[int], List[List[int]], float, float]:
        """Run the NDS-BRKGA optimisation loop using inputs from the JointContext.

        Args:
            context: The search and problem context.

        Returns:
            Tuple[List[int], List[List[int]], float, float]:
                - selected_bins: List of unique bin IDs selected.
                - local_routes: Constructed routes (local node indices).
                - total_profit: Net profit (Revenue - Distance Cost).
                - total_cost: Total routing cost.
        """
        params = self._effective_config(asdict(self._config)) if self._config else NDSBRKGAConfig()
        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], [], 0.0, 0.0

        rng = np.random.default_rng(params.seed)
        start_time = time.perf_counter()

        # ----------------------------------------------------------------
        # Phase 0: Pre-computation (Risk & Adaptive Thresholds)
        # ----------------------------------------------------------------
        bin_mass = context.bin_mass_kg()
        overflow_risk = compute_overflow_risk(
            context.current_fill,
            bin_mass,
            context.scenario_tree,
            context.overflow_penalty_frac,
        )
        thresholds = compute_adaptive_thresholds(
            overflow_risk,
            threshold_min=params.selection_threshold_min,
            threshold_max=params.selection_threshold_max,
        )

        # ----------------------------------------------------------------
        # Phase 1: Initialise population
        # ----------------------------------------------------------------
        # Revenue scaled for population logic: R = revenue per 1% fill unit
        R_scaled = context.revenue_scaled()
        costs_scaled = context.cost_per_km

        pop = Population.initialise(
            n_bins=n_bins,
            thresholds=thresholds,
            dist_matrix=context.distance_matrix,
            wastes={i: context.current_fill[i - 1] for i in range(1, n_bins + 1)},
            capacity=context.capacity,
            R=R_scaled,
            C=costs_scaled,
            overflow_risk=overflow_risk,
            current_fill=context.current_fill,
            bin_density=context.bin_density,
            bin_volume=context.bin_volume,
            max_fill=context.max_fill,
            overflow_penalty_frac=context.overflow_penalty_frac,
            scenario_tree=context.scenario_tree,
            params=params,
            mandatory_override=context.mandatory_override or [],
        )

        # Track global best
        best_chrom, best_obj = pop.best_chromosome()
        best_neg_profit = best_obj[0]

        # ----------------------------------------------------------------
        # Phase 2: Generational loop
        # ----------------------------------------------------------------
        for _gen in range(params.max_generations):
            elapsed = time.perf_counter() - start_time
            if params.time_limit > 0 and elapsed >= params.time_limit:
                break

            new_chromosomes = pop.breed_next_generation(
                n_elite=params.n_elite,
                n_mutants=params.n_mutants,
                bias_elite=params.bias_elite,
                rng=rng,
            )

            new_objectives = evaluate_population(
                new_chromosomes,
                thresholds,
                context.distance_matrix,
                {i: context.current_fill[i - 1] for i in range(1, n_bins + 1)},
                context.capacity,
                R_scaled,
                costs_scaled,
                overflow_risk,
                params.overflow_penalty,
                context.mandatory_override or [],
            )

            from logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2 import (
                select_elite_nsga2,
            )

            _, new_ranks = select_elite_nsga2(new_objectives, params.n_elite)
            pop = Population(new_chromosomes, new_objectives, new_ranks)

            gen_best_chrom, gen_best_obj = pop.best_chromosome()
            if gen_best_obj[0] < best_neg_profit:
                best_chrom = gen_best_chrom
                best_obj = gen_best_obj
                best_neg_profit = gen_best_obj[0]

        # ----------------------------------------------------------------
        # Phase 3: Decode best
        # ----------------------------------------------------------------
        routes = best_chrom.to_routes(
            thresholds,
            {i: context.current_fill[i - 1] for i in range(1, n_bins + 1)},
            context.capacity,
            context.mandatory_override or [],
        )
        selected_bins = sorted({b for route in routes for b in route})

        # Calculate final profit and cost for the return 4-tuple
        visited = set(selected_bins)
        revenue_total = sum(context.current_fill[b - 1] * R_scaled for b in visited)

        total_dist = 0.0
        dm = context.distance_matrix
        for route in routes:
            if not route:
                continue
            total_dist += dm[0, route[0]]
            for k in range(len(route) - 1):
                total_dist += dm[route[k], route[k + 1]]
            total_dist += dm[route[-1], 0]

        cost_total = total_dist * costs_scaled
        profit_total = revenue_total - cost_total

        return selected_bins, routes, profit_total, cost_total
