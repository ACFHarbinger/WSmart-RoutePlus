"""Random Local Search Route Improver.

This module provides a stochastic local search strategy that applies multiple
different local search operators chosen randomly based on pre-defined or
configured probabilities.

Attributes:
    RandomLocalSearchRouteImprover: Stochastic local search improver class.

Example:
    >>> improver = RandomLocalSearchRouteImprover()
    >>> best_tour, metrics = improver.process(tour, distance_matrix=dm, iterations=100)
"""

from typing import Any, Callable, List, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces import IRouteImprovement
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.policies.helpers.local_search.local_search_sweeps import SweepLocalSearch

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@GlobalRegistry.register(
    PolicyTag.IMPROVEMENT,
    PolicyTag.HEURISTIC,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.STOCHASTIC,
)
@RouteImproverRegistry.register("random_local_search")
class RandomLocalSearchRouteImprover(IRouteImprovement):
    """Performs stochastic local search refinement.

    Applies random operators from the metaheuristic sub-package. Compatible
    with multi-route tours.

    Attributes:
        config (Dict[str, Any]): Configuration parameters.

    Example:
        >>> improver = RandomLocalSearchRouteImprover()
        >>> tour, metrics = improver.process(tour, distance_matrix=dm, max_iter=100)
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """Apply random local search operators stochastically.

        Args:
            tour: The initial tour sequence to refine (includes depot 0s).
            kwargs: Search context, including:
                - distance_matrix (np.ndarray): The distance matrix.
                - iterations (int): Total number of random moves to attempt.
                - params (Dict[str, float]): Probabilities for different operators.
                - seed (int): Random seed.
                - wastes (Dict[int, float]): Bin waste demands.
                - capacity (float): Vehicle capacity.
                - R (float): Revenue per kg.
                - C (float): Cost per km.

        Returns:
            Tuple[List[int], ImprovementMetrics]: The refined tour and performance metrics.
        """

        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or len(tour) < 3:
            return tour, {"algorithm": "RandomLocalSearchRouteImprover"}

        # Get parameters from config (via kwargs) with fallbacks
        n_iterations = kwargs.get("iterations", kwargs.get("n_iterations", self.config.get("iterations", 500)))
        op_probs = kwargs.get(
            "params",
            kwargs.get(
                "op_probs",
                self.config.get(
                    "params",
                    {
                        "two_opt": 0.2,
                        "two_opt_star": 0.2,
                        "swap": 0.15,
                        "swap_star": 0.15,
                        "relocate": 0.15,
                        "three_opt": 0.1,
                        "or_opt": 0.05,
                    },
                ),
            ),
        )

        seed = kwargs.get("seed", self.config.get("seed", 42))
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        R = kwargs.get("R", self.config.get("R", 1.0))
        C = kwargs.get("C", self.config.get("C", 1.0))

        dist_matrix = to_numpy(distance_matrix)

        routes = split_tour(tour)
        if not routes:
            return tour, {"algorithm": "RandomLocalSearchRouteImprover"}

        # Initialize SweepLocalSearch
        ls = SweepLocalSearch(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            improvement_threshold=1e-6,
            seed=seed,
        )
        ls.set_routes(routes)

        # Map probabilistic keys to ls methods
        op_map: dict[str, Callable[[], Any]] = {
            "two_opt": ls.two_opt_intra,
            "swap": ls.swap,
            "relocate": ls.relocate,
            "two_opt_star": ls.two_opt_star,
            "swap_star": ls.swap_star,
            "three_opt": ls.three_opt_intra,
            "or_opt": lambda: ls.or_opt(chain_len=2),
            "relocate_chain": lambda: ls.relocate_chain_op(max_chain_len=3),
            "cross_exchange": ls.cross_exchange_op,
        }

        active_ops = [op for op in op_probs.keys() if op in op_map]
        if not active_ops:
            return tour, {"algorithm": "RandomLocalSearchRouteImprover"}

        probs = np.array([op_probs.get(op, 0.0) for op in active_ops])
        probs = probs / (probs.sum() + 1e-10)

        rng = np.random.default_rng(seed)

        try:
            for _ in range(n_iterations):
                op_name = rng.choice(active_ops, p=probs)
                op_func = op_map[op_name]
                op_func()  # Apply the operator (ls handles success internally)

            # Re-assemble tour
            return assemble_tour(ls.get_routes()), {"algorithm": "RandomLocalSearchRouteImprover"}
        except Exception:
            return tour, {"algorithm": "RandomLocalSearchRouteImprover"}
