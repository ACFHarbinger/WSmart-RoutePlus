"""
Hyper-ACO Policy Adapter.

Adapts the Hyper-Heuristic ACO solver to the common policy interface.

Attributes:
    HyperACOPolicy: Hyper-Heuristic ACO policy class.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh import HyperACOPolicy
    >>> policy = HyperACOPolicy()
"""

import random
from typing import Any, Dict, Optional, Type, Union

from logic.src.configs.policies import HyperHeuristicACOConfig
from logic.src.enums import PolicyTag
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.routing.base.generic_policy import GenericRoutingPolicy
from logic.src.policies.routing.base.sub_problem import SubProblem
from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco import (
    HyperHeuristicACO,
)
from logic.src.utils.registry import ALL


@ALL.register(
    PolicyTag.HYPER_HEURISTIC,
    PolicyTag.SWARM_INTELLIGENCE,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("aco_hh")
class HyperACOPolicy(GenericRoutingPolicy):
    """
    Hyper-Heuristic ACO policy class.

    Uses ACO to construct sequences of local search operators.

    Attributes:
        config: HyperHeuristicACOConfig dataclass, raw dict from YAML, or None.
    """

    _solver = HyperHeuristicACO

    def __init__(self, config: Optional[Union[HyperHeuristicACOConfig, Dict[str, Any]]] = None):
        """Initialize Hyper-ACO policy with optional config.

        Args:
            config: HyperHeuristicACOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Get the configuration class.

        Returns:
            Optional[Type]: Configuration class.
        """
        return HyperHeuristicACOConfig

    def _get_config_key(self) -> str:
        """Get the configuration key.

        Returns:
            str: Configuration key.
        """
        return "hyper_aco"

    def _solve(self, problem: SubProblem, cfg: Any) -> Any:
        """Build the greedy warm start, then run the ACO-HH engine on it."""
        initial_routes = build_greedy_routes(
            dist_matrix=problem.dist_matrix,
            wastes=problem.wastes,
            capacity=problem.capacity,
            R=problem.revenue,
            C=problem.cost_unit,
            mandatory_nodes=problem.mandatory_nodes,
            rng=random.Random(cfg.seed) if cfg.seed is not None else random.Random(),
        )
        return type(self)._solver(problem, cfg, initial_routes).solve()
