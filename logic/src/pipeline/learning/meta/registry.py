"""
Meta-Learning Strategy Registry.

Attributes:
    META_STRATEGY_REGISTRY (dict[str, type[WeightAdjustmentStrategy]]): Dictionary of meta-learning strategies.
    get_meta_strategy (callable): Function to get meta-learning strategy by name.

Example:
    >>> from logic.src.pipeline.learning.meta import get_meta_strategy
    >>> strategy = get_meta_strategy("bandit")
    >>> strategy
    <logic.src.pipeline.learning.meta.contextual_bandits.WeightContextualBandit object at 0x...>
"""

from typing import Dict, Type

from logic.src.pipeline.learning.meta.contextual_bandits import WeightContextualBandit
from logic.src.pipeline.learning.meta.hypernet_strategy import HyperNetworkStrategy
from logic.src.pipeline.learning.meta.multi_objective.weight_optimizer import MORLWeightOptimizer
from logic.src.pipeline.learning.meta.td_learning import CostWeightManager
from logic.src.pipeline.learning.meta.weight_optimizer import RewardWeightOptimizer
from logic.src.pipeline.learning.meta.weight_strategy import WeightAdjustmentStrategy

META_STRATEGY_REGISTRY: Dict[str, Type[WeightAdjustmentStrategy]] = {
    "rnn": RewardWeightOptimizer,
    "rwa": RewardWeightOptimizer,
    "bandit": WeightContextualBandit,
    "morl": MORLWeightOptimizer,
    "tdl": CostWeightManager,
    "hypernet": HyperNetworkStrategy,
}


def get_meta_strategy(name: str, **kwargs) -> WeightAdjustmentStrategy:
    """Get meta-learning strategy by name.

    Args:
        name: Name of the meta-learning strategy to retrieve.
        kwargs: Additional keyword arguments to pass to the strategy constructor.

    Returns:
        WeightAdjustmentStrategy: The meta-learning strategy.
    """
    strategy_cls = META_STRATEGY_REGISTRY.get(name.lower())
    if strategy_cls is None:
        raise ValueError(f"Unknown meta strategy: {name}")
    return strategy_cls(**kwargs)
