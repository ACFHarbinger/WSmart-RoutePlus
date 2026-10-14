r"""Evolution strategies for Memetic Differential Evolution (MDE).

This module implements the Strategy Pattern to support different approaches for
handling local search optimization within the MDE framework.

Attributes:
    EvolutionStrategy: Abstract base class for evolution strategies.
    BaldwinianStrategy: Baldwinian evolution strategy.
    LamarckianStrategy: Lamarckian evolution strategy.
    create_evolution_strategy: Factory function to create evolution strategy instances.

Example:
    >>> strategy = create_evolution_strategy("lamarckian")
"""

from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np


class EvolutionStrategy(ABC):
    """
    Defines the interface for determining which continuous vector survives
    the selection process when local search has been applied.

    Attributes:
        None.
    """

    @abstractmethod
    def get_surviving_vector(
        self,
        original_vector: np.ndarray,
        optimized_routes: List[List[int]],
        encoder_func: Callable[[List[List[int]]], np.ndarray],
    ) -> np.ndarray:
        """
        Determine which continuous vector should survive selection.

        Args:
            original_vector: The continuous trial vector before local search.
            optimized_routes: The discrete routes after local search optimization.
            encoder_func: Function to encode discrete routes back to continuous vector.

        Returns:
            The continuous vector that should replace the target in the population.
        """
        pass


class BaldwinianStrategy(EvolutionStrategy):
    """
    Baldwinian evolution strategy.

    The improved fitness from local search is used for selection, but the
    genetic material (continuous Random Key vector) remains unchanged.

    Characteristics:
        - Phenotypic learning: Only observable traits improve
        - Genotypic stability: Genetic encoding is preserved
        - Slower convergence: Good solutions must be rediscovered each generation
        - Better diversity: Population maintains genetic variety

    Use When:
        - You want to maintain population diversity
        - You suspect premature convergence issues
        - Local search may produce overfitted solutions
        - Genotype-phenotype mapping is expensive

    Attributes:
        None.
    """

    def get_surviving_vector(
        self,
        original_vector: np.ndarray,
        optimized_routes: List[List[int]],
        encoder_func: Callable[[List[List[int]]], np.ndarray],
    ) -> np.ndarray:
        """
        Return the original continuous vector unchanged.

        Args:
            original_vector: The continuous trial vector before local search.
            optimized_routes: Ignored in Baldwinian strategy.
            encoder_func: Ignored in Baldwinian strategy.

        Returns:
            The original continuous vector (genetic material unchanged).
        """
        return original_vector


class LamarckianStrategy(EvolutionStrategy):
    """
    Lamarckian evolution strategy.

    The genetic material (continuous Random Key vector) is updated to reflect
    the improvements discovered by local search.

    Characteristics:
        - Genotypic learning: Genetic code is modified by experience
        - Faster convergence: Good solutions are directly encoded
        - Potential loss of diversity: Population may converge quickly
        - Direct inheritance: Offspring start with learned improvements

    Use When:
        - You want faster convergence to high-quality solutions
        - Local search consistently produces improvements
        - Computational budget is limited
        - You have a robust encoder that preserves solution quality

    Attributes:
        None.
    """

    def get_surviving_vector(
        self,
        original_vector: np.ndarray,
        optimized_routes: List[List[int]],
        encoder_func: Callable[[List[List[int]]], np.ndarray],
    ) -> np.ndarray:
        """
        Reverse-encode the optimized routes into a continuous vector.

        Args:
            original_vector: Ignored in Lamarckian strategy.
            optimized_routes: The discrete routes after local search optimization.
            encoder_func: Function to encode discrete routes back to continuous vector.

        Returns:
            A new continuous vector encoding the optimized solution.
        """
        return encoder_func(optimized_routes)


def create_evolution_strategy(strategy_name: str) -> EvolutionStrategy:
    """
    Factory function to create evolution strategy instances.

    Args:
        strategy_name: Name of the strategy. Options: "baldwinian", "lamarckian"

    Returns:
        An instance of the requested evolution strategy.

    Raises:
        ValueError: If strategy_name is not recognized.

    Examples:
        >>> strategy = create_evolution_strategy("lamarckian")
        >>> isinstance(strategy, LamarckianStrategy)
        True

        >>> strategy = create_evolution_strategy("baldwinian")
        >>> isinstance(strategy, BaldwinianStrategy)
        True
    """
    strategy_name_lower = strategy_name.lower().strip()

    if strategy_name_lower == "baldwinian":
        return BaldwinianStrategy()
    elif strategy_name_lower == "lamarckian":
        return LamarckianStrategy()
    else:
        raise ValueError(f"Unknown evolution strategy: '{strategy_name}'. Valid options: 'baldwinian', 'lamarckian'")
