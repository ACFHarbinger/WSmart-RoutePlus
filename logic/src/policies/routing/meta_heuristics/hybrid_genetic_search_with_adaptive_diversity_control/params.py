"""
Configuration parameters for Hybrid Genetic Search Adaptive Diversity Control.

Attributes:
    HGSADCParams: Data class for HGS-ADC parameters.

Example:
    >>> params = HGSADCParams(pop_size=50)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.src.configs.policies import HGSADCConfig


@dataclass
class HGSADCParams:
    """
    Parameters for the HGSADC algorithm.

    Attributes:
        pop_size: Population size.
        mu: Number of elitist parents.
        nb_close: Number of nearest neighbors to consider for elitism.
        generations: Number of generations.
        n_vehicles: Number of vehicles.

    Example:
        >>> params = HGSADCParams()
    """

    pop_size: int = 25
    mu: int = 25
    nb_close: int = 4
    generations: int = 50
    n_vehicles: int = 0

    @classmethod
    def from_config(cls, config: "HGSADCConfig") -> "HGSADCParams":
        """
        Create params from config.

        Args:
            config: Configuration object.

        Returns:
            HGSADCParams: Parameters instance.
        """
        return cls(
            pop_size=getattr(config, "pop_size", 25),
            mu=getattr(config, "mu", 25),
            nb_close=getattr(config, "nb_close", 4),
            generations=getattr(config, "generations", 50),
            n_vehicles=getattr(config, "n_vehicles", 0),
        )
