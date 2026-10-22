"""Hybrid Genetic Search with Adaptive Diversity Control (HGS-ADC) configuration.

Attributes:
    HGSADCConfig: Configuration for the HGS-ADC policy.

Example:
    >>> from configs.policies.hgs_adc import HGSADCConfig
    >>> config = HGSADCConfig()
    >>> config.pop_size
    25
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HGSADCConfig:
    """
    HGS-ADC configuration parameters.

    Attributes:
        pop_size (int): Population size.
        mu (int): Number of parents selected for reproduction.
        nb_close (int): Number of nearest neighbors for local search.
        generations (int): Number of generations.
        n_vehicles (int): Number of vehicles.
    """

    pop_size: int = 25
    mu: int = 25
    nb_close: int = 4
    generations: int = 50
    n_vehicles: int = 0


    @classmethod
    def from_config(cls, config: "HGSADCConfig") -> "HGSADCConfig":
        """
        Create params from config.

        Args:
            config: Configuration object.

        Returns:
            HGSADCConfig: Parameters instance.
        """
        return cls(
            pop_size=getattr(config, "pop_size", 25),
            mu=getattr(config, "mu", 25),
            nb_close=getattr(config, "nb_close", 4),
            generations=getattr(config, "generations", 50),
            n_vehicles=getattr(config, "n_vehicles", 0),
        )
