"""
NDS-BRKGA Configuration for Hydra.

This dataclass mirrors :class:`~logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams`
with Hydra-compatible field types.  It is referenced by the Hydra config
system when ``policy=nds_brkga`` is specified on the command line.

Usage::

    python main.py test_sim \\
        --policies nds_brkga \\
        policy.nds_brkga.pop_size=80 \\
        policy.nds_brkga.time_limit=120

Attributes:
    NDSBRKGAConfig: Configuration for the NDS-BRKGA policy.

Example:
    >>> from configs.policies.nds_brkga import NDSBRKGAConfig
    >>> config = NDSBRKGAConfig()
    >>> config.pop_size
    60
    >>> config.max_generations
    200
    >>> config.time_limit
    90.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class NDSBRKGAConfig:
    """
    Hydra configuration for the NDS-BRKGA joint packing-and-routing policy.

    Attributes:
        pop_size: Total population size per generation.
        n_elite: Number of elite chromosomes preserved via NSGA-II each generation.
        n_mutants: Number of random mutant chromosomes injected each generation.
        bias_elite: Probability of inheriting each gene from the elite parent
            during biased uniform crossover.
        max_generations: Maximum number of evolutionary generations.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed for reproducibility.
        overflow_penalty: Weight applied to the overflow-cost objective.
        seed_selection_strategy: Name of the packing algorithm used
            to generate seeded initial chromosomes.
        seed_routing_strategy: Name of the routing operator used to build the
            routing component of seed chromosomes (``"greedy"`` uses the
            built-in nearest-neighbour heuristic).
        n_seed_solutions: Number of seed solutions per sub-problem strategy.
        selection_threshold_min: Threshold for max-risk bins (almost always selected).
        selection_threshold_max: Threshold for zero-risk bins (rarely selected).
        packing: Unused placeholder for Hydra config composition
            compatibility.
        route_improvement: Unused placeholder for Hydra config composition
            compatibility.
    """

    pop_size: int = 60
    n_elite: int = 15
    n_mutants: int = 10
    bias_elite: float = 0.70
    max_generations: int = 200
    time_limit: float = 90.0
    seed: Optional[int] = 42
    overflow_penalty: float = 10.0

    seed_selection_strategy: str = "fractional_knapsack"
    seed_routing_strategy: str = "greedy"
    n_seed_solutions: int = 5

    selection_threshold_min: float = 0.10
    selection_threshold_max: float = 0.90

    # Hydra compatibility placeholders
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)


    @classmethod
    def from_config(cls, config: object) -> "NDSBRKGAConfig":
        """Construct from a configuration object (dataclass or dict-like).

        Args:
            config (object): Any object with attribute or dict-key access.

        Returns:
            NDSBRKGAConfig: Populated parameter instance.
        """

        def _get(attr: str, default):  # type: ignore[no-untyped-def]
            if isinstance(config, dict):
                return config.get(attr, default)
            return getattr(config, attr, default)

        return cls(
            pop_size=int(_get("pop_size", 60)),
            n_elite=int(_get("n_elite", 15)),
            n_mutants=int(_get("n_mutants", 10)),
            bias_elite=float(_get("bias_elite", 0.70)),
            max_generations=int(_get("max_generations", 200)),
            time_limit=float(_get("time_limit", 90.0)),
            seed=_get("seed", 42),
            overflow_penalty=float(_get("overflow_penalty", 10.0)),
            seed_selection_strategy=str(_get("seed_selection_strategy", "fractional_knapsack")),
            seed_routing_strategy=str(_get("seed_routing_strategy", "greedy")),
            n_seed_solutions=int(_get("n_seed_solutions", 5)),
            selection_threshold_min=float(_get("selection_threshold_min", 0.10)),
            selection_threshold_max=float(_get("selection_threshold_max", 0.90)),
        )
