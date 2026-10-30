"""Scenario plan of the multi-scenario benchmark campaign.

A campaign is a list of independent waste accumulation scenarios. Each scenario fixes the depot
placement, the seed and the number of samples of one simulator invocation, and renders them as
Hydra overrides. Scenarios use disjoint seed ranges, so their deposits are independent.

Attributes:
    CampaignScenario: One simulator invocation of the campaign.
    benchmark_campaign: Scenarios split evenly between remote and central depots.

Example:
    >>> [s.depot_mode for s in benchmark_campaign(n_per_depot=1)]
    ['remote', 'centroid']
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

DEFAULT_DEPOT_MODES: Tuple[str, ...] = ("remote", "centroid")


@dataclass(frozen=True)
class CampaignScenario:
    """One simulator invocation of the campaign.

    Attributes:
        depot_mode: Depot placement (``sim.graph.depot_mode``).
        seed: Seed of the first sample (``sim.seed``); sample ``i`` uses ``seed + i``.
        n_samples: Independent deposit realizations of this invocation (``sim.graph.n_samples``).
    """

    depot_mode: str
    seed: int
    n_samples: int

    def sample_seeds(self) -> List[int]:
        """Seeds of the deposit realizations of this scenario.

        Returns:
            List[int]: ``seed + i`` for each sample ``i``.
        """
        return [self.seed + i for i in range(self.n_samples)]

    def overrides(self) -> List[str]:
        """Hydra overrides that select this scenario.

        Returns:
            List[str]: ``key=value`` overrides for the simulator task.
        """
        return [
            f"sim.graph.depot_mode={self.depot_mode}",
            f"sim.seed={self.seed}",
            f"sim.graph.n_samples={self.n_samples}",
        ]


def benchmark_campaign(
    n_per_depot: int = 5,
    base_seed: int = 42,
    depot_modes: Tuple[str, ...] = DEFAULT_DEPOT_MODES,
) -> List[CampaignScenario]:
    """Build the campaign: ``n_per_depot`` independent scenarios for each depot placement.

    Args:
        n_per_depot: Scenarios per depot placement.
        base_seed: Seed of the first scenario.
        depot_modes: Depot placements, in run order.

    Returns:
        List[CampaignScenario]: One scenario per depot placement, with disjoint seed ranges.

    Raises:
        ValueError: If ``n_per_depot`` is not positive.
    """
    if n_per_depot < 1:
        raise ValueError("n_per_depot must be positive")
    return [
        CampaignScenario(depot_mode=mode, seed=base_seed + k * n_per_depot, n_samples=n_per_depot)
        for k, mode in enumerate(depot_modes)
    ]
