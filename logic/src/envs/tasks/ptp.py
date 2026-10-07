"""
PTP and MVPTP problem definitions.

Attributes:
    PTP: Profitable Tour Problem (PTP) definition.

Example:
    >>> import torch
    >>> from logic.src.envs.tasks.ptp import PTP
    >>> dataset = {
    ...     "locs": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
    ...     "waste": torch.tensor([[0.0, 10.0]]),
    ...     "depot": torch.tensor([0.0]),
    ...     "cost_km": 1.0,
    ...     "revenue_kg": 2.0,
    ... }
    >>> pi = torch.tensor([[[0, 1, 0]]])
    >>> length, cost_dict, _ = PTP.get_costs(dataset, pi)
    >>> print(length)
    tensor([-2.0])
"""

import torch

from logic.src.constants.tasks import COST_KM, REVENUE_KG
from logic.src.envs.tasks.base import BaseProblem


class PTP(BaseProblem):
    """
    Profitable Tour Problem (PTP).

    Objective: Maximize Profit (Revenue - Cost).

    Attributes:
        NAME: Environment name identifier.
    """

    NAME = "ptp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Compute PTP costs/rewards.

        Args:
            dataset: Problem data.
            pi: Tours [batch, nodes].
            cw_dict: Cost weights dictionary.
            dist_matrix: Optional distance matrix.

        Returns:
            Tuple of (negative_profit, cost_dict, None).
        """
        PTP.validate_tours(pi)
        if pi.size(-1) == 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return (
                z,
                {"length": z, "waste": z, "overflows": z, "total": z},
                None,
            )

        waste_with_depot = PTP.get_waste_with_depot(dataset, pi)
        w = waste_with_depot.gather(1, pi)
        if "max_waste" in dataset:
            w = w.clamp(max=dataset["max_waste"][:, None])
        waste = w.sum(dim=-1)
        length = PTP.get_tour_length(dataset, pi, dist_matrix)

        cost_km = dataset.get("cost_km", COST_KM)
        revenue_kg = dataset.get("revenue_kg", REVENUE_KG)

        neg_profit = length * cost_km - waste * revenue_kg
        if cw_dict is not None:
            neg_profit = cw_dict.get("length", 1.0) * length * cost_km - cw_dict.get("waste", 1.0) * waste * revenue_kg

        return (
            neg_profit,
            {"length": length, "waste": waste, "overflows": torch.zeros_like(neg_profit), "total": neg_profit},
            None,
        )
