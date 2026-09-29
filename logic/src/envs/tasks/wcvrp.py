"""wcvrp.py module.

Attributes:
    WCVRP: Waste Collection VRP task definition for offline route evaluation.

Example:
    >>> import wcvrp
"""

import torch

from logic.src.envs.tasks.base import BaseProblem


class WCVRP(BaseProblem):
    """
    Waste Collection VRP (WCVRP).

    Objective: Minimize Overflow + TourLength - CollectedWaste.

    Attributes:
        NAME: Environment name identifier.
    """

    NAME = "wcvrp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Compute WCVRP costs.

        Args:
            dataset: Problem data.
            pi: Tours.
            cw_dict: Cost weights.
            dist_matrix: Optional distance matrix.

        Returns:
            Tuple of (cost, dict, None).
        """
        WCVRP.validate_tours(pi)
        if pi.size(-1) == 1:
            overflows = (dataset["waste"] >= dataset.get("max_waste", 1.0)).float().sum(-1)
            return (
                overflows,
                {
                    "overflows": overflows,
                    "length": torch.zeros_like(overflows),
                    "waste": torch.zeros_like(overflows),
                    "total": overflows,
                },
                None,
            )

        waste_with_depot = WCVRP.get_waste_with_depot(dataset, pi)
        visited_mask = torch.zeros_like(waste_with_depot, dtype=torch.bool)
        visited_mask.scatter_(1, pi, True)

        max_w = dataset.get("max_waste", torch.tensor(1.0, device=dataset["waste"].device))
        if max_w.dim() == 1:
            max_w = max_w.unsqueeze(-1)
        overflow_mask = waste_with_depot >= max_w
        overflows = torch.sum(overflow_mask[:, 1:] & ~visited_mask[:, 1:], dim=-1).float()

        length = WCVRP.get_tour_length(dataset, pi, dist_matrix)
        waste = waste_with_depot.gather(1, pi).clamp(max=max_w).sum(dim=-1)

        cost = overflows + length - waste
        if cw_dict is not None:
            cost = (
                cw_dict.get("overflows", 1.0) * overflows
                + cw_dict.get("length", 1.0) * length
                - cw_dict.get("waste", 1.0) * waste
            )

        return cost, {"overflows": overflows, "length": length, "waste": waste, "total": cost}, None
