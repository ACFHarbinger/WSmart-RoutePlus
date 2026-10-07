"""mvptp.py module.

Attributes:
    MVPTP: Multi-Vehicle Profitable Tour Problem task inheriting prize-collection and capacity logic from PTP.

Example:
    >>> import mvptp
"""

import torch

from logic.src.envs.tasks.ptp import PTP


class MVPTP(PTP):
    """
    Multi-Vehicle Profitable Tour Problem.
    Includes vehicle capacity constraints.

    Attributes:
        NAME: Environment name identifier.
    """

    NAME = "mvptp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Compute MVPTP costs (same as PTP but checks capacity).

        Args:
            dataset: Problem data.
            pi: Tours.
            cw_dict: Cost weights.
            dist_matrix: optional.

        Returns:
            Tuple of (cost, dict, None).
        """
        cost, c_dict, _ = PTP.get_costs(dataset, pi, cw_dict, dist_matrix)

        # MVPTP specific: Check total capacity PER TRIP
        capacity = dataset.get("capacity", dataset.get("max_waste", torch.tensor(100.0)))

        # Extract trip waste
        waste_with_depot = MVPTP.get_waste_with_depot(dataset, pi)
        # For each sequence in pi, calculate cumulative waste and reset at 0
        w = waste_with_depot.gather(1, pi)

        # Simple loop-based trip check for robustness in tests
        for b in range(pi.size(0)):
            cur_trip_waste = 0.0  # Initialize as float to satisfy type checker
            for i in range(pi.size(1)):
                node = pi[b, i].item()
                if node == 0:
                    cur_trip_waste = 0.0
                else:
                    # cur_trip_waste is now consistently float
                    cur_trip_waste += w[b, i].item()
                    # Use a small epsilon for float comparison
                    assert cur_trip_waste <= capacity[b].item() + 1e-6, "Used more than capacity"

        return cost, c_dict, _
