"""ttop.py module.

Attributes:
    TTOP: Temporal Team Orienteering Problem task, inheriting CVRPP's
        prize-collection objective and per-trip capacity check, and adding
        a per-trip time-budget feasibility check on top.

Example:
    >>> import ttop
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from logic.src.envs.tasks.cvrpp import CVRPP
from logic.src.envs.temporal import get_default_temporal_params


class TTOP(CVRPP):
    """
    Temporal Team Orienteering Problem (TTOP).

    Same objective and capacity constraint as CVRPP (maximize
    waste-collection profit, subject to a per-trip vehicle capacity). TTOP
    adds a second, independent per-trip constraint on top: each trip
    (depot-to-depot leg of the tour) is also bounded by a working-shift
    *time* budget -- travel time plus per-bin service time -- from
    ``SimulationRepository.get_temporal_params()``. Both constraints must
    hold simultaneously; neither replaces the other.

    "Team" here is one vehicle making multiple trips within a period, not a
    concurrent fleet; see the module docstring of
    ``logic.src.envs.routing.ttop`` for the future true-fleet variant.

    Attributes:
        NAME: Environment name identifier.
    """

    NAME = "ttop"

    @staticmethod
    def get_costs(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        cw_dict: Optional[Dict[str, float]],
        dist_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], None]:
        """
        Compute TTOP costs: CVRPP's objective + capacity check, plus a
        per-trip time check.

        Args:
            dataset: Problem data. Optional per-instance overrides
                ``shift_hours``/``avg_speed_kmh``/``service_time_h``
                (each shape (batch,)) take precedence over
                routing defaults.
            pi: Tours [batch, nodes].
            cw_dict: Cost weights dictionary.
            dist_matrix: Optional distance matrix.

        Returns:
            Tuple of (negative_profit, cost_dict, None). cost_dict adds a
            "time" key (total time spent, hours) to CVRPP's
            {"length", "waste", "overflows", "total"}.

        Raises:
            AssertionError: If any trip exceeds its vehicle capacity
                (from CVRPP.get_costs) or its shift time budget.
        """
        cost, c_dict, aux = CVRPP.get_costs(dataset, pi, cw_dict, dist_matrix)

        if pi.size(-1) <= 1:
            c_dict["time"] = torch.zeros_like(cost)
            return cost, c_dict, aux

        default_shift_hours, default_avg_speed_kmh, default_service_time_h = get_default_temporal_params()
        bs = pi.size(0)
        device = pi.device
        shift_hours = dataset.get("shift_hours", torch.full((bs,), default_shift_hours, device=device))
        avg_speed_kmh = dataset.get("avg_speed_kmh", torch.full((bs,), default_avg_speed_kmh, device=device))
        service_time_h = dataset.get("service_time_h", torch.full((bs,), default_service_time_h, device=device))

        depot = dataset["depot"]
        loc_val = dataset.get("locs") if "locs" in dataset else dataset.get("loc")
        loc_with_depot = torch.cat((depot[:, None, :], loc_val), 1)

        # Coordinates in tour order, matching CVRPP's precedent of a
        # simple loop-based per-trip check for correctness/readability over
        # a fully vectorized (and harder to verify) version.  The feasibility
        # calculation must use the same road matrix as the objective whenever
        # one is supplied; otherwise an apparently legal Euclidean tour can
        # exceed the actual driving-time limit.
        coords = loc_with_depot.gather(1, pi.unsqueeze(-1).expand(*pi.size(), 2))
        temporal_distance_matrix = (
            dist_matrix if dist_matrix is not None else dataset.get("dist_matrix", dataset.get("dm"))
        )

        def leg_distance(
            batch: int,
            source: int,
            destination: int,
            source_coord: torch.Tensor,
            destination_coord: torch.Tensor,
        ) -> float:
            if temporal_distance_matrix is None:
                return torch.norm(destination_coord - source_coord).item()
            matrix_batch = batch if temporal_distance_matrix.dim() == 3 else 0
            if temporal_distance_matrix.dim() == 3:
                return temporal_distance_matrix[matrix_batch, source, destination].item()
            return temporal_distance_matrix[source, destination].item()

        time_spent = torch.zeros(bs, device=device)
        for b in range(bs):
            cur_trip_time = 0.0
            prev_coord = depot[b]
            for i in range(pi.size(1)):
                node = pi[b, i].item()
                node_coord = coords[b, i]
                prev_node = pi[b, i - 1].item() if i else 0
                dist = leg_distance(b, prev_node, node, prev_coord, node_coord)
                travel_time = dist / avg_speed_kmh[b].item()
                if node == 0:
                    # Returning to depot pays travel time, then the trip
                    # clock resets for the next outbound leg.
                    cur_trip_time += travel_time
                    assert cur_trip_time <= shift_hours[b].item() + 1e-6, (
                        f"TTOP: trip time {cur_trip_time:.4f}h exceeds shift budget "
                        f"{shift_hours[b].item():.4f}h at batch {b}, step {i}"
                    )
                    time_spent[b] += travel_time
                    cur_trip_time = 0.0
                else:
                    step_time = travel_time + service_time_h[b].item()
                    time_spent[b] += step_time
                    cur_trip_time += step_time
                    assert cur_trip_time <= shift_hours[b].item() + 1e-6, (
                        f"TTOP: trip time {cur_trip_time:.4f}h exceeds shift budget "
                        f"{shift_hours[b].item():.4f}h at batch {b}, step {i}"
                    )
                prev_coord = node_coord
            if pi[b, -1].item() != 0:
                final_return_distance = leg_distance(b, pi[b, -1].item(), 0, prev_coord, depot[b])
                cur_trip_time += final_return_distance / avg_speed_kmh[b].item()
                assert cur_trip_time <= shift_hours[b].item() + 1e-6, (
                    f"TTOP: trip time {cur_trip_time:.4f}h exceeds shift budget "
                    f"{shift_hours[b].item():.4f}h on final return at batch {b}"
                )
                time_spent[b] += final_return_distance / avg_speed_kmh[b].item()

        c_dict["time"] = time_spent
        return cost, c_dict, aux
