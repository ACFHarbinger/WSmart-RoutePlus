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

# NOT a top-level import: see the equivalent comment in
# logic.src.envs.generators.ttop -- importing SimulationRepository at module
# level here closes an import cycle through logic.src.data. Deferred to call
# time in get_costs below.


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
                get_temporal_params()'s defaults.
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

        from logic.src.pipeline.simulations.repository.base import SimulationRepository

        default_shift_hours, default_avg_speed_kmh, default_service_time_h = (
            SimulationRepository.get_temporal_params()
        )
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
        # a fully vectorized (and harder to verify) version.
        coords = loc_with_depot.gather(1, pi.unsqueeze(-1).expand(*pi.size(), 2))

        time_spent = torch.zeros(bs, device=device)
        for b in range(bs):
            cur_trip_time = 0.0
            prev_coord = depot[b]
            for i in range(pi.size(1)):
                node = pi[b, i].item()
                node_coord = coords[b, i]
                dist = torch.norm(node_coord - prev_coord).item()
                travel_time = dist / avg_speed_kmh[b].item()
                if node == 0:
                    # Returning to depot pays travel time, then the trip
                    # clock resets for the next outbound leg.
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
                # Final return-to-depot leg is charged for KPI purposes but
                # does not need a feasibility check (no further stop follows).
                time_spent[b] += torch.norm(depot[b] - prev_coord).item() / avg_speed_kmh[b].item()

        c_dict["time"] = time_spent
        return cost, c_dict, aux
