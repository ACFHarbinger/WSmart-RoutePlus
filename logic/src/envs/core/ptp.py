"""Shared utilities, economic objectives, and tour evaluation for Profitable Tour Problems.

This module defines the canonical economic formula (EUR) and vectorized tour
evaluation routines for PTP, CPTP, and CPTPTC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.constants.tasks import COST_KM, REVENUE_KG
from logic.src.constants.temporal import get_default_temporal_params
from logic.src.envs.common.objectives import profit_objective


def closed_tour_length(dm: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Sum directed road legs, closing open paths exactly once at the depot.

    ``pi`` is [B, L]; ``dm`` is [N, N] or [B, N, N] with depot at 0 and
    zero diagonal. Intermediate depot returns are ordinary directed legs.
    """
    if dm.dim() == 2:
        dm = dm.unsqueeze(0).expand(pi.shape[0], -1, -1)
    if pi.shape[1] == 0:
        return dm.new_zeros(pi.shape[0])
    batch = torch.arange(pi.shape[0], device=pi.device)
    return dm[batch[:, None], pi[:, :-1], pi[:, 1:]].sum(dim=-1) + dm[batch, 0, pi[:, 0]] + dm[batch, pi[:, -1], 0]


def calc_cumulative_trip_load(waste_seq: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Vectorized per-trip cumulative load calculation.

    Resets the accumulated load to zero whenever the sequence visits the depot (pi == 0),
    without Python loops over batch or tour lengths.

    Args:
        waste_seq: Tensor of waste gathered at each tour step, shape [B, L].
        pi: Action / node sequence, shape [B, L].

    Returns:
        torch.Tensor: Cumulative load at each step within its trip, shape [B, L].
    """
    cumsum = waste_seq.cumsum(dim=-1)
    reset_mask = pi == 0
    base = torch.zeros_like(cumsum)
    base[reset_mask] = cumsum[reset_mask]
    base = base.cummax(dim=-1).values
    return cumsum - base


def evaluate_profitable_tour(  # noqa: C901
    dataset: Dict[str, Any],
    pi: torch.Tensor,
    dist_matrix: Optional[torch.Tensor] = None,
    capacity: Optional[float] = None,
    shift_hours: Optional[float] = None,
    avg_speed_kmh: Optional[float] = None,
    service_time_h: Optional[float] = None,
    cw_dict: Optional[Dict[str, float]] = None,
    check_feasibility: bool = False,
) -> Dict[str, torch.Tensor]:
    """Vectorized evaluation of a completed tour sequence for profitable tour problems.

    Computes physical outcomes (distance km, collected waste kg, overflows, validity)
    and economic values (cost, reward).

    Args:
        dataset: Dictionary containing problem instances ('locs', 'depot', 'waste', etc.).
        pi: Tour sequence of shape [B, L] or [L].
        dist_matrix: Optional precomputed distance matrix [B, N, N] or [N, N].
        capacity: Optional vehicle capacity per trip. If None, read from dataset.
        shift_hours: Optional shift time limit for CPTPTC.
        avg_speed_kmh: Optional average vehicle speed.
        service_time_h: Optional service time per customer visit.
        cw_dict: Optional legacy cost weight dictionary.
        check_feasibility: If True, raises AssertionError on capacity or time limit violation.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of outcome tensors:
            - 'cost': negative profit in EUR (for minimization), shape [B]
            - 'reward': net profit in EUR (for maximization), shape [B]
            - 'length': total distance traveled in km, shape [B]
            - 'waste': total waste collected in normalized units or kg, shape [B]
            - 'overflows': count of capacity / constraint violations, shape [B]
            - 'valid': boolean feasibility tensor, shape [B]
            - 'total': alias for 'cost', shape [B]
            - 'time': total time spent in hours (if temporal parameters present), shape [B]
    """
    if pi.dim() == 1:
        pi = pi.unsqueeze(0)

    bs, seq_len = pi.shape
    device = pi.device

    if seq_len == 0 or (seq_len == 1 and bool((pi == 0).all().item())):
        z = torch.zeros(bs, device=device)
        return {
            "cost": z,
            "reward": z,
            "length": z,
            "waste": z,
            "overflows": z,
            "valid": torch.ones(bs, dtype=torch.bool, device=device),
            "total": z,
            "time": z,
        }

    # Extract locations, depot, and waste
    locs = dataset.get("locs") if "locs" in dataset else dataset.get("loc")
    depot = dataset.get("depot")
    raw_waste = dataset.get("waste")

    # Determine full coordinate layout (with depot at index 0)
    if locs is not None:
        if locs.dim() == 2:
            locs = locs.unsqueeze(0).expand(bs, -1, -1)
        if depot is not None and (
            (raw_waste is not None and locs.size(1) == raw_waste.size(-1))
            and not torch.allclose(locs[:, 0, :], depot if depot.dim() == 2 else depot.unsqueeze(0), atol=1e-4)
        ):
            depot_2d = depot.unsqueeze(1) if depot.dim() == 2 else depot.unsqueeze(0).unsqueeze(1)
            if depot_2d.size(0) != bs:
                depot_2d = depot_2d.expand(bs, -1, -1)
            full_locs = torch.cat([depot_2d, locs], dim=1)
        else:
            full_locs = locs
    else:
        full_locs = None

    # Calculate tour length
    dm_input = dist_matrix if dist_matrix is not None else dataset.get("dist_matrix", dataset.get("dm"))
    if dm_input is not None and isinstance(dm_input, torch.Tensor):
        length = closed_tour_length(dm_input, pi)
    elif full_locs is not None:
        coords = full_locs.gather(1, pi.unsqueeze(-1).expand(*pi.shape, full_locs.size(-1)))
        depot_coord = full_locs[:, 0:1, :]
        length = (
            (coords[:, 1:] - coords[:, :-1]).norm(p=2, dim=-1).sum(dim=1)
            + (coords[:, 0:1] - depot_coord).norm(p=2, dim=-1).squeeze(1)
            + (coords[:, -1:] - depot_coord).norm(p=2, dim=-1).squeeze(1)
        )
    else:
        length = torch.zeros(bs, device=device)

    if raw_waste is not None:
        if raw_waste.dim() == 1:
            raw_waste = raw_waste.unsqueeze(0).expand(bs, -1)
        elif raw_waste.dim() == 3:
            raw_waste = raw_waste[:, 0, :]
        num_loc = (
            full_locs.size(1)
            if full_locs is not None
            else dm_input.size(-1)
            if dm_input is not None
            else raw_waste.size(1)
        )
        if raw_waste.size(1) == num_loc - 1:
            full_waste = torch.cat([torch.zeros(bs, 1, device=device), raw_waste], dim=-1)
        else:
            full_waste = raw_waste
        w_seq = full_waste.gather(1, pi)
        if "max_waste" in dataset:
            max_w = dataset["max_waste"]
            if torch.is_tensor(max_w):
                w_seq = w_seq.clamp(max=max_w[:, None] if max_w.dim() == 1 else max_w)
        waste = w_seq.sum(dim=-1)
    else:
        w_seq = torch.zeros_like(pi, dtype=torch.float32)
        waste = torch.zeros(bs, device=device)

    # Constraint checks: Capacity per trip
    overflows = torch.zeros(bs, device=device)
    if capacity is not None or "capacity" in dataset or "max_waste" in dataset:
        cap_val = capacity
        if cap_val is None:
            cap_val = dataset.get("capacity", dataset.get("max_waste", 100.0))
        cap_tensor = torch.as_tensor(cap_val, dtype=torch.float32, device=device).reshape(-1)
        if cap_tensor.size(0) != bs:
            cap_tensor = cap_tensor.expand(bs)

        trip_loads = calc_cumulative_trip_load(w_seq, pi)
        max_trip_load = trip_loads.max(dim=-1).values
        load_excess = (max_trip_load - cap_tensor).clamp(min=0.0)
        has_load_overflow = load_excess > 1e-5
        overflows = overflows + has_load_overflow.float()
        if check_feasibility and has_load_overflow.any():
            raise AssertionError("Used more than capacity")

    # Time budget per trip (CPTPTC)
    time_spent = torch.zeros(bs, device=device)
    has_temporal = shift_hours is not None or "shift_hours" in dataset or "time_matrix" in dataset
    if has_temporal:
        def_shift, def_speed, def_service = get_default_temporal_params()
        shift_val = shift_hours if shift_hours is not None else dataset.get("shift_hours", def_shift)
        speed_val = avg_speed_kmh if avg_speed_kmh is not None else dataset.get("avg_speed_kmh", def_speed)
        serv_val = service_time_h if service_time_h is not None else dataset.get("service_time_h", def_service)

        shift_t = torch.as_tensor(shift_val, dtype=torch.float32, device=device).reshape(-1)
        if shift_t.size(0) != bs:
            shift_t = shift_t.expand(bs)
        speed_t = torch.as_tensor(speed_val, dtype=torch.float32, device=device).reshape(-1)
        if speed_t.size(0) != bs:
            speed_t = speed_t.expand(bs)
        serv_t = torch.as_tensor(serv_val, dtype=torch.float32, device=device).reshape(-1)
        if serv_t.size(0) != bs:
            serv_t = serv_t.expand(bs)

        # Build distance lookup for step-by-step leg times
        travel_matrix = dataset.get("time_matrix")
        dm_mat = dm_input
        if dm_mat is not None and dm_mat.dim() == 2:
            dm_mat = dm_mat.unsqueeze(0).expand(bs, -1, -1)

        for b in range(bs):
            cur_trip_time = 0.0
            prev_node = 0
            for i in range(seq_len):
                node = int(pi[b, i].item())
                if travel_matrix is not None:
                    leg_t = (
                        travel_matrix[b, prev_node, node].item()
                        if travel_matrix.dim() == 3
                        else travel_matrix[prev_node, node].item()
                    )
                elif dm_mat is not None:
                    leg_t = dm_mat[b, prev_node, node].item() / speed_t[b].item()
                elif full_locs is not None:
                    c_src = full_locs[b, prev_node]
                    c_dst = full_locs[b, node]
                    leg_t = float(torch.norm(c_dst - c_src).item()) / speed_t[b].item()
                else:
                    leg_t = 0.0

                if node == 0:
                    cur_trip_time += leg_t
                    if cur_trip_time > shift_t[b].item() + 1e-6:
                        overflows[b] += 1.0
                        if check_feasibility:
                            raise AssertionError(
                                f"CPTPTC: trip time {cur_trip_time:.4f}h exceeds shift budget {shift_t[b].item():.4f}h"
                            )
                    time_spent[b] += leg_t
                    cur_trip_time = 0.0
                else:
                    step_t = leg_t + serv_t[b].item()
                    cur_trip_time += step_t
                    time_spent[b] += step_t
                    if cur_trip_time > shift_t[b].item() + 1e-6:
                        overflows[b] += 1.0
                        if check_feasibility:
                            raise AssertionError(
                                f"CPTPTC: trip time {cur_trip_time:.4f}h exceeds shift budget {shift_t[b].item():.4f}h"
                            )
                prev_node = node

            # Final return to depot if not at depot
            if pi[b, -1].item() != 0:
                if travel_matrix is not None:
                    final_leg = (
                        travel_matrix[b, prev_node, 0].item()
                        if travel_matrix.dim() == 3
                        else travel_matrix[prev_node, 0].item()
                    )
                elif dm_mat is not None:
                    final_leg = dm_mat[b, prev_node, 0].item() / speed_t[b].item()
                elif full_locs is not None:
                    final_leg = float(torch.norm(full_locs[b, 0] - full_locs[b, prev_node]).item()) / speed_t[b].item()
                else:
                    final_leg = 0.0
                cur_trip_time += final_leg
                time_spent[b] += final_leg
                if cur_trip_time > shift_t[b].item() + 1e-6:
                    overflows[b] += 1.0
                    if check_feasibility:
                        raise AssertionError(
                            f"CPTPTC: trip time {cur_trip_time:.4f}h exceeds shift budget {shift_t[b].item():.4f}h"
                        )

    valid = overflows <= 0

    cost_km = dataset.get("cost_km", COST_KM)
    revenue_kg = dataset.get("revenue_kg", REVENUE_KG)

    profit = profit_objective(waste, length, revenue_kg=revenue_kg, cost_km=cost_km)
    if cw_dict is not None:
        profit = profit_objective(
            cw_dict.get("waste", 1.0) * waste,
            cw_dict.get("length", 1.0) * length,
            revenue_kg=revenue_kg,
            cost_km=cost_km,
        )
    neg_profit = -profit

    out = {
        "cost": neg_profit,
        "reward": profit,
        "length": length,
        "waste": waste,
        "overflows": overflows,
        "valid": valid,
        "total": neg_profit,
    }
    if has_temporal:
        out["time"] = time_spent
    return out


__all__ = [
    "COST_KM",
    "REVENUE_KG",
    "profit_objective",
    "calc_cumulative_trip_load",
    "evaluate_profitable_tour",
]
