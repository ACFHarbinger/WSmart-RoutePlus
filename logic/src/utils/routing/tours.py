"""
Tour helpers shared by the simulator and the route constructors.

Attributes:
    None

Example:
    >>> from logic.src.utils.routing.tours import get_route_cost
    >>> get_route_cost(dist, [0, 1, 2, 0])
"""

from typing import List, Optional, cast

import numpy as np
import torch


def get_route_cost(distancesC, tour):
    """
    Calculate total distance cost of a tour.

    Sums the edge distances along the tour path.
    Supports both NumPy arrays and PyTorch tensors.

    Args:
        distancesC (np.ndarray or torch.Tensor): Distance matrix
        tour (list or np.ndarray or torch.Tensor): Sequence of node IDs

    Returns:
        float: Total tour distance
    """
    if isinstance(tour, torch.Tensor) and isinstance(distancesC, torch.Tensor):
        return distancesC[tour[:-1], tour[1:]].sum().cpu().numpy().item()
    else:
        distancesC2 = distancesC.copy() if isinstance(distancesC, np.ndarray) else np.array(distancesC)
        tour2 = tour.copy() if isinstance(tour, np.ndarray) else np.array(tour)
        return np.sum(distancesC2[tour2[:-1], tour2[1:]]).item()


def _trip_time_matrix(
    distance_matrix: np.ndarray,
    time_matrix: Optional[np.ndarray],
    avg_speed_kmh: Optional[float],
    shift_hours: float,
    service_time_h: float,
) -> np.ndarray:
    """Validate temporal resources and return directed travel hours."""
    if not np.isfinite(shift_hours) or shift_hours <= 0:
        raise ValueError("shift_hours must be finite and positive")
    if not np.isfinite(service_time_h) or service_time_h < 0:
        raise ValueError("service_time_h must be finite and nonnegative")
    if time_matrix is None:
        avg_speed_kmh = 35.0 if avg_speed_kmh is None else avg_speed_kmh
        if not np.isfinite(avg_speed_kmh) or avg_speed_kmh <= 0:
            raise ValueError("avg_speed_kmh must be finite and positive")
        time_matrix = np.asarray(distance_matrix) / avg_speed_kmh
    if time_matrix.shape != distance_matrix.shape:
        raise ValueError("Travel-time and distance matrix shapes must match")
    if not np.isfinite(time_matrix).all() or (time_matrix < 0).any():
        raise ValueError("Travel times must be finite and nonnegative")
    return time_matrix


def get_multi_tour(
    tour: List[int],
    bins_waste: np.ndarray,
    max_capacity: float,
    distance_matrix: np.ndarray,
    shift_hours: Optional[float] = None,
    avg_speed_kmh: Optional[float] = None,
    service_time_h: Optional[float] = None,
    time_matrix: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Insert depot return trips to satisfy vehicle capacity and time budget constraints.

    Given a TSP tour that may violate capacity or duration limits, inserts depot visits (0)
    whenever cumulative load would exceed max_capacity or cumulative trip duration would
    exceed shift_hours. This converts a single long tour into multiple feasible depot round-trips.

    Args:
        tour: Initial TSP tour (format [0, ..., 0]).
        bins_waste: Waste amounts for each customer bin (0-indexed).
        max_capacity: Vehicle capacity limit.
        distance_matrix: Distance matrix between all nodes including depot (0).
        shift_hours: Optional per-trip time budget in hours.
        avg_speed_kmh: Optional average vehicle speed in km/h.
        service_time_h: Optional service time per bin in hours.
        time_matrix: Optional directed travel times in hours, including depot.

    Returns:
        List[int]: Modified tour with depot returns inserted. Format: [0, ..., 0, ..., 0]
    """
    customer_nodes = [x for x in tour if x != 0]
    if not customer_nodes:
        return [0, 0]

    service_time_h = 0.025 if service_time_h is None else service_time_h
    if shift_hours is not None:
        time_matrix = _trip_time_matrix(distance_matrix, time_matrix, avg_speed_kmh, shift_hours, service_time_h)
    travel_times = cast(np.ndarray, time_matrix)

    final_tour = [0]
    cur_load = 0.0
    cur_trip_time = 0.0
    cur_trip_bins = 0
    prev_node = 0

    for node in tour:
        if node == 0:
            if final_tour[-1] != 0:
                final_tour.append(0)
            cur_load = 0.0
            cur_trip_time = 0.0
            cur_trip_bins = 0
            prev_node = 0
            continue
        cur_bin = node - 1
        if cur_bin < 0 or cur_bin >= len(bins_waste):
            raise ValueError(f"Unknown customer node {node}")
        col_waste = float(bins_waste[cur_bin])
        if col_waste > max_capacity + 1e-6:
            raise ValueError(f"Customer {node} exceeds vehicle capacity on its own")
        if shift_hours is not None:
            solo_time = float(travel_times[0, node] + travel_times[node, 0]) + service_time_h
            if solo_time > shift_hours + 1e-6:
                raise ValueError(f"Customer {node} exceeds shift budget on its own")

        candidate_load = cur_load + col_waste

        exceeds_capacity = candidate_load > max_capacity + 1e-6
        exceeds_time = False
        if shift_hours is not None:
            candidate_trip_time = (
                cur_trip_time + float(travel_times[prev_node, node] + travel_times[node, 0]) + service_time_h
            )
            exceeds_time = candidate_trip_time > shift_hours + 1e-6

        # If adding this node violates either constraint and current trip already has stops:
        if (exceeds_capacity or exceeds_time) and cur_trip_bins > 0:
            final_tour.append(0)
            cur_load = 0.0
            cur_trip_time = 0.0
            cur_trip_bins = 0
            prev_node = 0

        final_tour.append(node)
        cur_load += col_waste
        if shift_hours is not None:
            cur_trip_time += float(travel_times[prev_node, node]) + service_time_h
        cur_trip_bins += 1
        prev_node = node

    if final_tour[-1] != 0:
        final_tour.append(0)
    return final_tour
