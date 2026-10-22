"""Declarative resource constraints of the routing environments.

A resource tracks one budget along a route (vehicle load, shift time) and masks
the nodes that would exceed it. Each works on batched TensorDict states (neural
rollouts) and on plain dict states (the classical simulator).

Attributes:
    Resource: Abstract resource constraint.
    LoadResource: Vehicle capacity, reset at each depot visit.
    ShiftTimeResource: Per-trip time budget (travel + service time).

Example:
    >>> env = CPTPEnv(resources=(LoadResource(),))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.pipeline.learning.envs.base.ops import as_batch_nodes


class Resource(ABC):
    """Declarative resource constraint of a routing environment.

    Supports both NumPy implementations (for the classical simulator) and
    batched PyTorch tensor adapters (for RL neural rollouts) without per-instance
    loops or scalar branching.

    Attributes:
        name: Resource identifier string (e.g. "load", "shift_time").
        reset_at_depot: Whether returning to the depot resets this resource.
            True for trip resources (e.g. vehicle capacity); False for day resources
            (e.g. total daily shift budget or battery limit).
    """

    name: str
    reset_at_depot: bool = True

    @abstractmethod
    def reset(self, state: Any) -> Any:
        """Initialize resource tracking fields in the episode state."""
        pass

    @abstractmethod
    def step(self, state: Any, action: Any, is_depot: Any) -> Any:
        """Update resource state following a routing step."""
        pass

    @abstractmethod
    def exceeds_limit(self, state: Any) -> Any:
        """Compute a boolean mask of nodes that would exceed the resource limit.

        Returns:
            Tensor or ndarray of shape [B, N]: True where selecting the node
            violates the resource constraint.
        """
        pass


class LoadResource(Resource):
    """Vehicle capacity / load resource constraint.

    Tracks remaining and used capacity per trip, resetting at each depot visit.
    Supports both PyTorch TensorDict states (batched) and NumPy dict states.
    """

    name: str = "load"
    reset_at_depot: bool = True

    def __init__(self, limit_key: str = "capacity", demand_key: str = "waste") -> None:
        self.limit_key = limit_key
        self.demand_key = demand_key

    def reset(self, state: Any) -> Any:
        if isinstance(state, (TensorDict, TensorDictBase)):
            bs = state.batch_size[0] if len(state.batch_size) > 0 else 1
            device = state.device
            cap = state.get(
                self.limit_key, state.get("capacity", state.get("vehicle_capacity", state.get("max_waste", 100.0)))
            )
            cap_t = torch.as_tensor(cap, dtype=torch.float32, device=device).reshape(-1)
            if cap_t.size(0) != bs:
                cap_t = cap_t.expand(bs)
            state["capacity"] = cap_t
            state["remaining_capacity"] = cap_t.clone()
            state["used_capacity"] = torch.zeros(bs, dtype=torch.float32, device=device)
            state["collected_waste"] = torch.zeros(bs, dtype=torch.float32, device=device)
            state["collected"] = state["collected_waste"]
        elif isinstance(state, dict):
            cap_val = float(state.get(self.limit_key, state.get("capacity", 100.0)))
            state["capacity"] = cap_val
            state["remaining_capacity"] = cap_val
            state["used_capacity"] = 0.0
            state["collected_waste"] = 0.0
            state["collected"] = 0.0
        return state

    def step(self, state: Any, action: Any, is_depot: Any) -> Any:
        if isinstance(state, (TensorDict, TensorDictBase)):
            demand = state.get(self.demand_key, state.get("waste", state.get("demand", None)))
            if demand is not None:
                if demand.dim() > 1:
                    idx = action.unsqueeze(-1) if action.dim() == 1 else action
                    demand_at_node = demand.gather(1, idx).squeeze(-1)
                else:
                    demand_at_node = demand
            else:
                demand_at_node = torch.zeros_like(state["remaining_capacity"])

            at_depot = is_depot.reshape(-1) if torch.is_tensor(is_depot) else is_depot
            new_remaining = (state["remaining_capacity"] - demand_at_node).clamp(min=0.0)
            state["remaining_capacity"] = torch.where(at_depot, state["capacity"], new_remaining)
            used = (
                state["used_capacity"]
                if "used_capacity" in state.keys()
                else (state["capacity"] - state["remaining_capacity"])
            )
            new_used = used + demand_at_node
            state["used_capacity"] = torch.where(at_depot, torch.zeros_like(new_used), new_used)
            state["collected"] = state["capacity"] - state["remaining_capacity"]
        elif isinstance(state, dict):
            at_depot = bool(action == 0)
            demand_arr = state.get(self.demand_key, state.get("waste", state.get("demand", [])))
            val = float(demand_arr[action]) if len(demand_arr) > action else 0.0
            if at_depot:
                state["remaining_capacity"] = state["capacity"]
                state["used_capacity"] = 0.0
            else:
                state["remaining_capacity"] = max(0.0, state["remaining_capacity"] - val)
                state["used_capacity"] += val
                state["collected_waste"] = state.get("collected_waste", 0.0) + val
                state["collected"] = state["used_capacity"]
        return state

    def exceeds_limit(self, state: Any) -> Any:
        if isinstance(state, (TensorDict, TensorDictBase)):
            demand = state.get(self.demand_key, state.get("waste", state.get("demand", None)))
            if demand is None:
                return torch.zeros(state.batch_size[0], 1, dtype=torch.bool, device=state.device)
            remaining = state["remaining_capacity"]
            rem_2d = remaining.unsqueeze(-1) if remaining.dim() == 1 else remaining
            return demand > rem_2d + 1e-5
        elif isinstance(state, dict):
            demand_arr = np.asarray(state.get(self.demand_key, state.get("waste", state.get("demand", []))))
            return demand_arr > state["remaining_capacity"] + 1e-5
        return False


class ShiftTimeResource(Resource):
    """Per-trip time budget resource constraint (travel + service time).

    Tracks elapsed and remaining shift time per trip, resetting at depot returns.
    """

    name: str = "shift_time"
    reset_at_depot: bool = True

    def __init__(
        self,
        default_shift_hours: float = 8.0,
        default_avg_speed_kmh: float = 30.0,
        default_service_time_h: float = 0.0833,
    ) -> None:
        self.default_shift_hours = default_shift_hours
        self.default_avg_speed_kmh = default_avg_speed_kmh
        self.default_service_time_h = default_service_time_h

    def reset(self, state: Any) -> Any:
        if isinstance(state, (TensorDict, TensorDictBase)):
            bs = state.batch_size[0] if len(state.batch_size) > 0 else 1
            device = state.device
            shift = state.get("shift_hours", torch.full((bs,), self.default_shift_hours, device=device))
            speed = state.get("avg_speed_kmh", torch.full((bs,), self.default_avg_speed_kmh, device=device))
            service = state.get("service_time_h", torch.full((bs,), self.default_service_time_h, device=device))
            state["shift_hours"] = shift
            state["avg_speed_kmh"] = speed
            state["service_time_h"] = service
            state["remaining_time"] = shift.clone()
            state["time_spent"] = torch.zeros(bs, device=device)
        elif isinstance(state, dict):
            shift = float(state.get("shift_hours", self.default_shift_hours))
            state["shift_hours"] = shift
            state["avg_speed_kmh"] = float(state.get("avg_speed_kmh", self.default_avg_speed_kmh))
            state["service_time_h"] = float(state.get("service_time_h", self.default_service_time_h))
            state["remaining_time"] = shift
            state["time_spent"] = 0.0
        return state

    def step(self, state: Any, action: Any, is_depot: Any) -> Any:
        if isinstance(state, (TensorDict, TensorDictBase)):
            bs = state.batch_size[0] if len(state.batch_size) > 0 else 1
            device = state.device
            at_depot = is_depot.reshape(-1) if torch.is_tensor(is_depot) else is_depot

            step_distance = state.get("step_distance", torch.zeros(bs, device=device))
            speed = state.get("avg_speed_kmh", torch.full((bs,), self.default_avg_speed_kmh, device=device))
            step_travel_time = step_distance / speed

            time_matrix = state.get("time_matrix", None)
            if time_matrix is not None and "prev_node" in state.keys():
                prev = state["prev_node"].reshape(-1)
                act = action.reshape(-1)
                step_travel_time = time_matrix[torch.arange(bs, device=device), prev, act]

            service = state.get("service_time_h", torch.full((bs,), self.default_service_time_h, device=device))
            step_service_time = torch.where(at_depot, torch.zeros_like(step_travel_time), service)
            step_time = step_travel_time + step_service_time

            state["time_spent"] = state.get("time_spent", torch.zeros_like(step_time)) + step_time
            rem_after = state["remaining_time"] - step_time
            state["remaining_time"] = torch.where(at_depot, state["shift_hours"], rem_after)
        elif isinstance(state, dict):
            at_depot = bool(action == 0)
            dist = float(state.get("step_distance", 0.0))
            speed = float(state.get("avg_speed_kmh", self.default_avg_speed_kmh))
            travel_t = dist / speed
            service_t = 0.0 if at_depot else float(state.get("service_time_h", self.default_service_time_h))
            total_t = travel_t + service_t
            state["time_spent"] = state.get("time_spent", 0.0) + total_t
            if at_depot:
                state["remaining_time"] = state["shift_hours"]
            else:
                state["remaining_time"] = max(0.0, state["remaining_time"] - total_t)
        return state

    def exceeds_limit(self, state: Any) -> Any:
        if isinstance(state, (TensorDict, TensorDictBase)):
            current = as_batch_nodes(
                state.get("current_node", torch.zeros(state.batch_size[0], 1, dtype=torch.long, device=state.device))
            )
            locs = state["locs"]
            time_matrix = state.get("time_matrix", None)
            if time_matrix is not None:
                time_curr_to_node = time_matrix.gather(
                    1, current[:, None, None].expand(-1, -1, time_matrix.size(-1))
                ).squeeze(1)
                time_node_to_depot = time_matrix[:, :, 0]
            else:
                dm = state.get("dm", None)
                if dm is None:
                    current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
                    dist_curr_to_node = (locs - current_loc.unsqueeze(1)).norm(p=2, dim=-1)
                    dist_node_to_depot = (locs - locs[:, 0:1, :]).norm(p=2, dim=-1)
                else:
                    dist_curr_to_node = dm.gather(1, current[:, None, None].expand(-1, -1, dm.size(-1))).squeeze(1)
                    dist_node_to_depot = dm[:, :, 0]

                avg_speed = state.get(
                    "avg_speed_kmh", torch.full((locs.size(0),), self.default_avg_speed_kmh, device=locs.device)
                ).unsqueeze(-1)
                time_curr_to_node = dist_curr_to_node / avg_speed
                time_node_to_depot = dist_node_to_depot / avg_speed

            service_time = state.get(
                "service_time_h", torch.full((locs.size(0),), self.default_service_time_h, device=locs.device)
            ).unsqueeze(-1)
            service_at_nodes = service_time.expand(-1, locs.size(1)).clone()
            service_at_nodes[:, 0] = 0.0

            total_trip_time_needed = time_curr_to_node + service_at_nodes + time_node_to_depot
            remaining_time = state["remaining_time"].unsqueeze(-1)
            return total_trip_time_needed > remaining_time + 1e-6
        return False


class TimeWindowResource(Resource):
    """Day clock with per-node time windows (CPTPTWTC).

    ``current_time`` is the time of day (h). It runs across trips: a depot return resets the
    shift budget (``ShiftTimeResource``), not the clock. Service at a node starts at
    ``max(arrival, tw_early)``; moving to a node is infeasible when service would start after
    its ``tw_late``, or when the vehicle could then no longer reach the depot before the day
    ends (``tw_late`` of the depot, ``day_hours``). The depot itself is never masked here, so a
    vehicle that is out of time can always go home.
    """

    name: str = "time_window"
    reset_at_depot: bool = False

    def __init__(
        self,
        default_day_hours: float = 10.0,
        default_avg_speed_kmh: float = 30.0,
        default_service_time_h: float = 0.0833,
    ) -> None:
        self.default_day_hours = default_day_hours
        self.default_avg_speed_kmh = default_avg_speed_kmh
        self.default_service_time_h = default_service_time_h

    def reset(self, state: Any) -> Any:
        if not isinstance(state, (TensorDict, TensorDictBase)):
            return state
        bs = state.batch_size[0] if len(state.batch_size) > 0 else 1
        device = state.device
        n_nodes = state["locs"].size(-2)
        day = state.get("day_hours", torch.full((bs,), self.default_day_hours, device=device))
        early = state.get("tw_early", torch.zeros(bs, n_nodes - 1, device=device))
        late = state.get("tw_late", day.unsqueeze(-1).expand(bs, n_nodes - 1).clone())
        if early.size(-1) == n_nodes - 1:  # windows of the bins only: the depot is open all day
            early = torch.cat([torch.zeros(bs, 1, device=device), early], dim=-1)
            late = torch.cat([day.unsqueeze(-1), late], dim=-1)
        state["day_hours"] = day
        state["tw_early"] = early
        state["tw_late"] = late
        state["current_time"] = torch.zeros(bs, device=device)
        return state

    def _travel_time_from_current(self, state: TensorDictBase) -> Tuple[torch.Tensor, torch.Tensor]:
        """Travel times (h) from the current node to every node, and from every node to the depot."""
        current = as_batch_nodes(
            state.get("current_node", torch.zeros(state.batch_size[0], 1, dtype=torch.long, device=state.device))
        )
        locs = state["locs"]
        time_matrix = state.get("time_matrix", None)
        if time_matrix is not None:
            to_node = time_matrix.gather(1, current[:, None, None].expand(-1, -1, time_matrix.size(-1))).squeeze(1)
            return to_node, time_matrix[:, :, 0]
        dm = state.get("dm", None)
        if dm is None:
            current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
            dist_to_node = (locs - current_loc.unsqueeze(1)).norm(p=2, dim=-1)
            dist_to_depot = (locs - locs[:, 0:1, :]).norm(p=2, dim=-1)
        else:
            dist_to_node = dm.gather(1, current[:, None, None].expand(-1, -1, dm.size(-1))).squeeze(1)
            dist_to_depot = dm[:, :, 0]
        speed = state.get(
            "avg_speed_kmh", torch.full((locs.size(0),), self.default_avg_speed_kmh, device=locs.device)
        ).unsqueeze(-1)
        return dist_to_node / speed, dist_to_depot / speed

    def step(self, state: Any, action: Any, is_depot: Any) -> Any:
        if not isinstance(state, (TensorDict, TensorDictBase)):
            return state
        bs = state.batch_size[0] if len(state.batch_size) > 0 else 1
        device = state.device
        act = action.reshape(-1)
        at_depot = is_depot.reshape(-1) if torch.is_tensor(is_depot) else torch.full((bs,), bool(is_depot), device=device)
        speed = state.get("avg_speed_kmh", torch.full((bs,), self.default_avg_speed_kmh, device=device))
        travel = state.get("step_distance", torch.zeros(bs, device=device)) / speed
        time_matrix = state.get("time_matrix", None)
        if time_matrix is not None and "prev_node" in state.keys():
            travel = time_matrix[torch.arange(bs, device=device), state["prev_node"].reshape(-1), act]
        rows = torch.arange(bs, device=device)
        start = torch.maximum(state["current_time"] + travel, state["tw_early"][rows, act])
        service = state.get("service_time_h", torch.full((bs,), self.default_service_time_h, device=device))
        state["current_time"] = start + torch.where(at_depot, torch.zeros_like(service), service)
        return state

    def exceeds_limit(self, state: Any) -> Any:
        if not isinstance(state, (TensorDict, TensorDictBase)):
            return False
        to_node, to_depot = self._travel_time_from_current(state)
        start = torch.maximum(state["current_time"].unsqueeze(-1) + to_node, state["tw_early"])
        service = state.get(
            "service_time_h", torch.full((to_node.size(0),), self.default_service_time_h, device=to_node.device)
        ).unsqueeze(-1)
        late = start > state["tw_late"] + 1e-6
        no_way_home = start + service + to_depot > state["tw_late"][:, :1] + 1e-6
        blocked = late | no_way_home
        blocked[:, 0] = False
        return blocked


__all__ = ["Resource", "LoadResource", "ShiftTimeResource", "TimeWindowResource"]
