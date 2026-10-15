"""Elementary Shortest Path Problem with Resource Constraints (ESPPRC) Base Environment.

This module provides the canonical base environment and declarative resource constraints
underpinning all routing problems in WSmart-Route (PTP, CPTP, TCPTP, CVRP, and unreleased
variants).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.envs.base.ops import OpsMixin, as_batch_nodes
from logic.src.envs.generators.base import Generator
from logic.src.envs.tasks.common.objectives import distance_objective, profit_objective


class Resource(ABC):
    """Declarative resource constraint for ESPPRC problems.

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


class ESPPRCGenerator(Generator):
    """Shared generator for ESPPRC problems.

    Produces customer locations, depot coordinates, and optional road distances.
    """

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        device: Union[str, torch.device] = "cpu",
        dm_filepath: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            device=device,
            **kwargs,
        )
        self.dm_filepath = dm_filepath

    def _generate(self, batch_size: Sequence[int]) -> TensorDict:
        bs = list(batch_size) if isinstance(batch_size, (list, tuple)) else [batch_size]

        # Generate customer locations
        locs = self._generate_locations(tuple(bs), self.num_loc)
        # Generate depot coordinate
        depot = self._generate_locations(tuple(bs), 1).squeeze(-2)

        td = TensorDict(
            {
                "locs": locs,
                "depot": depot,
            },
            batch_size=bs,
            device=self.device,
        )
        if self.dm_filepath:
            td["dm_filepath"] = self.dm_filepath
        return td


class ESPPRCEnv(RL4COEnvBase):
    """Elementary Shortest Path Problem with Resource Constraints (ESPPRC) Base Environment.

    Attributes:
        NAME: Canonical environment name.
        name: Alias for NAME.
        node_dim: Base dimensionality of node features.
        explicit_stop: If True, selecting depot while already at depot ends the episode.
        require_all_customers: If True, episode requires visiting all customers (CVRP style).
        mask_empty_bins: If True, masks bins with zero demand/waste.
    """

    NAME: str = "espprc"
    name: str = "espprc"
    node_dim: int = 3
    explicit_stop: bool = False
    require_all_customers: bool = False
    mask_empty_bins: bool = False

    def __init__(
        self,
        generator: Optional[Generator] = None,
        generator_params: Optional[dict] = None,
        resources: Sequence[Resource] = (),
        waste_weight: float = 1.0,
        cost_weight: float = 1.0,
        revenue_kg: Optional[float] = None,
        cost_km: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = ESPPRCGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)
        self.resources: Tuple[Resource, ...] = tuple(resources)
        self.waste_weight = revenue_kg if revenue_kg is not None else waste_weight
        self.cost_weight = cost_km if cost_km is not None else cost_weight
        self.dm_filepath: Optional[str] = (generator_params or {}).get("dm_filepath")
        self._road_ids: Optional[dict[Any, int]] = None
        self._road_matrix: Optional[torch.Tensor] = None
        self._road_cache: dict[tuple[int, ...], torch.Tensor] = {}

    def _attach_road_distances(self, tensordict: TensorDict) -> None:
        """Add road distance matrix ``dm`` (km) for instances built on real network topology."""
        if self.dm_filepath is None or "dm" in tensordict.keys() or "node_ids" not in tensordict.keys():
            return
        node_ids = tensordict["node_ids"]
        if node_ids.dim() != 2 or node_ids.shape[-1] != tensordict["locs"].shape[-2] - 1:
            return

        if self._road_matrix is None:
            from logic.src.data.network.file import load_labelled_distance_matrix

            ids, body = load_labelled_distance_matrix(self.dm_filepath)
            self._road_ids = {bin_id: pos for pos, bin_id in enumerate(ids)}
            self._road_matrix = torch.as_tensor(body, dtype=torch.float32)

        assert self._road_ids is not None

        def submatrix(row: torch.Tensor) -> torch.Tensor:
            key = tuple(int(i) for i in row.tolist())
            if key not in self._road_cache:
                assert self._road_ids is not None and self._road_matrix is not None
                missing = [i for i in key if i not in self._road_ids]
                if 0 not in self._road_ids or missing:
                    raise ValueError(f"Distance matrix {self.dm_filepath} lacks the depot or bins {missing[:5]}")
                pos = torch.tensor([self._road_ids[0]] + [self._road_ids[i] for i in key])
                self._road_cache[key] = self._road_matrix[pos][:, pos]
            return self._road_cache[key]

        bs = tensordict.batch_size[0]
        device = tensordict.device
        if bool((node_ids == node_ids[:1]).all()):
            dm = submatrix(node_ids[0]).to(device).unsqueeze(0).expand(bs, -1, -1)
        else:
            dm = torch.stack([submatrix(row) for row in node_ids]).to(device)
        tensordict["dm"] = dm

    def _reset_instance(self, tensordict: TensorDict) -> TensorDict:
        """Initialize ESPPRC state with depot prepending, tracking fields, and resources."""
        if "visited" in tensordict.keys():
            return tensordict

        device = tensordict.device
        bs = tensordict.batch_size

        locs = tensordict["locs"]
        waste = tensordict.get("waste", tensordict.get("demand", None))

        # Handle multi-day 3D waste tensors
        if waste is not None and waste.dim() > 2:
            current_day_t = tensordict.get("current_day", None)
            day_idx = int(current_day_t.reshape(-1)[0].item()) if current_day_t is not None else 0
            day_idx = min(day_idx, waste.shape[1] - 1)
            waste = waste[:, day_idx, :]
            tensordict["waste"] = waste

        gen_n = getattr(self.generator, "num_loc", None)
        needs_prepend = False
        if "depot" in tensordict.keys():
            if gen_n is not None and locs.shape[-2] == gen_n + 1:
                needs_prepend = False
            elif (
                gen_n is not None
                and locs.shape[-2] == gen_n
                or not torch.allclose(locs[..., 0, :], tensordict["depot"], atol=1e-4)
            ):
                needs_prepend = True

        if needs_prepend:
            tensordict["locs"] = torch.cat([tensordict["depot"].unsqueeze(1), locs], dim=1)
            locs = tensordict["locs"]
            if waste is not None:
                tensordict["waste"] = torch.cat([torch.zeros(*bs, 1, device=device), waste], dim=1)
        elif waste is not None and waste.shape[-1] == locs.shape[-2] - 1:
            tensordict["waste"] = torch.cat([torch.zeros(*bs, 1, device=device), waste], dim=1)

        num_nodes = tensordict["locs"].shape[-2]

        tensordict["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        tensordict["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        # For elementarity, customer nodes (1..N) are marked visited once served.
        # Index 0 (depot) is marked visited initially for masking, but depot is explicitly allowed via action mask.
        tensordict["visited"].index_fill_(-1, torch.tensor([0], device=device), True)

        tensordict["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        tensordict["tour_length"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        tensordict["collected_waste"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        tensordict["finished"] = torch.zeros(*bs, dtype=torch.bool, device=device)

        self._attach_road_distances(tensordict)

        # Initialize all registered resources
        for res in self.resources:
            res.reset(tensordict)

        return tensordict

    def _step(self, tensordict: TensorDict) -> TensorDict:
        return OpsMixin._step(self, tensordict)

    def _step_instance(self, tensordict: TensorDict) -> TensorDict:
        """Forward-only routing step updating elementarity, distance, and resources."""
        action = as_batch_nodes(tensordict["action"])
        current = as_batch_nodes(tensordict.get("current_node", torch.zeros_like(action)))

        is_depot = action == 0
        was_at_depot = current == 0

        finished = tensordict.get("finished", torch.zeros_like(is_depot, dtype=torch.bool))
        if self.explicit_stop:
            finished = finished | (is_depot & was_at_depot)

        locs = tensordict["locs"]
        dm = tensordict.get("dm", None)
        if dm is not None:
            distance = dm[torch.arange(locs.size(0), device=locs.device), current, action]
        else:
            current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
            next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
            distance = torch.norm(next_loc - current_loc, dim=-1)

        tensordict["step_distance"] = distance
        tensordict["prev_node"] = current.clone()
        tensordict["tour_length"] = tensordict["tour_length"] + distance
        tensordict["visited"] = tensordict["visited"].scatter(1, action.unsqueeze(-1), True)
        tensordict["current_node"] = action.unsqueeze(-1)
        tensordict["tour"] = torch.cat([tensordict["tour"], action.unsqueeze(-1)], dim=-1)
        tensordict["finished"] = finished

        waste = tensordict.get("waste", tensordict.get("demand", None))
        if waste is not None:
            if waste.dim() > 1:
                idx = action.unsqueeze(-1) if action.dim() == 1 else action
                waste_at_node = waste.gather(1, idx).squeeze(-1)
            else:
                waste_at_node = waste
            tensordict["collected_waste"] = tensordict["collected_waste"] + waste_at_node * (~is_depot).float()

        # Step all registered resources
        for res in self.resources:
            res.step(tensordict, action, is_depot)

        return tensordict

    def _get_action_mask(self, tensordict: TensorDict) -> torch.Tensor:
        """Compute action feasibility mask respecting elementarity and resource limits."""
        # Unvisited customer mask
        mask = ~tensordict["visited"].clone()

        # Apply resource constraints
        for res in self.resources:
            exceeds = res.exceeds_limit(tensordict)
            mask = mask & ~exceeds

        waste = tensordict.get("waste", tensordict.get("demand", None))
        if self.mask_empty_bins and waste is not None:
            keep = waste > 0
            pending = tensordict.get("mandatory", None)
            if pending is not None:
                if pending.size(-1) == keep.size(-1) - 1:
                    pending = torch.cat([torch.zeros_like(pending[:, :1], dtype=torch.bool), pending], dim=1)
                keep = keep | pending.bool()
            keep[:, 0] = True
            mask = mask & keep

        # Mandatory logic for waste collection
        mandatory = tensordict.get("mandatory", None)
        if mandatory is not None:
            if mandatory.size(-1) == mask.size(-1) - 1:
                mandatory = torch.cat([torch.zeros_like(mandatory[:, :1], dtype=torch.bool), mandatory], dim=1)
            pending_mandatory = mandatory & mask
            has_pending = pending_mandatory[:, 1:].any(dim=1)
            mask[:, 0] = ~has_pending
        elif self.require_all_customers:
            # CVRP: depot is valid unless already at depot with unserved feasible customers
            at_depot = tensordict["current_node"].squeeze(-1) == 0
            has_valid_cust = mask[:, 1:].any(dim=-1)
            mask[:, 0] = ~(at_depot & has_valid_cust)
        else:
            # Profitable routing: depot return is always valid
            mask[:, 0] = True

        # Finished routes stay at depot
        finished = tensordict.get("finished", None)
        if self.explicit_stop and finished is not None:
            mask[finished, 1:] = False
            mask[finished, 0] = True

        return mask

    def _check_done(self, tensordict: TensorDict) -> torch.Tensor:
        """Check episode termination."""
        current = tensordict["current_node"].reshape(-1)
        step = tensordict["i"].reshape(-1)
        if self.require_all_customers:
            all_cust_done = tensordict["visited"][..., 1:].all(dim=-1)
            at_depot = current == 0
            return all_cust_done & at_depot & (step > 0)

        if self.explicit_stop:
            mask = tensordict.get("action_mask", None)
            if mask is None:
                mask = self._get_action_mask(tensordict)
            exhausted = (current == 0) & (step > 0) & ~mask[:, 1:].any(dim=1)
            return tensordict["finished"] | exhausted

        return (current == 0) & (step > 0)

    def _get_reward(self, tensordict: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute episode reward with closing return-to-depot leg."""
        cost = tensordict["tour_length"]
        current = tensordict["current_node"].squeeze(-1)
        locs = tensordict["locs"]
        dm = tensordict.get("dm", None)

        if dm is not None:
            return_dist = dm[torch.arange(locs.size(0), device=locs.device), current, 0]
        else:
            current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
            depot_loc = locs[:, 0, :]
            return_dist = torch.norm(depot_loc - current_loc, dim=-1)

        not_at_depot = (current != 0).float()
        total_cost = cost + return_dist * not_at_depot

        if self.require_all_customers:
            return distance_objective(total_cost)

        waste = tensordict.get("collected_waste", torch.zeros_like(total_cost))
        tensordict["reward_waste"] = waste
        tensordict["reward_cost"] = -total_cost

        return profit_objective(waste, total_cost, revenue_kg=self.waste_weight, cost_km=self.cost_weight)


__all__ = [
    "Resource",
    "LoadResource",
    "ShiftTimeResource",
    "ESPPRCGenerator",
    "ESPPRCEnv",
]
