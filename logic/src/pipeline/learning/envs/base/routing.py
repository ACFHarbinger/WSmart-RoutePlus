"""Base environment and generator of the routing problems.

Every routing environment (PTP, CPTP, CPTPTC, CVRP and the unreleased variants)
builds elementary routes from the depot under declarative resource constraints
(:mod:`logic.src.pipeline.learning.envs.base.resources`): this module holds that shared state,
step, masking and reward logic.

Attributes:
    RoutingGenerator: Re-exported from :mod:`logic.src.pipeline.learning.envs.base.generator`.
    RoutingEnv: Shared routing environment.

Example:
    >>> class CVRPEnv(RoutingEnv): ...
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.pipeline.learning.envs.base.base import RL4COEnvBase
from logic.src.pipeline.learning.envs.base.generator import Generator, RoutingGenerator
from logic.src.pipeline.learning.envs.base.ops import OpsMixin, as_batch_nodes
from logic.src.pipeline.learning.envs.base.resources import Resource
from logic.src.pipeline.learning.envs.common.objectives import distance_objective, profit_objective
from logic.src.utils.data.temporal import select_instance_day


class RoutingEnv(RL4COEnvBase):
    """Base environment of the routing problems: elementary routes under resource constraints.

    Attributes:
        NAME: Canonical environment name.
        name: Alias for NAME.
        node_dim: Base dimensionality of node features.
        explicit_stop: If True, selecting depot while already at depot ends the episode.
        require_all_customers: If True, episode requires visiting all customers (CVRP style).
        mask_empty_bins: If True, masks bins with zero demand/waste.
    """

    NAME: str = "routing"
    name: str = "routing"
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
            generator = RoutingGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)
        self.resources: Tuple[Resource, ...] = tuple(resources)
        self.waste_weight = revenue_kg if revenue_kg is not None else waste_weight
        self.cost_weight = cost_km if cost_km is not None else cost_weight
        self.dm_filepath: Optional[str] = (generator_params or {}).get("dm_filepath")
        self._road_ids: Optional[dict[Any, int]] = None
        self._road_matrix: Optional[torch.Tensor] = None
        self._road_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._road_lookup: Optional[torch.Tensor] = None

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
            if ids and max(ids) < 200_000 and min(ids) >= 0:
                lookup = torch.full((max(ids) + 1,), -1, dtype=torch.long)
                for pos, bin_id in enumerate(ids):
                    lookup[bin_id] = pos
                self._road_lookup = lookup
            else:
                self._road_lookup = None

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
        elif (
            getattr(self, "_road_lookup", None) is not None
            and 0 in self._road_ids
            and node_ids.max().item() < len(self._road_lookup)
            and node_ids.min().item() >= 0
        ):
            depot_pos = self._road_ids[0]
            lookup = self._road_lookup.to(node_ids.device)
            cust_pos = lookup[node_ids.long()]
            if (cust_pos < 0).any():
                dm = torch.stack([submatrix(row) for row in node_ids]).to(device)
            else:
                depot_col = torch.full((bs, 1), depot_pos, dtype=torch.long, device=node_ids.device)
                all_pos = torch.cat([depot_col, cust_pos], dim=-1)
                mat = self._road_matrix.to(device)
                dm = mat[all_pos.unsqueeze(-1), all_pos.unsqueeze(-2)]
        else:
            dm = torch.stack([submatrix(row) for row in node_ids]).to(device)
        tensordict["dm"] = dm

    def _reset_instance(self, tensordict: TensorDict) -> TensorDict:
        """Initialize the routing state with depot prepending, tracking fields, and resources."""
        if "visited" in tensordict.keys():
            return tensordict

        device = tensordict.device
        bs = tensordict.batch_size

        locs = tensordict["locs"]
        waste = tensordict.get("waste", tensordict.get("demand", None))

        # Handle multi-day 3D waste tensors
        if waste is not None and waste.dim() > 2:
            current_day_t = tensordict.get("current_day", None)
            waste = select_instance_day(waste, current_day_t)
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


__all__ = ["RoutingGenerator", "RoutingEnv"]
