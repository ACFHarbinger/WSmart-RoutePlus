"""
CTOP Environment implementation.

Capacitated Team Orienteering Problem: single-vehicle, multi-trip variant of
CVRPP where each trip is bounded by *both* constraints simultaneously --
CVRPP's per-trip vehicle capacity, unchanged, plus a per-trip time budget
(travel + service time). Neither constraint replaces the other. The vehicle
may return to the depot and start a new trip with both budgets freshly
reset; the episode ends once it is back at the depot with nothing more
reachable even at full budget.

"Team" here means multiple *trips* by one vehicle within a period, not a
concurrent multi-vehicle fleet. A true multi-vehicle fleet CTOP is tracked
as future work (docs/moon/roadmaps/new_features.md §E.8) for the
heterogeneous-waste-stream scenario (e.g. one vehicle per waste type
running simultaneously) -- that needs a fleet dimension this class does
not have.

Attributes:
    CTOPEnv: CTOP environment.

Example:
    >>> from logic.src.envs.routing import get_env
    >>> env = get_env("ctop", num_loc=50)
    >>> td = env.reset()
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.generators.ctop import CTOPGenerator
from logic.src.envs.routing.cvrpp import CVRPPEnv
from logic.src.envs.temporal import get_default_temporal_params


class CTOPEnv(CVRPPEnv):
    """
    Capacitated Team Orienteering Problem: CVRPP plus a per-trip time budget.

    Attributes:
        name: Name of the environment.
    """

    name: str = "ctop"

    def __init__(
        self,
        generator: Optional[CTOPGenerator] = None,
        generator_params: Optional[dict] = None,
        waste_weight: float = 1.0,
        cost_weight: float = 1.0,
        revenue_kg: Optional[float] = None,
        cost_km: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> None:
        """
        Initialize CTOPEnv with a CTOPGenerator (not VRPPEnv's plain
        VRPPGenerator).

        Without this override, `get_env("ctop", shift_hours=6.5, ...)`
        silently builds a VRPPGenerator via VRPPEnv.__init__: the
        shift_hours/avg_speed_kmh/service_time_h kwargs are swallowed by
        VRPPGenerator's **kwargs, never reach a CTOPGenerator, and
        _reset_instance falls back to get_default_temporal_params()
        regardless of what was requested. Confirmed live (Hydra config
        overrides for these three keys were composing correctly but never
        actually reaching the environment) before this fix.

        Args:
            generator: Pre-built CTOPGenerator instance. Built from
                generator_params if not supplied.
            generator_params: Keyword arguments forwarded to CTOPGenerator
                when generator is None.
            waste_weight: Weight for waste collection in reward.
            cost_weight: Weight for travel cost in reward.
            revenue_kg: Optional revenue per kg (overrides waste_weight).
            cost_km: Optional cost per km (overrides cost_weight).
            device: Device for torch tensors ('cpu' or 'cuda').
            kwargs: Additional keyword arguments.
        """
        generator_params = generator_params or kwargs
        if generator is None:
            generator = CTOPGenerator(**generator_params, device=device)
        # Pass the already-built CTOPGenerator through: VRPPEnv.__init__'s
        # own `if generator is None` branch is then skipped, so it never
        # constructs the wrong (plain VRPPGenerator) type.
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            waste_weight=waste_weight,
            cost_weight=cost_weight,
            revenue_kg=revenue_kg,
            cost_km=cost_km,
            device=device,
            **kwargs,
        )

    def _reset_instance(self, tensordict: TensorDict) -> TensorDict:
        """Initialize CTOP state with per-trip time-budget tracking.

        Args:
            tensordict: Input TensorDict containing graph structure and node properties.

        Returns:
            TensorDict: Initialized CTOP state with temporal tracking fields.
        """
        is_resuming = "visited" in tensordict.keys()
        tensordict = super()._reset_instance(tensordict)

        bs = tensordict.batch_size[0]
        device = tensordict.device

        default_shift_hours, default_avg_speed_kmh, default_service_time_h = get_default_temporal_params()
        shift_hours = tensordict.get("shift_hours", torch.full((bs,), default_shift_hours, device=device))
        avg_speed_kmh = tensordict.get("avg_speed_kmh", torch.full((bs,), default_avg_speed_kmh, device=device))
        service_time_h = tensordict.get("service_time_h", torch.full((bs,), default_service_time_h, device=device))

        tensordict["shift_hours"] = shift_hours
        tensordict["avg_speed_kmh"] = avg_speed_kmh
        tensordict["service_time_h"] = service_time_h
        if not is_resuming:
            tensordict["remaining_time"] = shift_hours.clone()
            tensordict["time_spent"] = torch.zeros(bs, device=device)
        else:
            tensordict.setdefault("remaining_time", shift_hours.clone())
            tensordict.setdefault("time_spent", torch.zeros(bs, device=device))

        return tensordict

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:  # type: ignore[override]
        """Reset the environment.

        Args:
            tensordict: Input TensorDict containing graph structure and node properties.
            kwargs: Additional keyword arguments.

        Returns:
            TensorDict: Reset environment state.
        """
        return super()._reset(tensordict, **kwargs)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Step the environment.

        Args:
            tensordict: Input TensorDict containing action.

        Returns:
            TensorDict: Environment state after action.
        """
        return OpsMixin._step(self, tensordict)

    def _step_instance(self, tensordict: TensorDict) -> TensorDict:
        """Execute action with per-trip time-budget tracking.

        Reads the per-step travel distance from the ``tour_length`` delta
        produced by CVRPPEnv/VRPPEnv/OpsMixin's own step (so this honours a
        road distance matrix ``dm`` when present, exactly like every other
        distance consumer in the mixin) rather than recomputing distance
        from raw coordinates. That same ``super()`` call also applies
        CVRPP's capacity tracking, so both constraints update together.

        Args:
            tensordict: Input TensorDict containing action and state.

        Returns:
            TensorDict: Updated state with time_spent/remaining_time.
        """
        action = tensordict["action"]
        if action.dim() > 1:
            action = action.squeeze(-1)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        is_depot = action == 0

        bs = tensordict.batch_size[0]
        device = tensordict.device
        prev_tour_length = tensordict.get("tour_length", torch.zeros(bs, device=device)).clone()

        tensordict = super()._step_instance(tensordict)

        step_distance = tensordict["tour_length"] - prev_tour_length
        step_travel_time = step_distance / tensordict["avg_speed_kmh"]
        step_service_time = torch.where(is_depot, torch.zeros_like(step_travel_time), tensordict["service_time_h"])
        step_time = step_travel_time + step_service_time

        tensordict["time_spent"] = tensordict.get("time_spent", torch.zeros_like(step_time)) + step_time

        remaining_after_step = tensordict["remaining_time"] - step_time
        # New trip starts fresh at the depot; mid-trip, budget just depletes.
        tensordict["remaining_time"] = torch.where(is_depot, tensordict["shift_hours"], remaining_after_step)

        return tensordict

    def _get_action_mask(self, tensordict: TensorDict) -> torch.Tensor:
        """
        Mask nodes unreachable within the remaining per-trip time budget,
        layered on top of CVRPP's own capacity-aware mask.

        A customer is valid only if CVRPP's mask allows it (not visited,
        has waste, mandatory logic, fits remaining capacity) AND the
        vehicle can reach it, service it, and still get back to the depot
        before ``remaining_time`` runs out -- the same reach-and-return
        structure as OPEnv's distance budget, translated to time. Both the
        capacity and time constraints must hold; this only narrows CVRPP's
        mask further, never widens it.

        Args:
            tensordict: Input TensorDict containing graph structure and node properties.

        Returns:
            torch.Tensor: Boolean action mask, shape (batch, num_nodes).
        """
        mask = super()._get_action_mask(tensordict)

        current = tensordict["current_node"].squeeze(-1)
        locs = tensordict["locs"]
        dm = tensordict.get("dm")
        if dm is None:
            current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
            dist_current_to_node = (locs - current_loc.unsqueeze(1)).norm(p=2, dim=-1)
            dist_node_to_depot = (locs - locs[:, 0:1, :]).norm(p=2, dim=-1)
        else:
            dist_current_to_node = dm.gather(1, current[:, None, None].expand(-1, -1, dm.size(-1))).squeeze(1)
            dist_node_to_depot = dm[:, :, 0]

        avg_speed = tensordict["avg_speed_kmh"].unsqueeze(-1)
        service_time = tensordict["service_time_h"].unsqueeze(-1).expand_as(dist_current_to_node).clone()
        service_time[:, 0] = 0.0
        remaining = tensordict["remaining_time"].unsqueeze(-1)

        required_time = (dist_current_to_node + dist_node_to_depot) / avg_speed + service_time
        exceeds_budget = required_time > remaining + 1e-6

        mask = mask & ~exceeds_budget

        return mask

    def _check_done(self, tensordict: TensorDict) -> torch.Tensor:
        """
        Episode ends at the depot once no customer remains reachable even
        with a freshly reset trip budget -- i.e. true exhaustion, not just
        "chose to return this time" (which would end a single-trip VRPP but
        must not end a multi-trip CTOP episode early).

        Args:
            tensordict: Input TensorDict containing graph structure and node properties.

        Returns:
            torch.Tensor: Boolean tensor indicating if the episode is done, shape (batch,).
        """
        current = tensordict["current_node"].squeeze(-1)
        step = tensordict["i"].squeeze(-1) if tensordict["i"].dim() > 1 else tensordict["i"]
        at_depot = (current == 0) & (step > 0)

        mask = self._get_action_mask(tensordict)
        exhausted = ~mask[:, 1:].any(dim=1)

        return at_depot & exhausted
