"""Batched VRPP primitives shared by the vectorised expert policies.

The vectorised policies mirror the simulator's routing policies (ALNS, HGS, HGS-ALNS,
HVPL, ILS, K-sparse ACO and the random local search route improver) on a batch of
instances at once. They share one solution encoding and the helpers in this module:

- A solution is a **giant tour**: a ``[B, N]`` long tensor holding the visited bins in
  visiting order, packed to the left, with ``0`` padding on the right (``N`` = number of
  bins, the depot is node 0). Bins absent from the row are unvisited.
- Routes are recovered with the Split dynamic programme of Vidal (2012), the same
  decoder as the simulator's :class:`LinearSplit`: the giant tour is cut into
  capacity-feasible depot-to-depot routes that maximise ``R * waste - C * distance``.
  With ``allow_skip=True`` Split may also leave out bins that do not pay, as the
  simulator's Split does in VRPP mode.

Attributes:
    VRPPBatch: Distances, wastes, capacities and objective weights of a batch.
    parse_instance: Build a :class:`VRPPBatch` from an environment TensorDict.
    compact: Pack the bins of each giant tour to the left.
    profit_split: Split giant tours into routes; return routes, profit and cost.
    routes_to_actions: CPTP action sequences for a batch of route lists.

Example:
    >>> inst = parse_instance(td, env)
    >>> routes, profit, cost = profit_split(tours, inst)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from tensordict import TensorDict

from logic.src.constants.simulation import VEHICLE_CAPACITY

NEG_INF = float("-inf")


@dataclass
class VRPPBatch:
    """A batch of VRPP instances.

    Attributes:
        dist: Distance matrices ``[B, M, M]`` with the depot at index 0 (``M = N + 1``).
        waste: Waste per node ``[B, M]`` (0 at the depot).
        capacity: Vehicle capacity per instance ``[B]``.
        R: Revenue per unit of waste.
        C: Cost per unit of distance.
    """

    dist: torch.Tensor
    waste: torch.Tensor
    capacity: torch.Tensor
    R: float = 1.0
    C: float = 1.0

    @property
    def batch_size(self) -> int:
        """Number of instances."""
        return int(self.dist.size(0))

    @property
    def n_bins(self) -> int:
        """Number of bins (customers) per instance."""
        return int(self.dist.size(1)) - 1

    @property
    def device(self) -> torch.device:
        """Device of the instance tensors."""
        return self.dist.device

    def subset(self, idx: torch.Tensor) -> "VRPPBatch":
        """The instances at ``idx``.

        Args:
            idx: Row indices.

        Returns:
            VRPPBatch: Sub-batch sharing the objective weights.
        """
        return VRPPBatch(self.dist[idx], self.waste[idx], self.capacity[idx], self.R, self.C)


def parse_instance(td: TensorDict, env: Optional[Any] = None) -> VRPPBatch:
    """Read distances, wastes, capacities and objective weights from ``td``.

    Distances are the road matrix ``dm`` when present, otherwise Euclidean distances
    between the locations. ``locs`` either includes the depot (row 0, as in a reset
    environment state) or comes with a separate ``depot`` entry that is prepended. Waste without a depot entry gets a 0 prepended.

    Args:
        td: Environment TensorDict.
        env: Environment, read for ``waste_weight`` (R) and ``cost_weight`` (C).

    Returns:
        VRPPBatch: The parsed batch.
    """
    B = td.batch_size[0]
    if "dm" in td.keys():
        dist = td["dm"].detach().float()
    else:
        locs = td["locs"].detach().float()
        if "depot" in td.keys():
            depot = td["depot"].detach().float()
            if depot.dim() == 2:
                depot = depot.unsqueeze(1)
            # A reset environment state already holds the depot in row 0 of ``locs``.
            if not torch.equal(locs[:, :1], depot.expand(-1, 1, -1)):
                locs = torch.cat([depot, locs], dim=1)
        dist = torch.cdist(locs, locs)
    if dist.dim() == 2:
        dist = dist.unsqueeze(0).expand(B, -1, -1)
    dist = dist.clone()
    M = dist.size(-1)
    dist[:, torch.arange(M), torch.arange(M)] = 0.0
    device = dist.device

    if "waste" in td.keys():
        waste = td["waste"].detach().float()
        if waste.dim() == 1:
            waste = waste.unsqueeze(0).expand(B, -1)
        if waste.size(-1) == M - 1:
            waste = torch.cat([torch.zeros_like(waste[:, :1]), waste], dim=-1)
        waste = waste.clone()
        waste[:, 0] = 0.0
    else:
        waste = torch.zeros(B, M, device=device)

    if "capacity" in td.keys():
        cap = td["capacity"].detach().float().reshape(-1)
        cap = cap.expand(B).clone() if cap.numel() == 1 else cap[:B].clone()
    else:
        cap = torch.full((B,), float(VEHICLE_CAPACITY), device=device)

    R = float(getattr(env, "waste_weight", 1.0)) if env is not None else 1.0
    C = float(getattr(env, "cost_weight", 1.0)) if env is not None else 1.0
    return VRPPBatch(dist=dist, waste=waste.to(device), capacity=cap.to(device), R=R, C=C)


def compact(tours: torch.Tensor) -> torch.Tensor:
    """Pack the non-zero entries of every row to the left, keeping their order.

    Args:
        tours: Giant tours ``[B, N]`` with 0 for empty slots.

    Returns:
        torch.Tensor: Packed giant tours ``[B, N]``.
    """
    N = tours.size(1)
    key = (tours == 0).long() * (N + 1) + torch.arange(N, device=tours.device).unsqueeze(0)
    order = torch.argsort(key, dim=1)
    return torch.gather(tours, 1, order)


def lengths(tours: torch.Tensor) -> torch.Tensor:
    """Number of bins in every giant tour.

    Args:
        tours: Giant tours ``[B, N]``.

    Returns:
        torch.Tensor: Lengths ``[B]``.
    """
    return (tours > 0).sum(dim=1)


def visited_mask(tours: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Boolean mask ``[B, M]`` of the nodes present in every giant tour (depot excluded).

    Args:
        tours: Giant tours ``[B, N]``.
        n_nodes: Number of nodes ``M`` including the depot.

    Returns:
        torch.Tensor: Visited mask.
    """
    mask = torch.zeros(tours.size(0), n_nodes, dtype=torch.bool, device=tours.device)
    mask.scatter_(1, tours, True)
    mask[:, 0] = False
    return mask


def tour_from_mask_order(order: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Giant tours made of the nodes of ``order`` whose ``keep`` flag is set.

    Args:
        order: Node order ``[B, N]``.
        keep: Keep flags ``[B, N]`` aligned with ``order``.

    Returns:
        torch.Tensor: Packed giant tours.
    """
    return compact(torch.where(keep, order, torch.zeros_like(order)))


def giant_tour_cost(tours: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """Length of ``depot -> tour -> depot`` with no intermediate depot returns.

    Args:
        tours: Packed giant tours ``[B, N]``.
        dist: Distance matrices ``[B, M, M]``.

    Returns:
        torch.Tensor: Lengths ``[B]``.
    """
    B = tours.size(0)
    zero = torch.zeros(B, 1, dtype=tours.dtype, device=tours.device)
    path = torch.cat([zero, tours, zero], dim=1)
    b = torch.arange(B, device=tours.device).unsqueeze(1)
    return dist[b, path[:, :-1], path[:, 1:]].sum(dim=1)


def _split_tables(
    tours: torch.Tensor, inst: VRPPBatch
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefix sums of load and distance and depot legs along the giant tours."""
    B, N = tours.shape
    b = torch.arange(B, device=tours.device).unsqueeze(1)
    w = torch.gather(inst.waste, 1, tours)
    cum_load = torch.cat([torch.zeros(B, 1, device=tours.device), torch.cumsum(w, dim=1)], dim=1)
    legs = torch.zeros(B, N, device=tours.device)
    if N > 1:
        legs[:, 1:] = inst.dist[b, tours[:, :-1], tours[:, 1:]]
    cum_dist = torch.cat([torch.zeros(B, 1, device=tours.device), torch.cumsum(legs, dim=1)], dim=1)
    d0 = inst.dist[b, 0, tours]  # depot -> node at position k (0-based)
    dx0 = inst.dist[b, tours, 0]  # node -> depot
    return w, cum_load, cum_dist, d0, dx0


def profit_split(  # noqa: C901 - one DP with the skip and route-limit branches of LinearSplit
    tours: torch.Tensor,
    inst: VRPPBatch,
    max_vehicles: int = 0,
    allow_skip: bool = False,
    reconstruct: bool = True,
) -> Tuple[Optional[List[List[List[int]]]], torch.Tensor, torch.Tensor]:
    """Split giant tours into capacity-feasible routes of maximum profit.

    Same recursion as the simulator's ``LinearSplit``: ``V[i]`` is the best profit of the
    first ``i`` tour positions, a route covers a contiguous tour segment whose load fits
    the capacity, and with ``allow_skip`` position ``i`` may be left out (``V[i] =
    V[i-1]``). With ``max_vehicles > 0`` at most that many routes are used.

    Args:
        tours: Packed giant tours ``[B, N]``.
        inst: Instance batch.
        max_vehicles: Route limit (0 = unlimited).
        allow_skip: Allow the split to leave bins out.
        reconstruct: Also return the route lists.

    Returns:
        Tuple of (routes per instance or None, profit ``[B]``, distance ``[B]``).
    """
    B, N = tours.shape
    device = tours.device
    R, C = inst.R, inst.C
    if N == 0:
        z = torch.zeros(B, device=device)
        return ([[] for _ in range(B)] if reconstruct else None), z, z.clone()

    w, cum_load, cum_dist, d0, dx0 = _split_tables(tours, inst)
    cum_rev = torch.cat([torch.zeros(B, 1, device=device), torch.cumsum(w * R, dim=1)], dim=1)
    cap = inst.capacity.unsqueeze(1) + 1e-5
    is_pad = tours == 0

    K = max_vehicles if max_vehicles and max_vehicles > 0 else 0
    if K == 0:
        V = torch.full((B, N + 1), NEG_INF, device=device)
        V[:, 0] = 0.0
        P = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
        for i in range(1, N + 1):
            js = torch.arange(0, i, device=device)
            load = cum_load[:, i : i + 1] - cum_load[:, :i]
            seg_dist = d0[:, :i] + (cum_dist[:, i : i + 1] - cum_dist[:, 1 : i + 1]) + dx0[:, i - 1 : i]
            val = V[:, :i] + (cum_rev[:, i : i + 1] - cum_rev[:, :i]) - C * seg_dist
            val = torch.where(load <= cap, val, torch.full_like(val, NEG_INF))
            best, arg = val.max(dim=1)
            V[:, i] = best
            P[:, i] = js[arg]
            # Padding never costs anything: carry the value over.
            skip = is_pad[:, i - 1] | (allow_skip & (V[:, i - 1] > V[:, i]))
            if allow_skip or bool(is_pad[:, i - 1].any()):
                V[:, i] = torch.where(skip, V[:, i - 1], V[:, i])
                P[:, i] = torch.where(skip, torch.full_like(P[:, i], -2), P[:, i])
        profit = V[:, N]
        P_cpu = P.cpu().tolist()
        tours_cpu = tours.cpu().tolist()
        routes_all: List[List[List[int]]] = []
        for bi in range(B):
            routes: List[List[int]] = []
            i = N
            while i > 0:
                j = P_cpu[bi][i]
                if j == -2:
                    i -= 1
                    continue
                if j < 0:
                    break
                r = [n for n in tours_cpu[bi][j:i] if n != 0]
                if r:
                    routes.append(r)
                i = j
            routes.reverse()
            routes_all.append(routes)
    else:
        V = torch.full((B, K + 1, N + 1), NEG_INF, device=device)
        V[:, 0, 0] = 0.0
        P = torch.full((B, K + 1, N + 1), -1, dtype=torch.long, device=device)
        for i in range(1, N + 1):
            js = torch.arange(0, i, device=device)
            load = cum_load[:, i : i + 1] - cum_load[:, :i]
            seg_dist = d0[:, :i] + (cum_dist[:, i : i + 1] - cum_dist[:, 1 : i + 1]) + dx0[:, i - 1 : i]
            gain = (cum_rev[:, i : i + 1] - cum_rev[:, :i]) - C * seg_dist
            gain = torch.where(load <= cap, gain, torch.full_like(gain, NEG_INF))
            for k in range(1, K + 1):
                val = V[:, k - 1, :i] + gain
                best, arg = val.max(dim=1)
                V[:, k, i] = best
                P[:, k, i] = js[arg]
            skip_from = V[:, :, i - 1]
            skip = is_pad[:, i - 1].unsqueeze(1) | (allow_skip & (skip_from > V[:, :, i]))
            V[:, :, i] = torch.where(skip, skip_from, V[:, :, i])
            P[:, :, i] = torch.where(skip, torch.full_like(P[:, :, i], -2), P[:, :, i])
        profit, kbest = V[:, :, N].max(dim=1)
        P_cpu = P.cpu().tolist()
        tours_cpu = tours.cpu().tolist()
        kb = kbest.cpu().tolist()
        routes_all = []
        for bi in range(B):
            routes = []
            i, k = N, kb[bi]
            while i > 0 and k >= 0:
                j = P_cpu[bi][k][i]
                if j == -2:
                    i -= 1
                    continue
                if j < 0:
                    break
                r = [n for n in tours_cpu[bi][j:i] if n != 0]
                if r:
                    routes.append(r)
                i, k = j, k - 1
            routes.reverse()
            routes_all.append(routes)

    # Instances where no feasible split exists (a bin heavier than the vehicle):
    # fall back to dropping the infeasible bins' contribution.
    profit = torch.where(torch.isfinite(profit), profit, torch.full_like(profit, -1e18))
    cost = torch.tensor(
        [route_list_distance(r, inst.dist[bi]) for bi, r in enumerate(routes_all)],
        device=device,
        dtype=torch.float32,
    )
    rev = torch.tensor(
        [float(inst.waste[bi, [n for rr in r for n in rr]].sum()) * R if r else 0.0 for bi, r in enumerate(routes_all)],
        device=device,
        dtype=torch.float32,
    )
    profit = rev - C * cost
    return (routes_all if reconstruct else None), profit, cost


def route_list_distance(routes: List[List[int]], dist: torch.Tensor) -> float:
    """Total distance of depot-to-depot routes.

    Args:
        routes: Routes of one instance (bins only).
        dist: Distance matrix ``[M, M]`` of that instance.

    Returns:
        float: Total distance.
    """
    total = 0.0
    for r in routes:
        if not r:
            continue
        path = [0] + list(r) + [0]
        idx = torch.tensor(path, device=dist.device)
        total += float(dist[idx[:-1], idx[1:]].sum())
    return total


def routes_to_tours(routes: List[List[List[int]]], n_bins: int, device: Any) -> torch.Tensor:
    """Giant tours ``[B, N]`` that list the bins of ``routes`` in route order.

    Args:
        routes: Route lists, one per instance.
        n_bins: Number of bins ``N``.
        device: Target device.

    Returns:
        torch.Tensor: Packed giant tours.
    """
    out = torch.zeros(len(routes), n_bins, dtype=torch.long, device=device)
    for b, rl in enumerate(routes):
        flat = [n for r in rl for n in r][:n_bins]
        if flat:
            out[b, : len(flat)] = torch.tensor(flat, dtype=torch.long, device=device)
    return out


def routes_to_actions(routes: List[List[List[int]]], device: Any) -> torch.Tensor:
    """Padded CPTP action sequences: each route then a depot return, then the day end.

    Same layout as the profitable nearest-neighbour expert: ``r1 0 r2 0 ... rk 0 0``;
    staying at the depot is a single ``0``.

    Args:
        routes: Route lists, one per instance.
        device: Target device.

    Returns:
        torch.Tensor: Actions ``[B, L]`` padded with 0.
    """
    seqs: List[List[int]] = []
    for rl in routes:
        seq: List[int] = []
        for r in rl:
            if r:
                seq.extend(int(n) for n in r)
                seq.append(0)
        seqs.append(seq + [0] if seq else [0])
    width = max(len(s) for s in seqs) + 1
    actions = torch.zeros(len(seqs), width, dtype=torch.long, device=device)
    for b, s in enumerate(seqs):
        actions[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return actions


def replay_reward(td: TensorDict, env: Optional[Any], actions: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Reward of ``actions`` replayed through ``env``; ``fallback`` without an environment.

    Args:
        td: Reset TensorDict.
        env: Environment or None.
        actions: Action sequences ``[B, L]``.
        fallback: Reward to return when there is no environment ``[B]``.

    Returns:
        torch.Tensor: Reward ``[B]``.
    """
    if env is None or not hasattr(env, "step"):
        return fallback
    with torch.no_grad():
        cur = env.reset(td.clone())
        for i in range(actions.size(1)):
            cur["action"] = actions[:, i]
            cur = env.step(cur)["next"]
        return env.get_reward(cur, actions)


def hash_rows(tours: torch.Tensor) -> List[Tuple[int, ...]]:
    """Hashable key per giant tour, for visited-solution tracking.

    Args:
        tours: Packed giant tours ``[B, N]``.

    Returns:
        List of tuples, one per row.
    """
    return [tuple(int(x) for x in row if x) for row in tours.cpu().tolist()]
