"""Profitable nearest-neighbour expert for imitation learning on PTP and CPTP.

Builds a nearest-neighbour tour over the bins that hold waste, shortens it with 2-opt, then
drops each bin whose detour costs more than its waste is worth. When the remaining tour does
not pay, the vehicle stays at the depot. Distances are the environment's road matrix ``dm``
when present, otherwise the Euclidean distance between ``locs``.

The action sequence follows the CPTP day end: the tour returns to the depot and the depot is
chosen again to end the day; staying home is the depot chosen from the depot.

The three phases run on the whole batch at once with NumPy. Each phase keeps the move order,
tie-breaks and floating-point arithmetic of the original one-instance-at-a-time version, so
the tours are identical to it:

- nearest-neighbour ties go to the bin met first in the iteration order of the set of bins
  with waste, as ``min`` over that set did;
- 2-opt is first-improvement: after a move, the scan goes on from the next ``j`` for the same
  ``i`` on the changed tour, and every pass rebuilds the prefix sums;
- the drop phase removes the first bin of largest gain.

Attributes:
    ProfitableNearestNeighbourPolicy: Expert policy producing selective tours.

Example:
    >>> policy = ProfitableNearestNeighbourPolicy(env_name="cptp")
    >>> out = policy(env.reset(td), env)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase

# Bump when a change can alter the expert's tours; cached expert solves carry it.
EXPERT_VERSION = 1


def _paths(tours: List[List[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Pad ``[0] + tour + [0]`` for every tour; return the paths and their lengths."""
    lens = np.array([len(t) + 2 for t in tours], dtype=np.int64)
    paths = np.zeros((len(tours), int(lens.max(initial=2))), dtype=np.int64)
    for b, t in enumerate(tours):
        paths[b, 1 : 1 + len(t)] = t
    return paths, lens


def _nearest_neighbour(dms: np.ndarray, wastes: np.ndarray) -> List[List[int]]:
    """Nearest-neighbour order over the bins with waste, for every instance."""
    orders = [list({int(j) for j in np.flatnonzero(w[1:] > 0) + 1}) for w in wastes]
    width = max((len(o) for o in orders), default=0)
    if width == 0:
        return [[] for _ in orders]
    batch = len(orders)
    cand = np.zeros((batch, width), dtype=np.int64)
    left = np.zeros((batch, width), dtype=bool)
    for b, o in enumerate(orders):
        cand[b, : len(o)] = o
        left[b, : len(o)] = True
    rows = np.arange(batch)
    cur = np.zeros(batch, dtype=np.int64)
    tours = np.zeros((batch, width), dtype=np.int64)
    for step in range(width):
        live = left.any(axis=1)
        dist = np.where(left, dms[rows[:, None], cur[:, None], cand], np.inf)
        k = dist.argmin(axis=1)  # first minimum, as min() over the set kept the first
        nxt = cand[rows, k]
        tours[live, step] = nxt[live]
        left[rows[live], k[live]] = False
        cur = np.where(live, nxt, cur)
    return [tours[b, : len(o)].tolist() for b, o in enumerate(orders)]


def _prefix(dms: np.ndarray, rows: np.ndarray, paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward and backward prefix sums of the legs of ``paths`` (float32 sums, as float64)."""
    fwd_legs = dms[rows[:, None], paths[:, :-1], paths[:, 1:]]
    bwd_legs = dms[rows[:, None], paths[:, 1:], paths[:, :-1]]
    zero = np.zeros((len(rows), 1))
    return (
        np.concatenate([zero, np.cumsum(fwd_legs, axis=1)], axis=1),
        np.concatenate([zero, np.cumsum(bwd_legs, axis=1)], axis=1),
    )


def _two_opt(dms: np.ndarray, tours: List[List[int]], passes: int) -> List[List[int]]:
    """First-improvement 2-opt on every tour, at most ``passes`` full sweeps each."""
    paths, lens = _paths(tours)
    width = paths.shape[1]
    cols = np.arange(width)
    jcols = cols[:-1]
    fwd = np.zeros((len(tours), width))
    bwd = np.zeros((len(tours), width))
    active = lens >= 4  # shorter tours have no 2-opt move
    for _ in range(passes):
        todo = np.flatnonzero(active)
        if todo.size == 0:
            break
        fwd[todo], bwd[todo] = _prefix(dms, todo, paths[todo])
        improved = np.zeros(len(tours), dtype=bool)
        for i in range(1, width - 2):
            rows = todo[i <= lens[todo] - 3]
            start = np.full(rows.size, i + 1)
            while rows.size:
                p = paths[rows]
                a, b = p[:, i - 1 : i], p[:, i : i + 1]
                c, d = p[:, :-1], p[:, 1:]
                r = rows[:, None]
                # Road distances are asymmetric, so a reversed segment is priced from its own sum.
                old = dms[r, a, b] + (fwd[rows, :-1] - fwd[rows, i : i + 1]) + dms[r, c, d]
                new = dms[r, a, c] + (bwd[rows, :-1] - bwd[rows, i : i + 1]) + dms[r, b, d]
                valid = (jcols[None, :] >= start[:, None]) & (jcols[None, :] <= lens[rows, None] - 2)
                better = (new < old - 1e-9) & valid
                has = better.any(axis=1)
                rows, j = rows[has], better[has].argmax(axis=1)
                if rows.size == 0:
                    break
                seg = (cols[None, :] >= i) & (cols[None, :] <= j[:, None])
                order = np.where(seg, i + j[:, None] - cols[None, :], cols[None, :])
                paths[rows] = np.take_along_axis(paths[rows], order, axis=1)
                fwd[rows], bwd[rows] = _prefix(dms, rows, paths[rows])
                improved[rows] = True
                start = j + 1
        active = improved
    return [paths[b, 1 : lens[b] - 1].tolist() for b in range(len(tours))]


def _drop_unprofitable(
    dms: np.ndarray, wastes: np.ndarray, tours: List[List[int]], revenue: float, cost: float
) -> List[List[int]]:
    """Remove, one at a time, the first bin of largest positive gain from every tour."""
    paths, lens = _paths(tours)
    width = paths.shape[1]
    cols = np.arange(width)
    rows = np.flatnonzero(lens > 2)
    while rows.size:
        p = paths[rows]
        r = rows[:, None]
        prev, node, nxt = p[:, :-2], p[:, 1:-1], p[:, 2:]
        detour = dms[r, prev, node] + dms[r, node, nxt] - dms[r, prev, nxt]
        gains = cost * detour.astype(np.float64) - revenue * wastes[r, node].astype(np.float64)
        gains = np.where(cols[None, : width - 2] < lens[rows, None] - 2, gains, -np.inf)
        k = gains.argmax(axis=1)
        keep = gains[np.arange(rows.size), k] > 0
        rows, k = rows[keep], k[keep]
        if rows.size == 0:
            break
        src = np.minimum(np.where(cols[None, :] >= k[:, None] + 1, cols[None, :] + 1, cols[None, :]), width - 1)
        paths[rows] = np.take_along_axis(paths[rows], src, axis=1)
        lens[rows] -= 1
        rows = rows[lens[rows] > 2]
    return [paths[b, 1 : lens[b] - 1].tolist() for b in range(len(tours))]


class ProfitableNearestNeighbourPolicy(torch.nn.Module):
    """Selective nearest-neighbour expert.

    Attributes:
        env_name: Environment identifier.
        two_opt_passes: Maximum number of full 2-opt sweeps per tour.
    """

    def __init__(self, env_name: str, two_opt_passes: int = 5, seed: int = 42, device: str = "cpu", **kwargs: Any):
        """Initialise the expert.

        Args:
            env_name: Environment identifier.
            two_opt_passes: Maximum number of full 2-opt sweeps per tour.
            seed: Unused; the expert is deterministic.
            device: Unused; actions are placed on the input's device.
            kwargs: Ignored extra configuration keys.
        """
        super().__init__()
        self.env_name = env_name
        self.two_opt_passes = two_opt_passes

    @staticmethod
    def _tour_cost(D: np.ndarray, tour: List[int]) -> float:
        path = [0] + tour + [0]
        return float(sum(D[a, b] for a, b in zip(path[:-1], path[1:], strict=True)))

    def solve(self, dms: np.ndarray, wastes: np.ndarray, revenue: float, cost: float) -> List[List[int]]:
        """Return the bins to visit, in order, for every instance.

        Args:
            dms: Distance matrices ``[batch, nodes, nodes]`` (float32), depot at 0.
            wastes: Waste per node ``[batch, nodes]`` (float32), depot at 0.
            revenue: Reward per unit of waste.
            cost: Cost per unit of distance.

        Returns:
            One list of bins per instance; empty when no tour pays.
        """
        tours = _nearest_neighbour(dms, wastes)
        tours = _two_opt(dms, tours, self.two_opt_passes)
        tours = _drop_unprofitable(dms, wastes, tours, revenue, cost)
        for b, tour in enumerate(tours):
            if tour and revenue * wastes[b, tour].sum() - cost * self._tour_cost(dms[b], tour) <= 0:
                tours[b] = []
        return tours

    def forward(self, td: TensorDict, env: Optional[RL4COEnvBase] = None, **kwargs: Any) -> Dict[str, Any]:
        """Solve every instance of the batch.

        Args:
            td: Reset TensorDict with ``waste``, ``capacity`` and ``dm`` or ``locs``.
            env: Environment, used for the reward weights and to replay the actions.
            kwargs: Ignored decoding arguments.

        Returns:
            Dict with padded ``actions`` and the replayed ``reward``.
        """
        device = td.device if td.device is not None else td["locs"].device
        dms, wastes, caps, revenue, cost = self.inputs(td, env)
        actions = self.actions(self.solve(dms, wastes, revenue, cost), wastes, caps, device)
        return {"actions": actions, "reward": self._replay_reward(td, env, actions)}

    @staticmethod
    def actions(tours: List[List[int]], wastes: np.ndarray, caps: np.ndarray, device: Any) -> torch.Tensor:
        """Padded CPTP action sequences for ``tours``: a depot return whenever the load would
        exceed capacity, then the day end (depot, depot); staying home is a single depot."""
        sequences = []
        for b, tour in enumerate(tours):
            seq: List[int] = []
            load = 0.0
            for j in tour:
                if load + wastes[b, j] > caps[b]:
                    seq.append(0)
                    load = 0.0
                seq.append(j)
                load += wastes[b, j]
            # Return to the depot, then choose it again to end the day.
            sequences.append(seq + [0, 0] if seq else [0])
        width = max(len(s) for s in sequences) + 1
        actions = torch.zeros(len(sequences), width, dtype=torch.long, device=device)
        for b, s in enumerate(sequences):
            actions[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        return actions

    @staticmethod
    def inputs(td: TensorDict, env: Optional[RL4COEnvBase]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Everything the expert reads: distances, waste with the depot, capacity and weights."""
        if "dm" in td.keys():
            dms = td["dm"].detach().float().cpu().numpy()
        else:
            locs = td["locs"].detach().float()
            dms = torch.cdist(locs, locs).cpu().numpy()
        wastes = td["waste"].detach().float()
        if wastes.size(-1) == dms.shape[-1] - 1:
            wastes = torch.cat([torch.zeros_like(wastes[:, :1]), wastes], dim=-1)
        caps = td["capacity"].detach().float().reshape(-1).expand(td.batch_size[0]).cpu().numpy()
        revenue = float(getattr(env, "waste_weight", 1.0))
        cost = float(getattr(env, "cost_weight", 1.0))
        return dms, wastes.cpu().numpy(), caps, revenue, cost

    @staticmethod
    def _replay_reward(td: TensorDict, env: Optional[RL4COEnvBase], actions: torch.Tensor) -> torch.Tensor:
        if env is None:
            return torch.zeros(td.batch_size[0], device=actions.device)
        with torch.no_grad():
            cur = env.reset(td.clone())
            for i in range(actions.size(1)):
                cur["action"] = actions[:, i]
                cur = env.step(cur)["next"]
            return env.get_reward(cur, actions)
