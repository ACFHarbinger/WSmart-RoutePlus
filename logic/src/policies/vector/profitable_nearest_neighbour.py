"""Profitable nearest-neighbour expert for imitation learning on PTP and CPTP.

Builds a nearest-neighbour tour over the bins that hold waste, shortens it with 2-opt, then
drops each bin whose detour costs more than its waste is worth. When the remaining tour does
not pay, the vehicle stays at the depot. Distances are the environment's road matrix ``dm``
when present, otherwise the Euclidean distance between ``locs``.

The action sequence follows the CPTP day end: the tour returns to the depot and the depot is
chosen again to end the day; staying home is the depot chosen from the depot.

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
        return float(sum(D[a, b] for a, b in zip(path[:-1], path[1:])))

    def _two_opt(self, D: np.ndarray, tour: List[int]) -> List[int]:
        path = [0] + tour + [0]

        def prefix(p: List[int]):
            # Road distances are asymmetric, so a reversed segment is priced from its own sum.
            fwd = np.concatenate([[0.0], np.cumsum(D[p[:-1], p[1:]])])
            bwd = np.concatenate([[0.0], np.cumsum(D[p[1:], p[:-1]])])
            return fwd, bwd

        for _ in range(self.two_opt_passes):
            improved = False
            fwd, bwd = prefix(path)
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):
                    a, b, c, d = path[i - 1], path[i], path[j], path[j + 1]
                    old = D[a, b] + (fwd[j] - fwd[i]) + D[c, d]
                    new = D[a, c] + (bwd[j] - bwd[i]) + D[b, d]
                    if new < old - 1e-9:
                        path[i : j + 1] = path[i : j + 1][::-1]
                        fwd, bwd = prefix(path)
                        improved = True
            if not improved:
                break
        return path[1:-1]

    def _solve(self, D: np.ndarray, waste: np.ndarray, revenue: float, cost: float) -> List[int]:
        """Return the bins to visit, in order, for one instance (waste is indexed by node, depot 0)."""
        left = {int(j) for j in np.flatnonzero(waste[1:] > 0) + 1}
        tour: List[int] = []
        cur = 0
        while left:
            nxt = min(left, key=lambda x: D[cur, x])
            tour.append(nxt)
            left.remove(nxt)
            cur = nxt
        tour = self._two_opt(D, tour)
        while tour:
            path = [0] + tour + [0]
            gains = [
                cost * (D[path[k - 1], path[k]] + D[path[k], path[k + 1]] - D[path[k - 1], path[k + 1]])
                - revenue * waste[path[k]]
                for k in range(1, len(path) - 1)
            ]
            k = int(np.argmax(gains))
            if gains[k] <= 0:
                break
            del tour[k]
        if tour and revenue * waste[tour].sum() - cost * self._tour_cost(D, tour) <= 0:
            return []
        return tour

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
        if "dm" in td.keys():
            dms = td["dm"].detach().float().cpu().numpy()
        else:
            locs = td["locs"].detach().float()
            dms = torch.cdist(locs, locs).cpu().numpy()
        wastes = td["waste"].detach().float()
        if wastes.size(-1) == dms.shape[-1] - 1:
            wastes = torch.cat([torch.zeros_like(wastes[:, :1]), wastes], dim=-1)
        wastes = wastes.cpu().numpy()
        caps = td["capacity"].detach().float().reshape(-1).expand(td.batch_size[0]).cpu().numpy()
        revenue = float(getattr(env, "waste_weight", 1.0))
        cost = float(getattr(env, "cost_weight", 1.0))

        sequences = []
        for b in range(td.batch_size[0]):
            tour = self._solve(dms[b], wastes[b], revenue, cost)
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
        return {"actions": actions, "reward": self._replay_reward(td, env, actions)}

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
