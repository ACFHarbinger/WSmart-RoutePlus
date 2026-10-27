"""Disk cache of profitable nearest-neighbour expert solves.

A record maps the expert's full input for one instance to the tour it chose. The key is a hash
of everything the tour depends on: the distance matrix, the waste of every node (depot
included), the capacity, the revenue and cost weights, the number of 2-opt passes and
``EXPERT_VERSION``. A state that differs in any of them misses and is solved again, so a record
is never used for a state it was not computed for, whatever the day, dataset or fill history.

Only tours are stored. The action sequence and the reward are rebuilt from the tour and the
current state on every use, the reward by replaying the actions in the environment, exactly as
for a fresh solve; ``weighted_nll`` therefore sees the same reward with or without the cache.

Records are written as shards (``shard-*.pt``) in the cache directory, one per flush, and every
shard in the directory is read when the cache opens. With ``rl.imitation.fill_from=expert`` the
states of a run depend only on the dataset and the settings, so a rerun or a resumed run finds
every solve of the days already trained.

Attributes:
    CachedExpert: Expert wrapper that reads and writes the cache.

Example:
    >>> expert = CachedExpert(ProfitableNearestNeighbourPolicy("cptp"), "data/cache/expert")
    >>> out = expert(env.reset(td), env)
    >>> expert.flush()
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tensordict import TensorDict

from logic.src.policies.vectorized.routing.profitable_nearest_neighbour import (
    EXPERT_VERSION,
    ProfitableNearestNeighbourPolicy,
)
from logic.src.tracking.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)

SHARD_FORMAT = 1


class CachedExpert(torch.nn.Module):
    """Profitable nearest-neighbour expert with a disk cache of its tours.

    Attributes:
        expert: The wrapped expert.
        directory: Cache directory.
        hits: Instances answered from the cache since creation.
        misses: Instances solved since creation.
    """

    def __init__(self, expert: ProfitableNearestNeighbourPolicy, directory: str, flush_every: int = 100_000):
        """Open the cache and read every shard in ``directory``.

        Args:
            expert: Expert whose solves are cached.
            directory: Cache directory; created when missing.
            flush_every: Write a shard once this many new records are pending.
        """
        super().__init__()
        self.expert = expert
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.hits = 0
        self.misses = 0
        self._records: Dict[str, List[int]] = {}
        self._pending: Dict[str, List[int]] = {}
        for shard in sorted(self.directory.glob("shard-*.pt")):
            data = torch.load(shard, weights_only=True)
            if data.get("format") == SHARD_FORMAT and data.get("version") == EXPERT_VERSION:
                self._records.update(data["records"])
        logger.info(f"Expert cache {self.directory}: {len(self._records)} records")

    def __len__(self) -> int:
        return len(self._records)

    def _keys(self, dms: np.ndarray, wastes: np.ndarray, caps: np.ndarray, revenue: float, cost: float) -> List[str]:
        head = hashlib.blake2b(digest_size=16)
        head.update(f"profitable_nn:{EXPERT_VERSION}:{self.expert.two_opt_passes}:".encode())
        head.update(np.array([revenue, cost], dtype=np.float64).tobytes())
        # The environment expands one road matrix over the batch; hash it once.
        shared = dms.strides[0] == 0
        shared_digest = hashlib.blake2b(dms[0].tobytes(), digest_size=16).digest() if shared else b""
        keys = []
        for b in range(len(wastes)):
            h = head.copy()
            h.update(shared_digest or hashlib.blake2b(dms[b].tobytes(), digest_size=16).digest())
            h.update(f"{dms.shape[-1]}:".encode())
            h.update(np.ascontiguousarray(wastes[b], dtype=np.float32).tobytes())
            h.update(np.float32(caps[b]).tobytes())
            keys.append(h.hexdigest())
        return keys

    def forward(self, td: TensorDict, env: Optional[Any] = None, **kwargs: Any) -> Dict[str, Any]:
        """Expert output for ``td``, solving only the instances not in the cache.

        Args:
            td: Reset TensorDict, as for the expert.
            env: Environment, as for the expert.
            kwargs: Ignored decoding arguments.

        Returns:
            Dict with padded ``actions`` and the replayed ``reward``, as from the expert.
        """
        device = td.device if td.device is not None else td["locs"].device
        dms, wastes, caps, revenue, cost = self.expert.inputs(td, env)
        keys = self._keys(dms, wastes, caps, revenue, cost)
        tours: List[Optional[List[int]]] = [self._records.get(k) for k in keys]
        miss = [b for b, t in enumerate(tours) if t is None]
        if miss:
            sub = np.broadcast_to(dms[0], (len(miss),) + dms.shape[1:]) if dms.strides[0] == 0 else dms[miss]
            for b, tour in zip(miss, self.expert.solve(sub, wastes[miss], revenue, cost), strict=True):
                tours[b] = tour
                self._records[keys[b]] = tour
                self._pending[keys[b]] = tour
        self.hits += len(keys) - len(miss)
        self.misses += len(miss)
        if len(self._pending) >= self.flush_every:
            self.flush()
        actions = self.expert.actions(tours, wastes, caps, device)  # type: ignore[arg-type]
        return {"actions": actions, "reward": self.expert._replay_reward(td, env, actions)}

    def flush(self) -> Optional[Path]:
        """Write the pending records as a new shard; return its path, or None if none pending."""
        if not self._pending:
            return None
        path = self.directory / f"shard-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{len(self._records)}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "format": SHARD_FORMAT,
                "expert": "profitable_nn",
                "version": EXPERT_VERSION,
                "two_opt_passes": self.expert.two_opt_passes,
                "records": self._pending,
            },
            tmp,
        )
        tmp.replace(path)  # a crash mid-write leaves a .tmp, never a partial shard
        logger.info(f"Expert cache: wrote {len(self._pending)} records to {path.name}")
        self._pending = {}
        return path
