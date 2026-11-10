"""Time training on several devices: tours gathered by dataset row, shared baselines and state.

The distributed tests start two CPU processes (gloo backend), each playing an interleaved half of the pool as a
distributed sampler would, and check that both end with the update a single process makes from all the tours.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from logic.src.data.datasets import TensorDictDataset
from logic.src.pipeline.learning.common.baselines.base import global_batch_mean
from logic.src.pipeline.learning.common.baselines.rollout import _rank0_decision
from logic.src.pipeline.learning.common.epoch import (
    _build_visited_mask,
    apply_time_step,
    broadcast_time_state,
    gather_epoch_actions,
)
from tensordict import TensorDict

N_INST, N_BINS, BATCH = 8, 5, 2


def _tours() -> torch.Tensor:
    """One tour per instance (node ids, 0 = depot), padded with zeros: instance i visits bins i % 5 + 1 and 5."""
    tours = torch.zeros(N_INST, 4, dtype=torch.long)
    for i in range(N_INST):
        tours[i, :3] = torch.tensor([i % N_BINS + 1, N_BINS, 0])
    return tours


def _dataset() -> TensorDictDataset:
    waste_full = torch.full((N_INST, 3, N_BINS), 0.1)
    td = TensorDict(
        {
            "locs": torch.rand(N_INST, N_BINS, 2, generator=torch.Generator().manual_seed(0)),
            "depot": torch.full((N_INST, 2), 5.0),
            "waste": torch.full((N_INST, N_BINS), 0.5),  # the current day's fills
            "_waste_full": waste_full,  # the deposit series
        },
        batch_size=[N_INST],
    )
    return TensorDictDataset(td)


def _expected_waste() -> torch.Tensor:
    ds = _dataset()
    tours = _tours()
    batches = [tours[i : i + BATCH] for i in range(0, N_INST, BATCH)]
    apply_time_step(ds, batches, day=0, env=None)
    return ds.data["waste"].clone()


def test_rows_place_tours_on_their_instances_in_any_order():
    tours = _tours()
    perm = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4])
    in_order = _build_visited_mask([tours], N_INST, N_BINS, torch.device("cpu"))
    shuffled = _build_visited_mask(
        [tours[perm[:4]], tours[perm[4:]]], N_INST, N_BINS, torch.device("cpu"), [perm[:4], perm[4:]]
    )
    assert torch.equal(in_order, shuffled)
    assert not in_order[:, 0].any()


def test_repeated_row_keeps_its_first_tour():
    tours = _tours()
    rows = torch.arange(N_INST)
    other = torch.zeros(1, 4, dtype=torch.long)
    other[0, 0] = 2  # a second, different tour of instance 0
    mask = _build_visited_mask([tours, other], N_INST, N_BINS, torch.device("cpu"), [rows, torch.tensor([0])])
    assert torch.equal(mask, _build_visited_mask([tours], N_INST, N_BINS, torch.device("cpu")))


def test_single_process_gather_is_identity():
    acts, rows = [torch.ones(2, 3)], [torch.tensor([0, 1])]
    out_a, out_r = gather_epoch_actions(acts, rows)
    assert out_a == acts and out_r == rows
    assert gather_epoch_actions(acts, [])[1] is None


def _worker(rank: int, world: int, port: int, results) -> None:
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        ds, tours = _dataset(), _tours()
        # DistributedSampler(shuffle=False): rank r gets instances r, r + world, ...; batches of BATCH.
        mine = torch.arange(rank, N_INST, world)
        acts = [tours[mine[i : i + BATCH]] for i in range(0, len(mine), BATCH)]
        rows = [mine[i : i + BATCH] for i in range(0, len(mine), BATCH)]
        all_acts, all_rows = gather_epoch_actions(acts, rows)
        apply_time_step(ds, all_acts, day=0, env=None, epoch_rows=all_rows)
        if rank == 1:
            ds.data["waste"] = ds.data["waste"] + 1.0  # a divergence the broadcast must remove
        broadcast_time_state(ds)
        reward = torch.tensor([1.0, 3.0]) if rank == 0 else torch.tensor([5.0, 7.0])
        results[rank] = {
            "waste": ds.data["waste"].clone(),
            "mean": float(global_batch_mean(reward)),
            "decision": _rank0_decision(rank == 0),
        }
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_two_processes_end_with_the_single_process_update():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_worker, args=(2, port, results), nprocs=2, join=True)
        results = dict(results)
    expected = _expected_waste()
    for rank in (0, 1):
        assert torch.allclose(results[rank]["waste"], expected)
        assert results[rank]["mean"] == pytest.approx(4.0)
        assert results[rank]["decision"] is True
