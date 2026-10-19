"""The expert cache answers only states it solved, with the output of a fresh solve."""

import pytest
import torch
from logic.src.pipeline.rl.envs.cptp import CPTPEnv
from logic.src.policies.vector.expert_cache import CachedExpert
from logic.src.policies.vector.profitable_nearest_neighbour import ProfitableNearestNeighbourPolicy
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _reset(waste: torch.Tensor, dm: torch.Tensor) -> tuple[CPTPEnv, TensorDict]:
    batch, n = waste.shape
    env = CPTPEnv(generator_params={"num_loc": n}, check_env_specs=False, waste_weight=10.0, cost_weight=1.0)
    td = TensorDict(
        {
            "locs": torch.rand(batch, n, 2, generator=torch.Generator().manual_seed(1)),
            "depot": torch.zeros(batch, 2),
            "waste": waste,
            "capacity": torch.full((batch,), 2.0),
            # One road matrix expanded over the batch, as the environment attaches it.
            "dm": dm.expand(batch, -1, -1),
        },
        batch_size=[batch],
    )
    return env, env.reset(td)


def _state(seed: int = 0, batch: int = 6, n: int = 12):
    g = torch.Generator().manual_seed(seed)
    dm = torch.rand(n + 1, n + 1, generator=g) * 2
    return torch.rand(batch, n, generator=g), dm


def test_hits_give_the_fresh_solve_and_change_nothing(tmp_path) -> None:
    waste, dm = _state()
    env, td = _reset(waste, dm)
    fresh = ProfitableNearestNeighbourPolicy("cptp")(td, env)
    cached = CachedExpert(ProfitableNearestNeighbourPolicy("cptp"), str(tmp_path))

    first = cached(td, env)
    second = cached(td, env)

    assert (cached.misses, cached.hits) == (6, 6)
    for out in (first, second):
        assert torch.equal(out["actions"], fresh["actions"])
        assert torch.equal(out["reward"], fresh["reward"])


def test_any_change_to_the_expert_input_misses(tmp_path) -> None:
    waste, dm = _state()
    cached = CachedExpert(ProfitableNearestNeighbourPolicy("cptp"), str(tmp_path))
    cached(*reversed(_reset(waste, dm)))

    changed = waste.clone()
    changed[0, 3] += 0.01  # one bin of one instance
    env, td = _reset(changed, dm)
    cached(td, env)
    assert (cached.misses, cached.hits) == (7, 5)

    env, td = _reset(waste, dm * 1.01)  # another road matrix
    cached(td, env)
    assert cached.misses == 13

    env.waste_weight = 3.0  # other economic weights
    cached(td, env)
    assert cached.misses == 19

    other_passes = CachedExpert(ProfitableNearestNeighbourPolicy("cptp", two_opt_passes=1), str(tmp_path))
    other_passes(*reversed(_reset(waste, dm)))
    assert other_passes.hits == 0


def test_records_survive_reopening_the_directory(tmp_path) -> None:
    waste, dm = _state(seed=3)
    env, td = _reset(waste, dm)
    cached = CachedExpert(ProfitableNearestNeighbourPolicy("cptp"), str(tmp_path))
    expected = cached(td, env)["actions"]
    assert cached.flush() is not None
    assert cached.flush() is None  # nothing pending

    reopened = CachedExpert(ProfitableNearestNeighbourPolicy("cptp"), str(tmp_path))
    assert len(reopened) == 6
    assert torch.equal(reopened(td, env)["actions"], expected)
    assert (reopened.misses, reopened.hits) == (0, 6)
