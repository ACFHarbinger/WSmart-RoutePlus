"""The empirical distribution draws new values on each call with the same generator."""

import pandas as pd
import pytest
import torch

from logic.src.data.distributions import Empirical

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _dist() -> Empirical:
    data = pd.DataFrame(torch.arange(0, 400).reshape(100, 4).numpy() % 100)
    return Empirical(dataset=data).set_sampling_method("sample_tensor")


def test_consecutive_days_differ():
    dist, generator = _dist(), torch.Generator().manual_seed(42)

    day_one = dist.sample((8, 4), rng=generator)
    day_two = dist.sample((8, 4), rng=generator)

    assert not torch.equal(day_one, day_two)


def test_same_seed_reproduces_the_same_days():
    first = [_dist().sample((8, 4), rng=g) for g in [torch.Generator().manual_seed(7)] for _ in range(1)]
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    d1, d2 = _dist(), _dist()

    assert torch.equal(first[0], d1.sample((8, 4), rng=g1))
    assert torch.equal(d1.sample((8, 4), rng=g1), [d2.sample((8, 4), rng=g2) for _ in range(2)][1])


def test_gamma_presets_draw_new_values_per_call():
    from logic.src.data.distributions import Gamma

    dist, generator = Gamma(option=2).set_sampling_method("sample_tensor"), torch.Generator().manual_seed(42)

    assert not torch.equal(dist.sample((8, 20), rng=generator), dist.sample((8, 20), rng=generator))
