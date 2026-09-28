"""M-cursor-03: scalar 1-based IDs equal the vectorized mask on the same fills."""

from typing import List, Optional

import numpy as np
import pytest
import torch
from logic.src.interfaces.context.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.selection_last_minute import LastMinuteSelection
from logic.src.policies.mandatory_selection.selection_lookahead import LookaheadSelection
from logic.src.policies.mandatory_selection.selection_service_level import ServiceLevelSelection
from logic.src.policies.vector.selection.last_minute import LastMinuteSelector
from logic.src.policies.vector.selection.lookahead import LookaheadSelector
from logic.src.policies.vector.selection.service_level import ServiceLevelSelector

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _percent_to_vector(values: List[float]) -> torch.Tensor:
    """Customer percents → fraction fills with a depot column prepended."""
    return torch.tensor([[0.0] + [v / 100.0 for v in values]], dtype=torch.float32)


def _ids_from_mask(mask: torch.Tensor) -> List[int]:
    """Vectorized [B, N] mask (index 0 = depot) → 1-based customer IDs."""
    return (mask[0, 1:].nonzero(as_tuple=True)[0] + 1).tolist()


def _scalar_context(
    fills: List[float],
    threshold: float = 0.0,
    rates: Optional[List[float]] = None,
    stds: Optional[List[float]] = None,
    horizon_days: int = 1,
    current_collection_day: int = 0,
) -> SelectionContext:
    n = len(fills)
    return SelectionContext(
        bin_ids=np.arange(n),
        current_fill=np.array(fills, dtype=np.float64),
        accumulation_rates=None if rates is None else np.array(rates, dtype=np.float64),
        std_deviations=None if stds is None else np.array(stds, dtype=np.float64),
        threshold=threshold,
        horizon_days=horizon_days,
        current_collection_day=current_collection_day,
    )


def test_last_minute_scalar_ids_match_vectorized_mask():
    fills = [50.0, 70.0, 95.0]
    threshold = 70.0
    scalar_ids, _ = LastMinuteSelection().select_bins(_scalar_context(fills, threshold=threshold))
    mask = LastMinuteSelector(threshold=threshold).select(_percent_to_vector(fills))
    assert scalar_ids == [2, 3]
    assert _ids_from_mask(mask) == scalar_ids
    assert mask[0, 0].item() is False


def test_service_level_scalar_ids_match_vectorized_mask():
    fills = [95.0, 50.0, 80.0]
    rates = [5.0, 5.0, 30.0]
    stds = [1.0, 1.0, 1.0]
    z = 0.84
    horizon = 1
    ctx = _scalar_context(fills, threshold=z, rates=rates, stds=stds, horizon_days=horizon)
    scalar_ids, _ = ServiceLevelSelection().select_bins(ctx)
    mask = ServiceLevelSelector(confidence_factor=z, horizon_days=horizon).select(
        _percent_to_vector(fills),
        accumulation_rates=_percent_to_vector(rates),
        std_deviations=_percent_to_vector(stds),
    )
    assert scalar_ids == _ids_from_mask(mask)
    assert 1 in scalar_ids  # 95 + 5 + 0.84 ≥ 100
    assert 2 not in scalar_ids
    assert 3 in scalar_ids  # 80 + 30 + 0.84 ≥ 100
    assert mask[0, 0].item() is False


def test_service_level_horizon_two_matches_when_both_are_set():
    fills = [70.0, 40.0]
    rates = [10.0, 10.0]
    stds = [0.0, 0.0]
    z = 0.84
    ctx = _scalar_context(fills, threshold=z, rates=rates, stds=stds, horizon_days=2)
    scalar_ids, _ = ServiceLevelSelection().select_bins(ctx)
    mask = ServiceLevelSelector(confidence_factor=z, horizon_days=2).select(
        _percent_to_vector(fills),
        accumulation_rates=_percent_to_vector(rates),
        std_deviations=_percent_to_vector(stds),
    )
    # 70 + 2*10 = 90 < 100; 40 + 2*10 = 60 < 100 → empty
    assert scalar_ids == []
    assert _ids_from_mask(mask) == []


def test_lookahead_seed_and_bundle_match():
    # Paper witness: fills (80, 40, 95), rates (25, 20, 10) → seed {1,3}, bundle {2}.
    fills = [80.0, 40.0, 95.0]
    rates = [25.0, 20.0, 10.0]
    ctx = _scalar_context(fills, rates=rates, current_collection_day=0)
    scalar_ids, _ = LookaheadSelection().select_bins(ctx)
    mask = LookaheadSelector(current_collection_day=0).select(
        _percent_to_vector(fills),
        accumulation_rates=_percent_to_vector(rates),
        current_collection_day=0,
    )
    assert set(scalar_ids) == {1, 2, 3}
    assert set(_ids_from_mask(mask)) == set(scalar_ids)
    assert mask[0, 0].item() is False


def test_lookahead_empty_seed_is_empty_on_both_sides():
    fills = [10.0, 20.0]
    rates = [5.0, 8.0]
    ctx = _scalar_context(fills, rates=rates)
    scalar_ids, _ = LookaheadSelection().select_bins(ctx)
    mask = LookaheadSelector().select(
        _percent_to_vector(fills),
        accumulation_rates=_percent_to_vector(rates),
    )
    assert scalar_ids == []
    assert _ids_from_mask(mask) == []
