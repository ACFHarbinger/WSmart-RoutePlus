"""Select each instance's day from batched temporal data."""

from typing import Optional

import torch


def select_instance_day(values: torch.Tensor, current_day: Optional[torch.Tensor]) -> torch.Tensor:
    """Slice ``[batch, days, ...]`` by per-instance days, clamped to the series.

    Missing days select day zero. A single day value broadcasts across the batch.
    """
    batch_size, n_days = values.shape[:2]
    if current_day is None:
        day = torch.zeros(batch_size, dtype=torch.long, device=values.device)
    else:
        day = current_day.to(device=values.device, dtype=torch.long).reshape(-1).expand(batch_size)
    day = day.clamp(min=0, max=n_days - 1)
    return values[torch.arange(batch_size, device=values.device), day]
