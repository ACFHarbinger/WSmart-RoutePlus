"""Per-node service time windows within a working day (CPTPTWTC).

Unlike :mod:`logic.src.data.time.time_windows`, which says on which simulation days routing is
allowed, these windows say when, within one day, each bin can be served: ``[tw_early, tw_late]``
in hours from the start of the day. The depot is open all day.

Attributes:
    DEFAULT_DAY_HOURS: Default working-day length (h).
    DEFAULT_TW_MIN_WIDTH_H: Default shortest bin window (h).
    DEFAULT_TW_MAX_WIDTH_H: Default longest bin window (h).
    sample_node_time_windows: Draw a window per bin.

Example:
    >>> early, late = sample_node_time_windows((4, 20))
    >>> bool((late - early >= DEFAULT_TW_MIN_WIDTH_H - 1e-6).all())
    True
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch

DEFAULT_DAY_HOURS = 10.0
DEFAULT_TW_MIN_WIDTH_H = 2.0
DEFAULT_TW_MAX_WIDTH_H = 6.0


def sample_node_time_windows(
    shape: Sequence[int],
    day_hours: float = DEFAULT_DAY_HOURS,
    min_width_h: float = DEFAULT_TW_MIN_WIDTH_H,
    max_width_h: float = DEFAULT_TW_MAX_WIDTH_H,
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw a time window per bin: width uniform in ``[min_width_h, max_width_h]``, placed uniformly in the day.

    Args:
        shape: Window tensor shape, e.g. ``(batch, num_bins)``.
        day_hours: Working-day length (h).
        min_width_h: Shortest window (h).
        max_width_h: Longest window (h).
        generator: Optional random generator (seeded datasets).
        device: Device of the returned tensors.

    Returns:
        ``(tw_early, tw_late)``, each of ``shape``, in hours from the start of the day.

    Raises:
        ValueError: When ``0 < min_width_h <= max_width_h <= day_hours`` does not hold.
    """
    if not 0 < min_width_h <= max_width_h <= day_hours:
        raise ValueError("time windows need 0 < tw_min_width_h <= tw_max_width_h <= day_hours")
    shape = tuple(shape)
    width = min_width_h + (max_width_h - min_width_h) * torch.rand(shape, generator=generator, device=device)
    early = (day_hours - width) * torch.rand(shape, generator=generator, device=device)
    return early, early + width


__all__ = ["DEFAULT_DAY_HOURS", "DEFAULT_TW_MAX_WIDTH_H", "DEFAULT_TW_MIN_WIDTH_H", "sample_node_time_windows"]
