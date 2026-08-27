"""
TTOP problem generator.

Attributes:
    TTOPGenerator: TTOPGenerator class.

Example:
    >>> from logic.src.envs.generators import TTOPGenerator
    >>> generator = TTOPGenerator(num_loc=50)
    >>> instance = generator.generate()
"""

from __future__ import annotations

from typing import Union

import torch
from tensordict import TensorDict

from logic.src.envs.generators.vrpp import VRPPGenerator

# NOT a top-level import: logic.src.pipeline.simulations.repository transitively
# imports logic.src.data, which imports logic.src.data.processor.setup, which
# imports logic.src.pipeline.simulations.repository -- a top-level import here
# closes that cycle during logic.src.envs.generators package init. Deferred to
# call time in __init__ below, matching this module's own pattern in
# VRPPGenerator (see its local `from logic.src.utils.data.loader import
# load_grid_base`).


class TTOPGenerator(VRPPGenerator):
    """
    Generator for Temporal Team Orienteering Problem (TTOP) instances.

    Same node/waste layout as VRPP (reused as-is: locations, depot, waste
    values), plus the per-instance temporal resource fields consumed by
    ``TTOPEnv``/``TTOP``: the working-shift time budget, average driving
    speed, and per-bin service time. Defaults come from
    ``SimulationRepository.get_temporal_params()`` (one driver's 7h shift);
    pass ``shift_hours``/``avg_speed_kmh``/``service_time_h`` to override.

    Attributes:
        shift_hours: Total time budget per trip (h).
        avg_speed_kmh: Average driving speed (km/h).
        service_time_h: Time to visit and empty a single bin (h).
    """

    def __init__(
        self,
        *args,
        shift_hours: Union[float, None] = None,
        avg_speed_kmh: Union[float, None] = None,
        service_time_h: Union[float, None] = None,
        **kwargs,
    ) -> None:
        """
        Initialize TTOP generator.

        Args:
            args: Positional arguments forwarded to VRPPGenerator.
            shift_hours: Override for the per-trip time budget (h).
            avg_speed_kmh: Override for the average driving speed (km/h).
            service_time_h: Override for the per-bin service time (h).
            kwargs: Additional keyword arguments forwarded to VRPPGenerator.
        """
        super().__init__(*args, **kwargs)
        from logic.src.pipeline.simulations.repository.base import SimulationRepository

        default_shift_hours, default_avg_speed_kmh, default_service_time_h = (
            SimulationRepository.get_temporal_params()
        )
        self.shift_hours = shift_hours if shift_hours is not None else default_shift_hours
        self.avg_speed_kmh = avg_speed_kmh if avg_speed_kmh is not None else default_avg_speed_kmh
        self.service_time_h = service_time_h if service_time_h is not None else default_service_time_h

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate TTOP instances: VRPP fields plus temporal resource fields.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with VRPP's fields (locs, depot, waste, capacity,
            max_waste) plus:
                shift_hours: [*B]    — per-trip time budget (h)
                avg_speed_kmh: [*B]  — average driving speed (km/h)
                service_time_h: [*B] — per-bin service time (h)
        """
        td = super()._generate(batch_size)
        td["shift_hours"] = torch.full((*batch_size,), self.shift_hours, device=self.device)
        td["avg_speed_kmh"] = torch.full((*batch_size,), self.avg_speed_kmh, device=self.device)
        td["service_time_h"] = torch.full((*batch_size,), self.service_time_h, device=self.device)
        return td
