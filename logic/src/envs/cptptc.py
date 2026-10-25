"""Capacitated Profitable Tour Problem with Time Constraints (CPTPTC) environment, generator, and task specification.

In CPTPTC, each trip is bounded by two simultaneous constraints: vehicle capacity
and working shift time budget (driving time + customer service time).

The module also holds ``CPTPTCGenerator``, the problem's instance generator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from logic.src.constants.temporal import get_default_temporal_params
from logic.src.envs.base.resources import LoadResource, ShiftTimeResource
from logic.src.envs.common.released_specs import CPTPTC_SPEC
from logic.src.envs.core.ptp import evaluate_profitable_tour
from logic.src.envs.cptp import CPTP, CPTPEnv
from logic.src.envs.ptp import PTPGenerator
from logic.src.envs.spec import PROBLEMS


class CPTPTCGenerator(PTPGenerator):
    """
    Generator for Capacitated Profitable Tour Problem with Time Constraints (CPTPTC) instances.

    Same node/waste layout as PTP (reused as-is: locations, depot, waste
    values), plus the per-instance temporal resource fields consumed by
    ``CPTPTCEnv``/``CPTPTC``: the working-shift time budget, average driving
    speed, and per-bin service time. Defaults describe one driver's 7h
    shift; pass ``shift_hours``/``avg_speed_kmh``/``service_time_h`` to
    override them.

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
        Initialize CPTPTC generator.

        Args:
            args: Positional arguments forwarded to PTPGenerator.
            shift_hours: Override for the per-trip time budget (h).
            avg_speed_kmh: Override for the average driving speed (km/h).
            service_time_h: Override for the per-bin service time (h).
            kwargs: Additional keyword arguments forwarded to PTPGenerator.
        """
        super().__init__(*args, **kwargs)
        default_shift_hours, default_avg_speed_kmh, default_service_time_h = get_default_temporal_params()
        self.shift_hours = shift_hours if shift_hours is not None else default_shift_hours
        self.avg_speed_kmh = avg_speed_kmh if avg_speed_kmh is not None else default_avg_speed_kmh
        self.service_time_h = service_time_h if service_time_h is not None else default_service_time_h

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
        """Generate CPTPTC instances: PTP fields plus temporal resource fields.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with PTP's fields (locs, depot, waste, capacity,
            max_waste) plus:
                shift_hours: [*B]    — per-trip time budget (h)
                avg_speed_kmh: [*B]  — average driving speed (km/h)
                service_time_h: [*B] — per-bin service time (h)
        """
        td = super()._generate(batch_size)
        bs = tuple(td.batch_size)
        td["shift_hours"] = torch.full(bs, self.shift_hours, device=self.device)
        td["avg_speed_kmh"] = torch.full(bs, self.avg_speed_kmh, device=self.device)
        td["service_time_h"] = torch.full(bs, self.service_time_h, device=self.device)
        return td


class CPTPTCEnv(CPTPEnv):
    """Capacitated Profitable Tour Problem with Time Constraints Environment.

    Attributes:
        NAME: Environment identifier "cptptc".
        explicit_stop: False. Ends on exhaustion.
    """

    NAME: str = "cptptc"
    name: str = "cptptc"
    explicit_stop: bool = True

    def __init__(
        self,
        generator: Optional[CPTPTCGenerator] = None,
        generator_params: Optional[dict] = None,
        waste_weight: float = 1.0,
        cost_weight: float = 1.0,
        revenue_kg: Optional[float] = None,
        cost_km: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = CPTPTCGenerator(**generator_params, device=device)

        super().__init__(
            generator=generator,
            generator_params=generator_params,
            waste_weight=waste_weight,
            cost_weight=cost_weight,
            revenue_kg=revenue_kg,
            cost_km=cost_km,
            device=device,
            **kwargs,
        )
        self.resources = (LoadResource(), ShiftTimeResource())

    def _check_done(self, tensordict: TensorDict) -> torch.Tensor:
        """End at the depot once explicit stop triggers or customer actions are exhausted."""
        current = tensordict["current_node"].reshape(-1)
        step = tensordict["i"].reshape(-1)
        exhausted = (current == 0) & (step > 0) & ~self._get_action_mask(tensordict)[:, 1:].any(dim=1)
        finished = tensordict.get("finished", torch.zeros_like(current, dtype=torch.bool))
        return finished | exhausted


class CPTPTC(CPTP):
    """Capacitated Profitable Tour Problem with Time Constraints (CPTPTC) task definition."""

    NAME: str = "cptptc"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        cw_dict: Optional[Dict[str, float]] = None,
        capacity: Optional[float] = None,
        shift_hours: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate a completed CPTPTC tour sequence with capacity and time budget checks."""
        return evaluate_profitable_tour(
            dataset,
            pi,
            dist_matrix=dist_matrix,
            cw_dict=cw_dict,
            capacity=capacity,
            shift_hours=shift_hours,
            **kwargs,
        )


PROBLEMS.register(CPTPTC_SPEC)

__all__ = ["CPTPTCGenerator", "CPTPTCEnv", "CPTPTC", "CPTPTC_SPEC"]
