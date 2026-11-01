"""Capacitated Multi-vehicle Profitable Tour Problem (CMPTP) environment, generator, and task specification.

In CMPTP, the vehicle has finite capacity per trip. Visiting the depot unloads the vehicle
and resets trip capacity. Selecting the depot while already at the depot terminates the day.

The module also holds ``CMPTPGenerator``, the problem's instance generator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

from logic.src.envs.base.resources import LoadResource
from logic.src.envs.common.released_specs import CMPTP_SPEC
from logic.src.envs.core.ptp import evaluate_profitable_tour
from logic.src.envs.ptp import PTP, PTPEnv, PTPGenerator
from logic.src.envs.spec import PROBLEMS


class CMPTPGenerator(PTPGenerator):
    """Generator for Capacitated Multi-vehicle Profitable Tour Problem (CMPTP) instances.

    The instance layout is PTP's; the capacity it carries (``capacity``) binds per trip
    in ``CMPTPEnv``.
    """


class CMPTPEnv(PTPEnv):
    """Capacitated Multi-vehicle Profitable Tour Problem Environment.

    Attributes:
        NAME: Environment identifier "cmptp".
        explicit_stop: True. Selecting the depot while already at the depot ends the day.
    """

    NAME: str = "cmptp"
    name: str = "cmptp"
    explicit_stop: bool = True
    mask_empty_bins: bool = False

    def __init__(
        self,
        generator: Optional[PTPGenerator] = None,
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
            generator = CMPTPGenerator(**generator_params, device=device)

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
        # Register LoadResource on the routing base
        self.resources = (LoadResource(),)


class CMPTP(PTP):
    """Capacitated Multi-vehicle Profitable Tour Problem (CMPTP) task definition."""

    NAME: str = "cmptp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        cw_dict: Optional[Dict[str, float]] = None,
        capacity: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate a completed CMPTP tour sequence with vectorized capacity checks."""
        return evaluate_profitable_tour(
            dataset,
            pi,
            dist_matrix=dist_matrix,
            cw_dict=cw_dict,
            capacity=capacity,
            **kwargs,
        )


PROBLEMS.register(CMPTP_SPEC)

__all__ = ["CMPTPGenerator", "CMPTPEnv", "CMPTP", "CMPTP_SPEC"]
