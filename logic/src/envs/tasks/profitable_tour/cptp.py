"""Capacitated Profitable Tour Problem (CPTP) environment, generator, and task specification.

In CPTP, the vehicle has finite capacity per trip. Visiting the depot unloads the vehicle
and resets trip capacity. Selecting the depot while already at the depot terminates the day.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from logic.src.envs.tasks.common.metrics import DEFAULT_METRICS
from logic.src.envs.tasks.espprc import LoadResource
from logic.src.envs.tasks.profitable_tour.common import (
    evaluate_profitable_tour,
    profit_objective,
)
from logic.src.envs.tasks.profitable_tour.ptp import PTP, PTPEnv, PTPGenerator
from logic.src.envs.tasks.spec import PROBLEMS, ProblemSpec


class CPTPGenerator(PTPGenerator):
    """Generator for Capacitated Profitable Tour Problem (CPTP) instances."""

    pass


class CPTPEnv(PTPEnv):
    """Capacitated Profitable Tour Problem Environment.

    Attributes:
        NAME: Environment identifier "cptp".
        explicit_stop: True. Selecting the depot while already at the depot ends the day.
    """

    NAME: str = "cptp"
    name: str = "cptp"
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
            generator = CPTPGenerator(**generator_params, device=device)

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
        # Register LoadResource on the ESPPRC base
        self.resources = (LoadResource(),)


class CPTP(PTP):
    """Capacitated Profitable Tour Problem (CPTP) task definition."""

    NAME: str = "cptp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        cw_dict: Optional[Dict[str, float]] = None,
        capacity: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate a completed CPTP tour sequence with vectorized capacity checks."""
        return evaluate_profitable_tour(
            dataset,
            pi,
            dist_matrix=dist_matrix,
            cw_dict=cw_dict,
            capacity=capacity,
            **kwargs,
        )


CPTP_SPEC = ProblemSpec(
    name="cptp",
    objective=profit_objective,
    constraints=(LoadResource(),),
    metrics=DEFAULT_METRICS,
    env_target="logic.src.envs.tasks.profitable_tour.cptp:CPTPEnv",
    expand_pool=True,
)
PROBLEMS.register(CPTP_SPEC)

__all__ = ["CPTPGenerator", "CPTPEnv", "CPTP", "CPTP_SPEC"]
