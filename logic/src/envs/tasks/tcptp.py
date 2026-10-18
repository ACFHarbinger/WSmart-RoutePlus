"""Time-constrained Capacitated Profitable Tour Problem (TCPTP) environment, generator, and task specification.

In TCPTP, each trip is bounded by two simultaneous constraints: vehicle capacity
and working shift time budget (driving time + customer service time).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base.resources import LoadResource, ShiftTimeResource
from logic.src.envs.generators.tcptp import TCPTPGenerator
from logic.src.envs.tasks.common.released_specs import TCPTP_SPEC
from logic.src.envs.tasks.core.ptp import evaluate_profitable_tour
from logic.src.envs.tasks.cptp import CPTP, CPTPEnv
from logic.src.envs.tasks.spec import PROBLEMS


class TCPTPEnv(CPTPEnv):
    """Time-constrained Capacitated Profitable Tour Problem Environment.

    Attributes:
        NAME: Environment identifier "tcptp".
        explicit_stop: False. Ends on exhaustion.
    """

    NAME: str = "tcptp"
    name: str = "tcptp"
    explicit_stop: bool = False

    def __init__(
        self,
        generator: Optional[TCPTPGenerator] = None,
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
            generator = TCPTPGenerator(**generator_params, device=device)

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
        """End at the depot once no customer is reachable even on a fresh trip.

        A depot return starts a new trip, so it ends a multi-trip TCPTP day only on
        exhaustion (the pre-WP-07 ``envs/routing/tcptp.py`` rule).
        """
        current = tensordict["current_node"].reshape(-1)
        step = tensordict["i"].reshape(-1)
        exhausted = ~self._get_action_mask(tensordict)[:, 1:].any(dim=1)
        return (current == 0) & (step > 0) & exhausted


class TCPTP(CPTP):
    """Time-constrained Capacitated Profitable Tour Problem (TCPTP) task definition."""

    NAME: str = "tcptp"

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
        """Evaluate a completed TCPTP tour sequence with capacity and time budget checks."""
        return evaluate_profitable_tour(
            dataset,
            pi,
            dist_matrix=dist_matrix,
            cw_dict=cw_dict,
            capacity=capacity,
            shift_hours=shift_hours,
            **kwargs,
        )


PROBLEMS.register(TCPTP_SPEC)

__all__ = ["TCPTPGenerator", "TCPTPEnv", "TCPTP", "TCPTP_SPEC"]
