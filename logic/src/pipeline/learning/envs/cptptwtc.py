"""Capacitated Profitable Tour Problem with Time Windows and Time Constraints (CPTPTWTC).

CPTPTWTC extends CPTPTC: every trip is still bounded by the vehicle capacity and by the
working-shift time budget (driving + service time), and every bin also has a time window
``[tw_early, tw_late]`` in hours from the start of the day. A vehicle that arrives early
waits until the window opens; service must start no later than the window closes; and the
vehicle must be back at the depot before the depot's window (the working day) closes. The
day clock runs across trips: a depot return resets the trip's shift budget, not the clock.

The module also holds ``CPTPTWTCGenerator``, the problem's instance generator, and
``CPTPTWTC_SPEC``, its problem specification.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from logic.src.constants.temporal import get_default_temporal_params
from logic.src.data.time.node_time_windows import (
    DEFAULT_DAY_HOURS,
    DEFAULT_TW_MAX_WIDTH_H,
    DEFAULT_TW_MIN_WIDTH_H,
    sample_node_time_windows,
)
from logic.src.pipeline.learning.envs.base.resources import LoadResource, ShiftTimeResource, TimeWindowResource
from logic.src.pipeline.learning.envs.common.metrics import DEFAULT_METRICS
from logic.src.pipeline.learning.envs.common.objectives import profit_objective
from logic.src.pipeline.learning.envs.core.ptp import evaluate_profitable_tour
from logic.src.pipeline.learning.envs.cptptc import CPTPTC, CPTPTCEnv, CPTPTCGenerator
from logic.src.pipeline.learning.envs.spec import PROBLEMS, ProblemSpec


class CPTPTWTCGenerator(CPTPTCGenerator):
    """Generator for CPTPTWTC instances: CPTPTC fields plus per-bin time windows.

    Attributes:
        day_hours: Length of the working day (h); the depot is open for all of it.
        tw_min_width_h: Shortest bin time window (h).
        tw_max_width_h: Longest bin time window (h).
    """

    def __init__(
        self,
        *args,
        day_hours: Optional[float] = None,
        tw_min_width_h: Optional[float] = None,
        tw_max_width_h: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Initialize the CPTPTWTC generator.

        Args:
            args: Positional arguments forwarded to CPTPTCGenerator.
            day_hours: Override for the working-day length (h).
            tw_min_width_h: Override for the shortest bin window (h).
            tw_max_width_h: Override for the longest bin window (h).
            kwargs: Keyword arguments forwarded to CPTPTCGenerator (shift_hours, avg_speed_kmh, ...).
        """
        super().__init__(*args, **kwargs)
        self.day_hours = float(day_hours if day_hours is not None else DEFAULT_DAY_HOURS)
        self.tw_min_width_h = float(tw_min_width_h if tw_min_width_h is not None else DEFAULT_TW_MIN_WIDTH_H)
        self.tw_max_width_h = float(tw_max_width_h if tw_max_width_h is not None else DEFAULT_TW_MAX_WIDTH_H)
        sample_node_time_windows((1, 1), self.day_hours, self.tw_min_width_h, self.tw_max_width_h)  # validates

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
        """Generate CPTPTWTC instances.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with CPTPTC's fields plus:
                day_hours: [*B]   — working-day length (h); the depot window is [0, day_hours]
                tw_early: [*B, N] — opening time of each bin's window (h)
                tw_late: [*B, N]  — closing time of each bin's window (h)
        """
        td = super()._generate(batch_size)
        bs = tuple(td.batch_size)
        n = td["locs"].shape[-2]
        early, late = sample_node_time_windows(
            (*bs, n), self.day_hours, self.tw_min_width_h, self.tw_max_width_h, device=self.device
        )
        td["day_hours"] = torch.full(bs, self.day_hours, device=self.device)
        td["tw_early"] = early
        td["tw_late"] = late
        return td


class CPTPTWTCEnv(CPTPTCEnv):
    """CPTPTWTC environment: CPTPTC's capacity and shift budget plus bin time windows.

    Attributes:
        NAME: Environment identifier "cptptwtc".
    """

    NAME: str = "cptptwtc"
    name: str = "cptptwtc"

    def __init__(
        self,
        generator: Optional[CPTPTWTCGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or {k: v for k, v in kwargs.items() if k not in ("waste_weight", "cost_weight", "revenue_kg", "cost_km")}
        if generator is None:
            generator = CPTPTWTCGenerator(**generator_params, device=device)
        super().__init__(generator=generator, generator_params=generator_params, device=device, **kwargs)
        self.resources = (LoadResource(), ShiftTimeResource(), TimeWindowResource(default_day_hours=DEFAULT_DAY_HOURS))


def time_window_violations(
    dataset: Dict[str, Any],
    pi: torch.Tensor,
    dist_matrix: Optional[torch.Tensor] = None,
    avg_speed_kmh: Optional[float] = None,
    service_time_h: Optional[float] = None,
) -> torch.Tensor:
    """Count, per instance, the visits of ``pi`` that start after their window closes.

    Replays the tour on the day clock with the shift resource's travel-time model (road
    distance or Euclidean distance over ``avg_speed_kmh``, or ``time_matrix`` when present):
    arriving early waits until ``tw_early``; a visit is late when it starts after
    ``tw_late``; the final arrival at the depot is late after ``day_hours``.

    Args:
        dataset: Instances with locs, depot, tw_early, tw_late and day_hours.
        pi: Tours [B, L] (node 0 is the depot).
        dist_matrix: Optional distance matrix [B, N+1, N+1] or [N+1, N+1].
        avg_speed_kmh: Optional speed override.
        service_time_h: Optional per-bin service time override.

    Returns:
        Late visits per instance, shape [B].
    """
    _, def_speed, def_service = get_default_temporal_params()
    pi = pi if pi.dim() == 2 else pi.unsqueeze(0)
    locs = dataset["locs"]
    if locs.dim() == 2:
        locs = locs.unsqueeze(0)
    depot = dataset.get("depot") if hasattr(dataset, "get") else None
    # Raw instances hold the bins only; a reset state already starts with the depot (as RoutingEnv does).
    if depot is not None and not torch.allclose(locs[..., 0, :], depot.reshape(locs[..., 0, :].shape), atol=1e-4):
        locs = torch.cat([depot.reshape(locs[..., 0, :].shape).unsqueeze(-2), locs], dim=-2)
    bsz, device = pi.size(0), pi.device
    speed = torch.as_tensor(avg_speed_kmh if avg_speed_kmh is not None else dataset.get("avg_speed_kmh", def_speed),
                            dtype=torch.float32, device=device).expand(bsz)
    service = torch.as_tensor(service_time_h if service_time_h is not None else dataset.get("service_time_h", def_service),
                              dtype=torch.float32, device=device).expand(bsz)
    day = torch.as_tensor(dataset.get("day_hours", DEFAULT_DAY_HOURS), dtype=torch.float32, device=device).expand(bsz)
    early = dataset["tw_early"].to(device)
    late = dataset["tw_late"].to(device)
    if early.shape[-1] == locs.shape[-2] - 1:  # prepend the depot's window
        early = torch.cat([torch.zeros(bsz, 1, device=device), early], dim=-1)
        late = torch.cat([day.unsqueeze(-1), late], dim=-1)
    time_matrix = dataset.get("time_matrix")
    if dist_matrix is not None and dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(bsz, -1, -1)
    rows = torch.arange(bsz, device=device)
    clock = torch.zeros(bsz, device=device)
    violations = torch.zeros(bsz, dtype=torch.long, device=device)
    prev = torch.zeros(bsz, dtype=torch.long, device=device)
    # Append a final depot return so the last trip is checked against the day's end.
    tour = torch.cat([pi.to(device).long(), torch.zeros(bsz, 1, dtype=torch.long, device=device)], dim=1)
    done = torch.zeros(bsz, dtype=torch.bool, device=device)
    for step in range(tour.size(1)):
        node = tour[:, step]
        if time_matrix is not None:
            travel = time_matrix[rows, prev, node]
        elif dist_matrix is not None:
            travel = dist_matrix[rows, prev, node] / speed
        else:
            travel = (locs[rows, node] - locs[rows, prev]).norm(dim=-1) / speed
        moving = ~done & (node != prev)
        arrival = clock + travel
        start = torch.maximum(arrival, early[rows, node])
        violations += (moving & (start > late[rows, node] + 1e-6)).long()
        is_bin = node != 0
        clock = torch.where(moving, start + torch.where(is_bin, service, torch.zeros_like(service)), clock)
        # Trailing depot padding (repeated zeros) ends the tour.
        done = done | ((node == 0) & (prev == 0) & (step > 0))
        prev = torch.where(moving, node, prev)
    return violations


class CPTPTWTC(CPTPTC):
    """Capacitated Profitable Tour Problem with Time Windows and Time Constraints task definition."""

    NAME: str = "cptptwtc"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        cw_dict: Optional[Dict[str, float]] = None,
        capacity: Optional[float] = None,
        shift_hours: Optional[float] = None,
        check_feasibility: bool = False,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate a CPTPTWTC tour: CPTPTC's capacity and shift checks plus the time windows.

        Returns:
            CPTPTC's outcome dict plus ``tw_violations`` (late visits per instance, [B]).

        Raises:
            AssertionError: With ``check_feasibility``, when a visit starts after its window closes.
        """
        out = evaluate_profitable_tour(
            dataset,
            pi,
            dist_matrix=dist_matrix,
            cw_dict=cw_dict,
            capacity=capacity,
            shift_hours=shift_hours,
            check_feasibility=check_feasibility,
            **kwargs,
        )
        late = time_window_violations(
            dataset, pi, dist_matrix=dist_matrix,
            avg_speed_kmh=kwargs.get("avg_speed_kmh"), service_time_h=kwargs.get("service_time_h"),
        )
        if check_feasibility and bool((late > 0).any()):
            raise AssertionError(f"CPTPTWTC: {int(late.sum())} visit(s) start after their time window closes")
        out["tw_violations"] = late
        return out


CPTPTWTC_SPEC = ProblemSpec(
    name="cptptwtc",
    objective=profit_objective,
    metrics=DEFAULT_METRICS + ("time_spent",),
    env_target="logic.src.pipeline.learning.envs.cptptwtc:CPTPTWTCEnv",
    expand_pool=True,
    per_trip_shift=True,
)

PROBLEMS.register(CPTPTWTC_SPEC)

__all__ = ["CPTPTWTCGenerator", "CPTPTWTCEnv", "CPTPTWTC", "CPTPTWTC_SPEC", "time_window_violations"]
