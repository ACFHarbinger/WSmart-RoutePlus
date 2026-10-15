"""Stochastic Prize-Collecting TSP (SPCTSP) task and environment on ESPPRC base."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

from logic.src.envs.tasks.pctsp import PCTSP, PCTSPEnv, PCTSPGenerator
from logic.src.envs.tasks.spec import PROBLEMS, ProblemSpec

# SPCTSP draws on the PCTSP generator, whose instances carry the stochastic prize.
SPCTSPGenerator = PCTSPGenerator


class SPCTSPEnv(PCTSPEnv):
    """Stochastic Prize-Collecting TSP Environment on ESPPRC base."""

    NAME: str = "spctsp"
    name: str = "spctsp"
    # The prize received is the stochastic one (the pre-WP-07 envs/routing/spctsp.py).
    _stochastic: bool = True

    def __init__(
        self,
        generator: Optional[SPCTSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = SPCTSPGenerator(**generator_params, device=device)
        super().__init__(generator=generator, generator_params=generator_params, device=device, **kwargs)


class SPCTSP(PCTSP):
    """Stochastic Prize-Collecting TSP (SPCTSP) task definition."""

    NAME = "spctsp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        return PCTSP.evaluate_tour(dataset, pi, dist_matrix=dist_matrix, **kwargs)


SPCTSP_SPEC = ProblemSpec(
    name="spctsp",
    objective=lambda neg_reward, **kw: -float(neg_reward),
    constraints=(),
    metrics=("cost", "reward", "saved_penalty", "length"),
    env_target="logic.src.envs.tasks.spctsp:SPCTSPEnv",
    expand_pool=True,
)
PROBLEMS.register(SPCTSP_SPEC)

__all__ = ["SPCTSPGenerator", "SPCTSPEnv", "SPCTSP", "SPCTSP_SPEC"]
