"""Stochastic Prize-Collecting TSP (SPCTSP) task and environment on ESPPRC base."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from logic.src.envs.tasks.pctsp import PCTSP, PCTSPEnv, PCTSPGenerator
from logic.src.envs.tasks.spec import PROBLEMS, ProblemSpec


class SPCTSPGenerator(PCTSPGenerator):
    """Generator for SPCTSP instances."""

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
        td = super()._generate(batch_size)
        bs = list(batch_size) if isinstance(batch_size, (list, tuple)) else [batch_size]
        stochastic_prize = td["deterministic_prize"] * (0.8 + 0.4 * torch.rand((*bs, self.num_loc), device=self.device))
        td["stochastic_prize"] = stochastic_prize
        td["real_prize"] = stochastic_prize
        return td


class SPCTSPEnv(PCTSPEnv):
    """Stochastic Prize-Collecting TSP Environment on ESPPRC base."""

    NAME: str = "spctsp"
    name: str = "spctsp"

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
