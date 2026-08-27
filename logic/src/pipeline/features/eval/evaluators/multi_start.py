"""Multi-start evaluation implementation.

Attributes:
    MultiStartEval: Evaluates a policy using multi-start search.

Example:
    >>> from logic.src.pipeline.features.eval.evaluators import MultiStartEval
    >>> evaluator = MultiStartEval(config)
    >>> evaluator.evaluate()
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from logic.src.pipeline.features.eval.eval_base import EvalBase
from logic.src.utils.functions import move_to


class MultiStartEval(EvalBase):
    """Evaluation with multiple starts (POMO-style).

    Attributes:
        env: The environment to evaluate on.
        num_starts: Number of starts for multi-start search.
        progress: Whether to show progress bar.
    """

    def __init__(self, env: Any, num_starts: Optional[int] = None, progress: bool = True, **kwargs):
        """Initialize Class.

        Args:
            env (Any): Description of env.
            num_starts (Optional[int]): Description of num_starts.
            progress (bool): Description of progress.
            kwargs (Any): Description of kwargs.
        """
        super().__init__(env, progress, **kwargs)
        self.num_starts = num_starts

    def __call__(self, policy: Any, data_loader: DataLoader, **kwargs) -> dict:  # type: ignore[override]
        """call  .

        Args:
            policy (Any): Description of policy.
            data_loader (DataLoader): Description of data_loader.
            kwargs (Any): Description of kwargs.

        Returns:
            Any: Description of return value.
        """
        policy.eval()
        results = []
        start_time = time.time()

        for batch in tqdm(data_loader, disable=not self.progress, desc="Multi-Start Eval"):
            batch = move_to(batch, self.device)  # type: ignore[arg-type]
            with torch.inference_mode():
                out = policy(batch, strategy="greedy", num_starts=self.num_starts, **kwargs)
                results.append(out)

        total_time = time.time() - start_time
        avg_reward = torch.cat([r["reward"] for r in results]).mean().item()

        return {
            "avg_reward": avg_reward,
            "total_time": total_time,
            "samples_per_second": len(data_loader.dataset) / total_time,  # type: ignore[arg-type]
        }
