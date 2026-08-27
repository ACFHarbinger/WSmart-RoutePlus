"""Greedy evaluation implementation.

Attributes:
    GreedyEval: Evaluates a policy using greedy search.

Example:
    >>> from logic.src.pipeline.features.eval.evaluators import GreedyEval
    >>> evaluator = GreedyEval(config)
    >>> evaluator.evaluate()
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from logic.src.pipeline.features.eval.eval_base import EvalBase
from logic.src.utils.functions import move_to


class GreedyEval(EvalBase):
    """Greedy evaluation.

    Attributes:
        None
    """

    def __call__(self, policy: Any, data_loader: DataLoader, return_results: bool = False, **kwargs) -> dict:
        """call  .

        Args:
            policy (Any): Description of policy.
            data_loader (DataLoader): Description of data_loader.
            return_results (bool): Description of return_results.
            kwargs (Any): Description of kwargs.

        Returns:
            Any: Description of return value.
        """
        policy.eval()
        results = []
        start_time = time.time()

        for batch in tqdm(data_loader, disable=not self.progress, desc="Greedy Eval"):
            batch = move_to(batch, self.device)  # type: ignore[arg-type]
            with torch.inference_mode():
                out = policy(batch, strategy="greedy", **kwargs)
                results.append(out)

        total_time = time.time() - start_time
        avg_reward = torch.cat([r["reward"] for r in results]).mean().item()

        metrics = {
            "avg_reward": avg_reward,
            "duration": total_time,
            "samples_per_second": len(data_loader.dataset) / total_time,  # type: ignore[arg-type]
        }

        if return_results:
            # Assuming output has 'reward' and 'actions' (or 'sequences'?)
            # AttentionModel output usually has 'reward', 'log_likelihood', 'actions'
            metrics["rewards"] = torch.cat([r["reward"] for r in results]).cpu()  # type: ignore[assignment]

            # Pad sequences to max length
            max_len = max([r["actions"].size(-1) for r in results])
            padded_actions = []
            for r in results:
                act = r["actions"]
                pad_len = max_len - act.size(-1)
                if pad_len > 0:
                    act = torch.nn.functional.pad(act, (0, pad_len), value=0)
                padded_actions.append(act)
            metrics["sequences"] = torch.cat(padded_actions).cpu()  # type: ignore[assignment]
            metrics["results"] = results  # type: ignore[assignment]
        return metrics
