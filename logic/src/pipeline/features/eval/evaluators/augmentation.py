"""Augmentation evaluation implementation.

Attributes:
    AugmentationEval: Evaluates a policy using Test-Time Augmentation.

Example:
    >>> from logic.src.pipeline.features.eval.evaluators import AugmentationEval
    >>> evaluator = AugmentationEval(config)
    >>> evaluator.evaluate()
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from logic.src.data.processor.transforms import StateAugmentation
from logic.src.pipeline.features.eval.eval_base import EvalBase
from logic.src.utils.functions import move_to


class AugmentationEval(EvalBase):
    """Evaluation with data augmentation (Test-Time Augmentation).
    Performs N augmentations and picks the best result for each sample.

    Attributes:
        env: The environment to evaluate on.
        num_augment: Number of augmentations to perform.
        augment_fn: Name of the augmentation function.
        progress: Whether to show progress bar.
    """

    def __init__(
        self,
        env: Any,
        num_augment: int = 8,
        augment_fn: str = "dihedral8",
        progress: bool = True,
        **kwargs,
    ):
        """Initialize Class.

        Args:
            env (Any): Description of env.
            num_augment (int): Description of num_augment.
            augment_fn (str): Description of augment_fn.
            progress (bool): Description of progress.
            kwargs (Any): Description of kwargs.
        """
        super().__init__(env, progress, **kwargs)
        self.samples = num_augment
        self.augmentation = StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)

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

        for batch in tqdm(data_loader, disable=not self.progress, desc=f"Augmentation Eval ({self.samples})"):
            batch = move_to(batch, self.device)  # type: ignore[arg-type]
            with torch.inference_mode():
                # Apply augmentation: returns [batch_size * num_augment, ...]
                aug_batch = self.augmentation(batch)
                out = policy(aug_batch, strategy="greedy", **kwargs)

                # Reshape and pick best from augmentations
                reward = out["reward"].view(self.samples, batch.batch_size[0], -1).max(0).values
                out["reward"] = reward
                results.append(out)

        total_time = time.time() - start_time
        avg_reward = torch.cat([r["reward"] for r in results]).mean().item()

        metrics = {
            "avg_reward": avg_reward,
            "total_time": total_time,
            "samples_per_second": len(data_loader.dataset) / total_time,  # type: ignore[arg-type]
        }

        if return_results:
            metrics["results"] = results  # type: ignore[assignment]
        return metrics
