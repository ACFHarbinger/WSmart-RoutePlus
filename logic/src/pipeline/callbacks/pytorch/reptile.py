"""Reptile meta-learning callback.

Implements the Reptile algorithm (Nichol et al. 2018) for cross-size and
cross-distribution generalisation in combinatorial optimisation, following
Manchanda et al. 2022 and Zhou et al. 2023.

Attributes:
    ReptileCallback: Callback that performs inner-loop task training and outer-loop meta-updates.

Example:
    >>> from logic.src.pipeline.callbacks.pytorch.reptile import ReptileCallback
    >>> cb = ReptileCallback(num_tasks=4, alpha=0.3, alpha_decay=0.99,
    ...                      min_size=20, max_size=100, data_type="size")
"""

from __future__ import annotations

import copy
import math
import random
from typing import TYPE_CHECKING

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import Callback
from torch.optim import Adam

if TYPE_CHECKING:
    from logic.src.pipeline.learning.common.base.module import RL4COLitModule


def _as_rl_module(pl_module: L.LightningModule) -> "RL4COLitModule":
    """Narrow a Lightning module to ``RL4COLitModule`` (it owns the env and generator used here).

    Args:
        pl_module: The module passed to the callback hook.

    Returns:
        RL4COLitModule: The same object, checked at run time.
    """
    from logic.src.pipeline.learning.common.base.module import RL4COLitModule

    assert isinstance(pl_module, RL4COLitModule)
    return pl_module


class ReptileCallback(Callback):
    """
    Reptile meta-learning callback for cross-size and cross-distribution generalization.

    Implements the Reptile algorithm (Nichol et al. 2018) adapted for combinatorial
    optimization, following Manchanda et al. 2022 and Zhou et al. 2023.

    The callback manages:
    - Task sampling from a configurable task set (size, distribution, or both).
    - Inner-loop optimization on sampled tasks.
    - Outer-loop meta-update by averaging task model deltas.
    - Learning rate scheduling with alpha decay.

    Args:
        num_tasks: Number of tasks per meta-batch (inner-loop iterations).
        alpha: Initial weight for the outer-loop Reptile update.
        alpha_decay: Multiplicative decay applied to alpha each meta-epoch.
        min_size: Minimum problem size for task sampling.
        max_size: Maximum problem size for task sampling.
        sch_bar: Fraction of total epochs after which LR decays by 0.1.
        data_type: Task type -- "size", "distribution", or "size_distribution".
        print_log: Whether to print sampled task info during training.

    Attributes:
        num_tasks: Number of tasks per meta-batch.
        alpha: Current outer-loop learning rate.
        alpha_decay: Multiplicative decay applied to alpha each meta-epoch.
        sch_bar: Fraction of total epochs after which LR decays by 0.1.
        print_log: Whether to print sampled task info during training.
        data_type: Task type — "size", "distribution", or "size_distribution".
        task_set: Full list of candidate tasks generated from the data_type.
        meta_model_state_dict: State dict snapshot taken at the start of each meta-batch.
        task_models: Collected task model state dicts within the current meta-batch.
        selected_tasks: Tasks sampled for the current meta-batch.
        task_params: Active task parameters used to configure the generator.
    """

    def __init__(
        self,
        num_tasks: int,
        alpha: float,
        alpha_decay: float,
        min_size: int,
        max_size: int,
        sch_bar: float = 0.9,
        data_type: str = "size",
        print_log: bool = True,
    ):
        """Initialize ReptileCallback.

        Args:
            num_tasks: Number of tasks per meta-batch (inner-loop iterations).
            alpha: Initial weight for the outer-loop Reptile update.
            alpha_decay: Multiplicative decay applied to alpha each meta-epoch.
            min_size: Minimum problem size for task sampling.
            max_size: Maximum problem size for task sampling.
            sch_bar: Fraction of total epochs after which LR decays by 0.1.
            data_type: Task type — "size", "distribution", or "size_distribution".
            print_log: Whether to print sampled task info during training.
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.sch_bar = sch_bar
        self.print_log = print_log
        self.data_type = data_type
        self.task_set = self._generate_task_set(data_type, min_size, max_size)
        self.meta_model_state_dict: dict = {}
        self.task_models: list = []
        self.selected_tasks: list = []
        self.task_params: tuple = ()

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Sample initial task batch and configure the first task.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module instance.
        """
        self._sample_task()

        # Cast to RL4COLitModule to access env and generator
        module = _as_rl_module(pl_module)
        if not hasattr(module, "env") or module.env.generator is None:
            return

        if self.data_type == "size_distribution":
            module.env.generator.loc_distribution = "gaussian_mixture"
            self.selected_tasks[0] = (module.env.generator.num_loc, 0, 0)
        elif self.data_type == "size":
            module.env.generator.loc_distribution = "uniform"
            self.selected_tasks[0] = (module.env.generator.num_loc,)
        elif self.data_type == "distribution":
            module.env.generator.loc_distribution = "gaussian_mixture"
            self.selected_tasks[0] = (0, 0)
        self.task_params = self.selected_tasks[0]

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save meta-model at start of each meta-batch, reset optimizer each epoch.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module instance.
        """
        self._alpha_scheduler()

        if trainer.current_epoch % self.num_tasks == 0:
            self.meta_model_state_dict = copy.deepcopy(pl_module.state_dict())
            self.task_models = []
            if self.print_log:
                print(
                    f"\n>> Meta epoch: {trainer.current_epoch // self.num_tasks} "
                    f"(Exact epoch: {trainer.current_epoch}), "
                    f"Training task: {self.selected_tasks}"
                )
        else:
            pl_module.load_state_dict(self.meta_model_state_dict)

        # Reset optimizer each epoch with optional LR decay
        max_epochs = trainer.max_epochs if trainer.max_epochs is not None else 100
        lr_decay = 0.1 if trainer.current_epoch + 1 == int(self.sch_bar * max_epochs) else 1.0
        old_lr = trainer.optimizers[0].param_groups[0]["lr"]
        new_optimizer = Adam(pl_module.parameters(), lr=old_lr * lr_decay)
        trainer.optimizers = [new_optimizer]

        module = _as_rl_module(pl_module)
        if self.print_log and hasattr(module, "env") and module.env.generator is not None:
            if hasattr(module.env.generator, "capacity"):
                print(f">> Training task: {self.task_params}, capacity: {module.env.generator.capacity}")
            else:
                print(f">> Training task: {self.task_params}")

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save task model and perform outer-loop update at end of meta-batch.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module instance.
        """
        self.task_models.append(copy.deepcopy(pl_module.state_dict()))

        if (trainer.current_epoch + 1) % self.num_tasks == 0:
            # Outer-loop Reptile update
            with torch.no_grad():
                state_dict = {
                    key: (
                        self.meta_model_state_dict[key]
                        + self.alpha
                        * torch.mean(
                            torch.stack(
                                [task_w[key] - self.meta_model_state_dict[key] for task_w in self.task_models],
                                dim=0,
                            ).float(),
                            dim=0,
                        )
                    )
                    for key in self.meta_model_state_dict
                }
                pl_module.load_state_dict(state_dict)

        # Sample new tasks at end of meta-batch
        if (trainer.current_epoch + 1) % self.num_tasks == 0:
            self._sample_task()

        # Load next task
        self._load_task(pl_module, task_idx=(trainer.current_epoch + 1) % self.num_tasks)

    def _sample_task(self) -> None:
        """Sample a batch of tasks from the task set."""
        self.selected_tasks = [random.sample(self.task_set, 1)[0] for _ in range(self.num_tasks)]

    def _load_task(self, pl_module: L.LightningModule, task_idx: int = 0) -> None:
        """Load a task by updating the environment generator parameters.

        Args:
            pl_module: The PyTorch Lightning module instance.
            task_idx: The index of the task to load.
        """
        self.task_params = self.selected_tasks[task_idx]
        module = _as_rl_module(pl_module)

        if not hasattr(module, "env") or module.env.generator is None:
            return

        gen = module.env.generator
        if self.data_type == "size_distribution":
            assert len(self.task_params) == 3
            gen.num_loc = self.task_params[0]
            gen.num_modes = self.task_params[1]
            gen.cdist = self.task_params[2]
        elif self.data_type == "distribution":
            assert len(self.task_params) == 2
            gen.num_modes = self.task_params[0]
            gen.cdist = self.task_params[1]
        elif self.data_type == "size":
            assert len(self.task_params) == 1
            gen.num_loc = self.task_params[0]

        if hasattr(gen, "capacity") and self.data_type in [
            "size_distribution",
            "size",
        ]:
            task_capacity = math.ceil(30 + self.task_params[0] / 5) if self.task_params[0] >= 20 else 20
            gen.capacity = task_capacity

    def _alpha_scheduler(self) -> None:
        """Decay the outer-loop learning rate."""
        self.alpha = max(self.alpha * self.alpha_decay, 0.0001)

    @staticmethod
    def _generate_task_set(data_type: str, min_size: int, max_size: int) -> list:
        """
        Generate the task set based on data_type.

        Following Zhou et al. 2023:
        - size: (n,) in [min_size, max_size]
        - distribution: (m, c) with Gaussian mixture modes/concentrations
        - size_distribution: (n, m, c) combining both

        Args:
            data_type: Type of task set to generate.
            min_size: Minimum problem size.
            max_size: Maximum problem size.

        Returns:
            List of tasks.
        """
        if data_type == "distribution":
            task_set = [(0, 0)] + [(m, c) for m in range(1, 10) for c in [1, 10, 20, 30, 40, 50]]
        elif data_type == "size":
            task_set = [(n,) for n in range(min_size, max_size + 1)]  # type: ignore[misc]
        elif data_type == "size_distribution":
            dist_set = [(0, 0), (1, 1)] + [(m, c) for m in [3, 5, 7] for c in [10, 30, 50]]
            task_set = [(n, m, c) for n in range(50, 201, 5) for (m, c) in dist_set]  # type: ignore[misc]
        else:
            raise ValueError(f"Unknown data_type: {data_type}. Use 'size', 'distribution', or 'size_distribution'.")

        print(f">> Generating training task set: {len(task_set)} tasks with type {data_type}")
        return task_set
