"""
Data logic for RL4COLitModule.

Attributes:
    DataMixin: Mixin for data loading logic.

Example:
    None
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

import torch
from torch.utils.data import DataLoader

from logic.src.data.datasets import TensorDictDataset
from logic.src.tracking.logging.pylogger import get_pylogger
from logic.src.utils.data.td_utils import tensordict_collate_fn

if TYPE_CHECKING:
    from logic.src.interfaces.env import IEnv
    from logic.src.interfaces.policy import IPolicy

logger = get_pylogger(__name__)


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Get a field from a dict-like or attribute-style config object."""
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _task_env_cfg(cfg: Any) -> Any:
    """Return the env config of the active task (``cfg.<task>.env``), else ``cfg.env``.

    The root config has no top-level ``env``; curriculum stages write the stage
    graph into ``cfg.train.env.graph`` (see ``features/train/engine.py``).
    """
    if cfg is None:
        return None
    task_cfg = getattr(cfg, str(getattr(cfg, "task", "train") or "train"), None)
    env_cfg = getattr(task_cfg, "env", None) if task_cfg is not None else None
    return env_cfg if env_cfg is not None else getattr(cfg, "env", None)


def _get_eval_graphs(cfg: Any) -> list:
    """Return the active task's env.eval_graphs as a list, or [] if absent/empty."""
    if cfg is None:
        return []
    try:
        env_cfg = _task_env_cfg(cfg)
        if env_cfg is None:
            return []
        eval_graphs = getattr(env_cfg, "eval_graphs", None)
        if not eval_graphs:
            return []
        return list(eval_graphs)
    except Exception:
        return []


def _create_eval_env_and_gen(cfg: Any, eval_graph: Any) -> tuple:
    """Build an env and its generator for a single eval_graph config entry.

    Args:
        cfg: Root config (DictConfig or plain dict-like).
        eval_graph: A single eval_graph entry (dict-like or object).

    Returns:
        Tuple of (env, generator) both on CPU.
    """
    from logic.src.envs import get_env

    env_cfg = _task_env_cfg(cfg)
    env_name = str(_cfg_get(env_cfg, "name", "vrpp") or "vrpp")
    env_graph = _cfg_get(env_cfg, "graph", None)
    train_cfg = getattr(cfg, "train", None)

    # Reward priority: eval_graph.reward → env.graph.reward → defaults
    # This allows per-graph objective weights for validation.
    eval_graph_reward = _cfg_get(eval_graph, "reward", None)
    env_graph_reward = _cfg_get(env_graph, "reward", None)
    active_reward = eval_graph_reward if eval_graph_reward is not None else env_graph_reward

    cost_weight = float(_cfg_get(active_reward, "cost_weight", 1.0) or 1.0)
    waste_weight = float(_cfg_get(active_reward, "waste_weight", 1.0) or 1.0)
    data_dist = str(_cfg_get(train_cfg, "data_distribution", "gamma3") or "gamma3")

    n_samples = int(_cfg_get(eval_graph, "n_samples", 512))
    num_loc = int(_cfg_get(eval_graph, "num_loc", _cfg_get(env_graph, "num_loc", 50)))
    area = str(_cfg_get(eval_graph, "area", _cfg_get(env_graph, "area", "riomaior")))
    waste_type = str(_cfg_get(eval_graph, "waste_type", _cfg_get(env_graph, "waste_type", "plastic")))
    focus_graph = _cfg_get(eval_graph, "focus_graph", None)
    focus_size = _cfg_get(eval_graph, "focus_size", None)
    vertex_method = str(_cfg_get(eval_graph, "vertex_method", _cfg_get(env_graph, "vertex_method", "mmn")) or "mmn")
    distance_method = str(
        _cfg_get(eval_graph, "distance_method", _cfg_get(env_graph, "distance_method", "ogd")) or "ogd"
    )
    dm_filepath = _cfg_get(eval_graph, "dm_filepath", _cfg_get(env_graph, "dm_filepath", None))
    start_day = int(_cfg_get(eval_graph, "start_day", _cfg_get(env_graph, "start_day", 0)) or 0)
    n_days = int(_cfg_get(eval_graph, "n_days", 1))

    # Use the training device for the eval env (so env.reset and policy run on GPU);
    # only the data generator is pinned to CPU for offline dataset generation.
    env_device = str(getattr(cfg, "device", "cpu") or "cpu")

    eval_env = get_env(
        env_name,
        num_loc=num_loc,
        area=area,
        waste_type=waste_type,
        focus_graph=focus_graph,
        focus_size=focus_size,
        n_samples=n_samples,
        start_day=start_day,
        n_days=n_days,
        cost_weight=cost_weight,
        waste_weight=waste_weight,
        data_distribution=data_dist,
        device=env_device,
        batch_size=n_samples,
        vertex_method=vertex_method,
        distance_method=distance_method,
        dm_filepath=dm_filepath,
    )

    gen = eval_env.generator
    if gen is not None and hasattr(gen, "to"):
        gen = gen.to("cpu")

    return eval_env, gen


class DataMixin:
    """Mixin for data loading logic.

    Attributes:
        env: Environment for data generation.
        policy: Policy for data generation.
        val_dataset_path: Path to validation dataset.
        train_dataset_path: Path to training dataset.
        batch_size: Batch size.
        num_workers: Number of workers.
        persistent_workers: Whether to use persistent workers.
        pin_memory: Whether to pin memory.
        local_rank: Local rank.
        cfg: Root configuration object.
        train_dataset: Training dataset.
        val_dataset: Single validation dataset (fallback when eval_graphs is empty).
        val_datasets: List of validation datasets, one per eval_graph entry.
        eval_envs: List of envs corresponding to val_datasets (used by StepMixin).
    """

    def __init__(self) -> None:
        """Initialize Class.

        Args:
            None.
        """
        # Type hints for attributes expected from the main class
        self.env: IEnv
        self.policy: IPolicy
        self.val_dataset_path: Optional[str]
        self.train_dataset_path: Optional[str]
        self.batch_size: int
        self.num_workers: int
        self.persistent_workers: bool
        self.pin_memory: bool
        self.local_rank: int
        self.cfg: Any
        self.train_dataset: Optional[Any] = None
        self.val_dataset: Optional[Any] = None
        # Multi-dataset support for eval_graphs
        self.val_datasets: Optional[List[Any]] = None
        self.eval_envs: Optional[List[Any]] = None

    def setup(self, stage: str) -> None:  # noqa: C901
        """
        Set up datasets for training and validation.

        Args:
            stage: The stage ('fit', 'validate', 'test', or 'predict').
        """
        if stage == "fit":
            gen = self.env.generator
            assert gen is not None
            if hasattr(gen, "to"):
                gen = gen.to("cpu")

            # Safely resolve training dataset size from config (cfg may be None in tests)
            _cfg_env = _task_env_cfg(self.cfg)
            _cfg_graph = _cfg_get(_cfg_env, "graph", None)
            n_train = int(_cfg_get(_cfg_graph, "n_samples", 10))

            assert gen is not None
            if self.train_dataset_path is not None and os.path.exists(self.train_dataset_path):
                if self.local_rank == 0:
                    logger.info(f"Loading training dataset from {self.train_dataset_path}")

                self.train_dataset = TensorDictDataset.load(self.train_dataset_path)
                if n_train < len(self.train_dataset):
                    self.train_dataset = TensorDictDataset(self.train_dataset.data[:n_train])
            else:
                if self.local_rank == 0:
                    logger.info(f"Generating training dataset ({n_train} instances) on CPU...")
                data = gen(batch_size=n_train)
                self.train_dataset = TensorDictDataset(data)

            # Build validation datasets from eval_graphs (multi-graph) or fall back to single dataset
            eval_graphs = _get_eval_graphs(self.cfg)
            if eval_graphs:
                self.val_datasets = []
                self.eval_envs = []
                for eval_graph in eval_graphs:
                    n_samples = int(_cfg_get(eval_graph, "n_samples", n_train))
                    num_loc = int(_cfg_get(eval_graph, "num_loc", 50))
                    eval_env, eval_gen = _create_eval_env_and_gen(self.cfg, eval_graph)

                    load_dataset = _cfg_get(eval_graph, "load_dataset", None)
                    if load_dataset is not None and os.path.exists(load_dataset):
                        if self.local_rank == 0:
                            logger.info(
                                f"Loading eval dataset from {load_dataset} ({n_samples} instances, {num_loc} nodes)"
                            )
                        val_ds = TensorDictDataset.load(load_dataset)
                    else:
                        if self.local_rank == 0:
                            logger.info(f"Generating eval dataset ({n_samples} instances, {num_loc} nodes) on CPU...")
                        val_data = eval_gen(batch_size=n_samples) if eval_gen is not None else None
                        if val_data is not None:
                            val_ds = TensorDictDataset(val_data)
                        else:
                            continue

                    # Trim to n_samples before possible day-expansion
                    if n_samples < len(val_ds):
                        val_ds = TensorDictDataset(val_ds.data[:n_samples])

                    # Multi-day expansion: replicate each instance once per eval day,
                    # injecting 'current_day' so the env selects the right waste slice.
                    n_days_eval = int(_cfg_get(eval_graph, "n_days", 1))
                    start_day_eval = int(_cfg_get(eval_graph, "start_day", 0) or 0)
                    waste_field = val_ds.data.get("waste", None)
                    if n_days_eval > 1 and waste_field is not None and waste_field.dim() > 2:
                        n_available = waste_field.shape[1]
                        day_tds = []
                        for day_offset in range(n_days_eval):
                            day_idx = start_day_eval + day_offset
                            if day_idx >= n_available:
                                break
                            td_day = val_ds.data.clone()
                            td_day["current_day"] = torch.full(val_ds.data.batch_size, day_idx, dtype=torch.long)
                            day_tds.append(td_day)
                        if len(day_tds) > 1:
                            merged = cast(Any, torch.cat(day_tds, dim=0))
                            val_ds = TensorDictDataset(merged)
                            if self.local_rank == 0:
                                logger.info(
                                    f"Expanded eval dataset to {len(val_ds)} instances "
                                    f"({n_samples} × {len(day_tds)} days)"
                                )

                    self.val_datasets.append(val_ds)
                    self.eval_envs.append(eval_env)
            else:
                # Fallback: single validation dataset using the training env
                self.val_datasets = None
                self.eval_envs = None
                # env.graph is injected by _build_stage_config; safe fallback when absent.
                env_graph_cfg = _cfg_get(_task_env_cfg(self.cfg), "graph", None)
                n_val = int(_cfg_get(env_graph_cfg, "n_samples", 512))
                if self.val_dataset_path is not None:
                    if self.local_rank == 0:
                        logger.info(f"Loading validation dataset from {self.val_dataset_path}")
                    self.val_dataset = TensorDictDataset.load(self.val_dataset_path)
                else:
                    if self.local_rank == 0:
                        logger.info(f"Generating validation dataset ({n_val} instances) on CPU...")
                    val_data = cast(Any, gen)(batch_size=n_val)
                    self.val_dataset = TensorDictDataset(val_data)
                    assert self.val_dataset is not None
            baseline = getattr(self, "baseline", None)
            if hasattr(baseline, "configure_comparison"):
                # Use training graph/distribution, independent of eval_graphs.
                baseline.configure_comparison(self.env, max(2, n_train), int(_cfg_get(self.cfg, "seed", 0) or 0))
        else:
            pass

    def train_dataloader(self) -> DataLoader:
        """
        Create the training DataLoader.

        Returns:
            DataLoader: DataLoader for training data.
        """
        return DataLoader(
            cast(Any, self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=tensordict_collate_fn,
            pin_memory=self.pin_memory if self.num_workers > 0 and self.train_dataset is not None else False,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0 and self.train_dataset is not None
            else False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Create the validation DataLoader(s).

        Returns a list of DataLoaders when eval_graphs is configured so that
        Lightning validates on each graph size independently.

        Returns:
            Single DataLoader or list of DataLoaders.
        """
        if self.val_datasets:
            return [
                DataLoader(
                    cast(Any, ds),
                    batch_size=self.batch_size,
                    collate_fn=tensordict_collate_fn,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory if self.num_workers > 0 else False,
                    persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
                )
                for ds in self.val_datasets
            ]

        assert self.val_dataset is not None
        return DataLoader(
            cast(Any, self.val_dataset),
            batch_size=self.batch_size,
            collate_fn=tensordict_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory if self.num_workers > 0 and self.val_dataset is not None else False,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0 and self.val_dataset is not None
            else False,
        )
