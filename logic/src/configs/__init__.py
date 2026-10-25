"""
Configuration dataclasses for WSmart-Route.

Attributes:
    Config: Root configuration.

Example:
    >>> from logic.src.configs import Config
    >>> config = Config()
    >>> print(config)
    Config(
        train=TrainConfig(
            max_epochs=100, batch_size=32, learning_rate=0.001, dropout_rate=0.1,
            clip_grad_norm=None, weight_decay=0.0, label_smoothing=0.0,
            warmup_epochs=10, reduce_lr_on_plateau=False, patience=5,
            min_lr=1e-06, patience_epochs=5, eval_freq_epochs=10,
            eval_batch_size=None, gradient_accumulation_steps=1, grad_clip=1.0,
            gradient_clipping=1.0, gradient_clipping_norm=None,
            target_temperature=1.0, target_temperature_decay_rate=0.999997,
            temperature_decay_steps=10000, learning_rate_decay=None,
            learning_rate_decay_rate=0.9999, patience=5, min_learning_rate=1e-5
        ),
        optim=OptimConfig(type='adamw', lr=0.001, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0.0, amsgrad=False, foreach=False, grad_clip_norm=None, gradient_clipping=None),
        data=DataConfig(data_dir="", train_size=100, val_size=20, test_size=20, n_train_workers=4, n_val_workers=2, n_test_workers=2, shuffle=True, pin_memory=False, persistent_workers=False, batch_size=32, val_batch_size=32, test_batch_size=32),
        eval=EvalConfig(eval_size=20, n_workers=2, batch_size=32, shuffle=True, pin_memory=False, persistent_workers=False),
        sim=SimConfig(n_agents=1, n_simulations=1000, n_parallel_sims=10, visualize=True, batch_size=32, val_batch_size=32, test_batch_size=32, val_size=20, test_size=20, n_train_workers=0, n_test_workers=0),
        tracking=TrackingConfig(name="", project_name=None, use_mlflow=False, use_tensorboard=False, use_wandb=True),
        seed=42, device='cuda', experiment_name=None, task='train', run_name=None, verbose=True, start=0, p={}, callbacks={})
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .envs import EnvConfig, GraphConfig, ObjectiveConfig
from .learning import RLConfig
from .models import DecoderConfig, DecodingConfig, EncoderConfig, ModelConfig, OptimConfig
from .policies.packing import PackingConfig
from .policies.route_improvement import RouteImprovingConfig
from .tasks import DataConfig, EvalConfig, HPOConfig, MetaRLConfig, SimConfig, SimHPOConfig, TrainConfig
from .tracking import TrackingConfig


@dataclass
class Config:
    """Root configuration.

    Attributes:
        train: Training configuration.
        optim: Optimizer configuration.
        rl: RL algorithm configuration.
        meta_rl: Meta-RL configuration.
        hpo: HPO configuration.
        eval: Evaluation configuration.
        sim: Simulation configuration.
        data: Data generation configuration.
        tracking: Tracking backend configuration (WSTracker + optional MLflow).
        seed: Random seed.
        device: Device to use ('cpu', 'cuda').
        experiment_name: Optional name for the experiment.
        task: Task to perform ('train', 'eval', 'test_sim', 'gen_data').
        wandb_mode: Weights & Biases mode ('online', 'offline', 'disabled').
        no_tensorboard: If True, disable TensorBoard logging.
        no_progress_bar: If True, disable the progress bar.
        run_name: Specific name for the run (separate from experiment_name).
        verbose: If True, enable verbose logging.
        p: Dictionary for arbitrary additional parameters.
        callbacks: Dictionary of callback configurations.
    """

    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    meta_rl: MetaRLConfig = field(default_factory=MetaRLConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    hpo_sim: SimHPOConfig = field(default_factory=SimHPOConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    seed: int = 42
    device: str = "cuda"
    experiment_name: Optional[str] = None
    task: str = "train"
    run_name: Optional[str] = None
    start: int = 0
    p: Dict[str, Any] = field(default_factory=dict)
    callbacks: Dict[str, Any] = field(default_factory=dict)
    # Flat keys from logic/configs/tasks/slurm.yaml. The file is a Hydra
    # tasks-group member, so composition rejects any root key Config lacks.
    # The sbatch helper reads the same names; they stay optional everywhere else.
    number_of_cores: Optional[int] = None
    command: Optional[str] = None
    model: Optional[Any] = None
    problem: Optional[str] = None
    gs: Optional[int] = None
    bs: Optional[int] = None
    ne: Optional[int] = None
    es: Optional[int] = None
    ebs: Optional[int] = None
    et: Optional[float] = None
    days: Optional[int] = None
    od: Optional[str] = None
    ns: Optional[int] = None
    lvl: Optional[Any] = None
    cf: Optional[Any] = None
    nv: Optional[int] = None


__all__ = [
    "EnvConfig",
    "ModelConfig",
    "TrainConfig",
    "OptimConfig",
    "RLConfig",
    "MetaRLConfig",
    "HPOConfig",
    "EvalConfig",
    "SimConfig",
    "SimHPOConfig",
    "DataConfig",
    "TrackingConfig",
    "PackingConfig",
    "RouteImprovingConfig",
    "Config",
    "EncoderConfig",
    "DecoderConfig",
    "DecodingConfig",
    "GraphConfig",
    "ObjectiveConfig",
]
