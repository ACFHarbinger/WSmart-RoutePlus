"""
HPO Config module.

Attributes:
    HPOConfig: Configuration for hyperparameter optimization.

Example:
    hpo_config = HPOConfig(
        method="dehbo",
        metric="reward",
        n_trials=50,
        n_epochs_per_trial=10,
        num_workers=4,
        search_space={
            "rl.entropy_weight": {"type": "float", "low": 0.0, "high": 0.1},
            "optim.lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        },
        graph=GraphConfig(),
        reward=ObjectiveConfig(),
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from logic.src.configs.envs.env import EnvConfig


@dataclass
class HPOConfig:
    """Hyperparameter optimization configuration.

    Attributes:
        method: HPO method ('dehbo', 'rs', 'gs', 'bo').
        metric: Optimization metric ('reward', 'cost').
        n_trials: Number of HPO trials.
        n_epochs_per_trial: Training epochs per trial.
        num_workers: Number of parallel workers for HPO.
        search_space: Dictionary defining the search space.
        graph: Graph configuration.
        reward: Objective/reward configuration.
    """

    method: str = "dehbo"  # dehbo, rs, gs, bo
    metric: str = "reward"
    n_trials: int = 0
    n_epochs_per_trial: int = 10
    num_workers: int = 4
    search_space: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "rl.entropy_weight": {"type": "float", "low": 0.0, "high": 0.1},
            "optim.lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        }
    )
    # NEW FIELDS:
    fevals: int = 100
    timeout: Optional[int] = None
    n_startup_trials: int = 5
    n_warmup_steps: int = 3
    min_fidelity: int = 1
    max_fidelity: int = 10
    # Ray Tune and DEA specifics
    interval_steps: int = 1
    eta: float = 10.0
    indpb: float = 0.2
    tournsize: int = 3
    cxpb: float = 0.7
    mutpb: float = 0.2
    n_pop: int = 20
    n_gen: int = 10
    cpu_cores: int = 1
    train_best: bool = True
    local_mode: bool = False
    num_samples: int = 20
    max_tres: int = 14
    reduction_factor: int = 3
    max_failures: int = 3
    grid: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0])
    max_conc: int = 4

    env: Any = field(default_factory=EnvConfig)
