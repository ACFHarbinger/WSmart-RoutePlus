"""
Simulation HPO Config module.

Attributes:
    SimHPOConfig: Configuration for simulation policy hyperparameter optimization.

Example:
    sim_hpo_config = SimHPOConfig(
        method="tpe",
        metric="profit",
        n_trials=20,
        num_workers=1,
        policy_name="alns",
        env=EnvConfig(),
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from logic.src.configs.envs.env import EnvConfig, GraphConfig


@dataclass
class SimHPOConfig:
    """Configuration for simulation policy hyperparameter optimization.

    Attributes:
        method: HPO method ('tpe', 'random', 'grid', 'nsgaii', etc.).
        metric: Optimization metric to maximize (e.g., 'profit').
        metrics: List of metrics for multi-objective optimization.
        n_trials: Number of HPO trials.
        num_workers: Number of parallel workers for HPO.
        search_space: Dictionary defining the search space.
        policy_name: Name of the policy to optimize (e.g., 'alns').
        selection_name: Optional name of the selection strategy (from filters/).
        acceptance_name: Optional name of the acceptance criterion (from rules/).
        improver_name: Optional name of the route improver (from interceptors/).
        policy_keywords: Optional keywords to filter policy parameters.
        selection_keywords: Optional keywords to filter selection parameters.
        acceptance_keywords: Optional keywords to filter acceptance parameters.
        improver_keywords: Optional keywords to filter improver parameters.
        env: Environment configuration for simulation trials.
    """

    method: str = "tpe"
    metric: str = "profit"
    metrics: List[str] = field(default_factory=list)
    n_trials: int = 10
    num_workers: int = 1
    search_space: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    policy_name: str = "alns"
    selection_name: Optional[str] = None
    acceptance_name: Optional[str] = None
    improver_name: Optional[str] = None
    policy_keywords: Optional[str] = None
    selection_keywords: Optional[str] = None
    acceptance_keywords: Optional[str] = None
    improver_keywords: Optional[str] = None
    graph: Any = field(default_factory=lambda: GraphConfig(n_samples=5))
    env: Optional[EnvConfig] = None
