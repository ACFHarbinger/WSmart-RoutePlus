"""
Configuration dataclass for the HNA policy (simulator-callable wrapper config).

Attributes:
    HNAPolicyConfig: Configuration for the HNA policy.

Example:
    >>> from configs.policies.hna import HNAPolicyConfig
    >>> config = HNAPolicyConfig()
    >>> config.time_limit
    60.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HNAPolicyConfig:
    """Configuration for the Hierarchical Neural Agent simulator policy adapter.

    This is the lightweight config for the *simulator-callable* shim
    (``policy_hna.py``), not the full Lightning training module config.

    Attributes:
        checkpoint_path: Path to a pre-trained HNAModule checkpoint.
            If None, the manager falls back to a threshold-based heuristic
            and the worker uses the greedy nearest-neighbour heuristic.
        device: Torch device string ("cpu", "cuda", "cuda:0", etc.).
        horizon: Lookahead horizon passed to the manager observation.
        overflow_penalty: Per-unit overflow penalty used in reward computation.
        greedy_threshold: Fill-level threshold (%) for the fallback mandatory
            selector when no checkpoint is provided.
        seed: Random seed for stochastic components.
        verbose: Enable per-day inference logging.
    """

    checkpoint_path: Optional[str] = None
    device: str = "cpu"
    horizon: int = 7
    overflow_penalty: float = 500.0
    greedy_threshold: float = 75.0
    seed: Optional[int] = None
    verbose: bool = False


    @classmethod
    def from_config(cls, config: Any) -> "HNAPolicyConfig":
        """Create HNAPolicyConfig from a configuration object or dictionary.

        Args:
            config: Configuration object or dictionary.

        Returns:
            "HNAPolicyConfig": Configuration parameters for the Hierarchical Neural Agent.
        """
        from dataclasses import fields
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            checkpoint_path=getattr(config, "checkpoint_path", None),
            device=getattr(config, "device", "cpu"),
            horizon=getattr(config, "horizon", 7),
            overflow_penalty=getattr(config, "overflow_penalty", 500.0),
            greedy_threshold=getattr(config, "greedy_threshold", 75.0),
            seed=getattr(config, "seed", None),
            verbose=getattr(config, "verbose", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert HNAPolicyConfig to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Dictionary representation of HNAPolicyConfig.
        """
        from dataclasses import fields
        return {f.name: getattr(self, f.name) for f in fields(self)}
