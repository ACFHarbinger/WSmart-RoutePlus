"""Expert-policy factory: simulator policy config -> vectorised policy.

Every vectorised expert takes the very config dataclass of its simulator
counterpart (``logic.src.configs.policies``). Policy classes load lazily so the
imitation trainers only import the expert configured for the run.

Attributes:
    EXPERT_TARGETS: Config class name -> (module, policy class).
    load_expert_policy_cls: Resolve the policy class for a config.
    create_expert_policy: Instantiate the expert for a config.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

EXPERT_TARGETS: Dict[str, Tuple[str, str]] = {
    "ALNSConfig": ("logic.src.policies.vector.alns", "VectorizedALNS"),
    "HGSConfig": ("logic.src.policies.vector.hgs", "VectorizedHGS"),
    "HGSALNSConfig": ("logic.src.policies.vector.hgs_alns", "VectorizedHGSALNS"),
    "HVPLConfig": ("logic.src.policies.vector.hybrid_volleyball_premier_league", "VectorizedHVPL"),
    "ILSConfig": ("logic.src.policies.vector.iterated_local_search", "IteratedLocalSearchPolicy"),
    "KSparseACOConfig": ("logic.src.policies.vector.ant_colony_system", "VectorizedACOPolicy"),
    "RandomLocalSearchPostConfig": ("logic.src.policies.vector.random_local_search", "RandomLocalSearchPolicy"),
    "ProfitableNNConfig": (
        "logic.src.policies.vector.profitable_nearest_neighbour",
        "ProfitableNearestNeighbourPolicy",
    ),
}


def load_expert_policy_cls(config: Any) -> Any:
    """Import the expert policy class named by ``config``'s class name.

    Args:
        config: Simulator policy config instance.

    Returns:
        The vectorised policy class.

    Raises:
        ValueError: If the config type has no vectorised expert.
    """
    try:
        module_name, attr = EXPERT_TARGETS[type(config).__name__]
    except KeyError:
        raise ValueError(
            f"Unknown policy config type: {type(config).__name__}. Supported types: {sorted(EXPERT_TARGETS)}"
        ) from None
    return getattr(importlib.import_module(module_name), attr)


def create_expert_policy(config: Any, env_name: str, seed: int, device: str) -> Any:
    """Instantiate the vectorised expert for ``config``.

    Args:
        config: Simulator policy config instance.
        env_name: Environment name.
        seed: Random seed.
        device: Torch device.

    Returns:
        The expert policy.
    """
    return load_expert_policy_cls(config)(env_name=env_name, config=config, seed=seed, device=device)
