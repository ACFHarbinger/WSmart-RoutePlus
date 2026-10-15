"""
Neural policy configuration.

Attributes:
    NeuralAgentConfig: Configuration for the Neural Agent policy.

Example:
    >>> from configs.policies.na import NeuralAgentConfig
    >>> config = NeuralAgentConfig()
    >>> config.seed
    None
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from logic.src.configs.envs.objective import ObjectiveConfig
from logic.src.configs.models.model import ModelConfig


@dataclass
class NeuralAgentConfig:
    """Configuration for Neural Agent policy.

    Attributes:
        model: Model configuration (architecture and load path).
        reward: Objective/reward weights configuration.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
        seed: Random seed for reproducibility.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    reward: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    packing: Optional[Any] = None
    route_improvement: Optional[Any] = None
    seed: Optional[int] = None
    waste_weight: float = 1.0
    cost_weight: float = 1.0
    overflow_penalty: float = 1.0
    selector_name: Optional[str] = None
    selector_threshold: float = 0.7
    decoding_strategy: str = "greedy"
    beam_width: int = 1
    reward_weight: float = 0.0
    length_penalty_alpha: float = 0.0

    @classmethod
    def from_config(cls, config: Any) -> NeuralAgentConfig:
        """Create the config from a configuration object or dictionary.

        Args:
            config: Configuration object or dictionary.

        Returns:
            NeuralAgentConfig: Configuration for the Neural Agent policy.
        """
        # 1. Check if the config is a wrapper dict with a single key pointing to the actual list
        if (isinstance(config, dict) or hasattr(config, "items")) and len(config) == 1:
            val = list(config.values())[0]
            if isinstance(val, (list, tuple)) or type(val).__name__ == "ListConfig":
                config = val

        # 2. Flatten the list into a single dict
        if isinstance(config, (list, tuple)) or type(config).__name__ == "ListConfig":
            flattened = {}
            for item in config:
                if isinstance(item, dict) or hasattr(item, "items"):
                    for k, v in item.items():
                        flattened[k] = v
            config = flattened

        if isinstance(config, dict) or hasattr(config, "get"):
            # Check for direct dict keys first
            decoding = config.get("decoding")
            if not decoding:
                # Try nested hierarchy model.decoder.decoding
                model = config.get("model", {})
                decoder = (
                    model.get("decoder", {})
                    if isinstance(model, dict) or hasattr(model, "get")
                    else getattr(model, "decoder", {})
                )
                decoding = (
                    decoder.get("decoding", {})
                    if isinstance(decoder, dict) or hasattr(decoder, "get")
                    else getattr(decoder, "decoding", {})
                )

            return cls(
                waste_weight=config.get("waste_weight", 1.0),
                cost_weight=config.get("cost_weight", 1.0),
                overflow_penalty=config.get("overflow_penalty", 1.0),
                selector_name=config.get("selector_name"),
                selector_threshold=config.get("selector_threshold", 0.7),
                decoding_strategy=decoding.get("strategy", "greedy")
                if isinstance(decoding, dict)
                else getattr(decoding, "strategy", "greedy"),
                beam_width=decoding.get("beam_width", 1)
                if isinstance(decoding, dict)
                else getattr(decoding, "beam_width", 1),
                reward_weight=decoding.get("reward_weight", 0.0)
                if isinstance(decoding, dict)
                else getattr(decoding, "reward_weight", 0.0),
                length_penalty_alpha=decoding.get("length_penalty_alpha", 0.0)
                if isinstance(decoding, dict)
                else getattr(decoding, "length_penalty_alpha", 0.0),
                seed=config.get("seed", 42),
            )

        # Handle Hydra-style config objects
        model = getattr(config, "model", None)
        decoder = getattr(model, "decoder", None) if model else None
        decoding = getattr(decoder, "decoding", None) if decoder else None

        return cls(
            waste_weight=getattr(config, "waste_weight", 1.0),
            cost_weight=getattr(config, "cost_weight", 1.0),
            overflow_penalty=getattr(config, "overflow_penalty", 1.0),
            selector_name=getattr(config, "selector_name", None),
            selector_threshold=getattr(config, "selector_threshold", 0.7),
            decoding_strategy=getattr(decoding, "strategy", "greedy") if decoding else "greedy",
            beam_width=getattr(decoding, "beam_width", 1) if decoding else 1,
            reward_weight=getattr(decoding, "reward_weight", 0.0) if decoding else 0.0,
            length_penalty_alpha=getattr(decoding, "length_penalty_alpha", 0.0) if decoding else 0.0,
            seed=getattr(config, "seed", 42),
        )
