"""
Configuration parameters for the Neural Policy.

Attributes:
    NeuralParams: Configuration parameters for the Neural Policy.

Example:
    >>> from logic.src.policies.routing.learning_algorithms import NeuralAgent
    >>> policy = NeuralAgent(model)
    >>> routes, metrics = policy.run_day(env)
    >>> print(f"Best routes: {routes}")
    >>> print(f"Metrics: {metrics}")
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class NeuralParams:
    """
    Configuration parameters for the Neural Policy.

    Attributes:
        waste_weight: Reward multiplier for collected waste (revenue).
        cost_weight: Penalty multiplier for travel distance.
        overflow_penalty: Penalty multiplier for bin overflows.
        selector_name: Name of vectorized selector for mandatory filtering.
        selector_threshold: Confidence threshold for node selection.
        decoding_strategy: Decoding strategy (greedy, sampling, beam_search).
        beam_width: Width of the search beam if beam_search is used.
        seed: Random seed for reproducibility.
    """

    waste_weight: float = 1.0
    cost_weight: float = 1.0
    overflow_penalty: float = 1.0
    selector_name: Optional[str] = None
    selector_threshold: float = 0.7
    decoding_strategy: str = "greedy"
    beam_width: int = 1
    reward_weight: float = 0.0
    length_penalty_alpha: float = 0.0
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> NeuralParams:
        """Create NeuralParams from a configuration object or dictionary.

        Args:
            config: Configuration object or dictionary.

        Returns:
            NeuralParams: Configuration parameters for the Neural Policy.
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert NeuralParams to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of NeuralParams.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
