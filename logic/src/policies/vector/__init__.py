"""Vectorised (GPU) policies and short-name model registry.

This package re-exports model policies and the vectorised solvers lazily
(owner ruling Q-4a: importing one expert or model must not pull the whole
neural stack), and provides ``get_policy``/``get_policy_class`` for the
short CLI identifiers.

Attributes:
    get_policy: Function to instantiate a policy by its identifier.
    get_policy_class: Function to retrieve a policy class by name.

Example:
    >>> from logic.src.policies.vector import get_policy
    >>> model = get_policy("am", embed_dim=128)
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

from torch import nn

# Lazy name -> "module:attribute" re-exports. Accessed via module __getattr__.
_LAZY: Dict[str, str] = {
    "ConstructivePolicy": "logic.src.models.common:ConstructivePolicy",
    "ImprovementPolicy": "logic.src.models.common:ImprovementPolicy",
    "NonAutoregressivePolicy": "logic.src.models.common:NonAutoregressivePolicy",
    "AttentionModelPolicy": "logic.src.models.core.attention_model.policy:AttentionModelPolicy",
    "DeepDecoderPolicy": (
        "logic.src.models.core.attention_model.deep_decoder_policy:DeepDecoderPolicy"
    ),
    "SymNCOPolicy": "logic.src.models.core.attention_model.symnco_policy:SymNCOPolicy",
    "DACTPolicy": "logic.src.models.core.dact.policy:DACTPolicy",
    "DeepACOPolicy": "logic.src.models.core.deepaco.policy:DeepACOPolicy",
    "GFACSPolicy": "logic.src.models.core.gfacs.policy:GFACSPolicy",
    "GLOPPolicy": "logic.src.models.core.glop.policy:GLOPPolicy",
    "MDAMPolicy": "logic.src.models.core.mdam.policy:MDAMPolicy",
    "MoEPolicy": "logic.src.models.core.moe.policy:MoEPolicy",
    "N2SPolicy": "logic.src.models.core.n2s.policy:N2SPolicy",
    "NARGNNPolicy": "logic.src.models.core.nargnn.policy:NARGNNPolicy",
    "NeuOptPolicy": "logic.src.models.core.neuopt.policy:NeuOptPolicy",
    "PointerNetworkPolicy": "logic.src.models.core.pointer_network.policy:PointerNetworkPolicy",
    "PolyNetPolicy": "logic.src.models.core.polynet.policy:PolyNetPolicy",
    "TemporalAMPolicy": (
        "logic.src.models.core.temporal_attention_model.policy:TemporalAMPolicy"
    ),
    "NeuralHeuristicHybrid": (
        "logic.src.models.core.hybrid_attention_model.hybrid_neural_heuristic_policy"
        ":NeuralHeuristicHybrid"
    ),
    "HybridTwoStagePolicy": (
        "logic.src.models.core.hybrid_attention_model.hybrid_two_step_policy"
        ":HybridTwoStagePolicy"
    ),
    "VectorizedALNS": "logic.src.policies.vector.alns:VectorizedALNS",
    "VectorizedACOPolicy": "logic.src.policies.vector.ant_colony_system:VectorizedACOPolicy",
    "VectorizedHGS": "logic.src.policies.vector.hgs:VectorizedHGS",
    "VectorizedHGSALNS": "logic.src.policies.vector.hgs_alns:VectorizedHGSALNS",
    "VectorizedHVPL": (
        "logic.src.policies.vector.hybrid_volleyball_premier_league:VectorizedHVPL"
    ),
    "IteratedLocalSearchPolicy": (
        "logic.src.policies.vector.iterated_local_search:IteratedLocalSearchPolicy"
    ),
}

# Short-name registry: CLI model name -> (module_path, class_name)
# This enables ``get_policy("am")`` without importing everything eagerly.
_POLICY_REGISTRY_SPEC: Dict[str, Tuple[str, str]] = {
    "am": ("logic.src.models.core.attention_model.policy", "AttentionModelPolicy"),
    "deep_decoder": (
        "logic.src.models.core.attention_model.deep_decoder_policy",
        "DeepDecoderPolicy",
    ),
    "deepaco": ("logic.src.models.core.deepaco.policy", "DeepACOPolicy"),
    "gfacs": ("logic.src.models.core.gfacs.policy", "GFACSPolicy"),
    "glop": ("logic.src.models.core.glop.policy", "GLOPPolicy"),
    "mdam": ("logic.src.models.core.mdam.policy", "MDAMPolicy"),
    "moe": ("logic.src.models.core.moe.policy", "MoEPolicy"),
    "nargnn": ("logic.src.models.core.nargnn.policy", "NARGNNPolicy"),
    "n2s": ("logic.src.models.core.n2s.policy", "N2SPolicy"),
    "neuopt": ("logic.src.models.core.neuopt.policy", "NeuOptPolicy"),
    "dact": ("logic.src.models.core.dact.policy", "DACTPolicy"),
    "pointer": (
        "logic.src.models.core.pointer_network.policy",
        "PointerNetworkPolicy",
    ),
    "polynet": ("logic.src.models.core.polynet.policy", "PolyNetPolicy"),
    "symnco": (
        "logic.src.models.core.attention_model.symnco_policy",
        "SymNCOPolicy",
    ),
    "temporal": (
        "logic.src.models.core.temporal_attention_model.policy",
        "TemporalAMPolicy",
    ),
    "hvpl": (
        "logic.src.policies.vector.hybrid_volleyball_premier_league",
        "VectorizedHVPL",
    ),
    "hgs": ("logic.src.policies.vector.hgs", "VectorizedHGS"),
    "hgs_alns": ("logic.src.policies.vector.hgs_alns", "VectorizedHGSALNS"),
}


def get_policy_class(name: str) -> type:
    """Look up a policy class by its short CLI name.

    Args:
        name: Short policy name (e.g. "am", "mdam", "deepaco").

    Returns:
        type: The policy class (not instantiated).

    Raises:
        ValueError: If the name is not found in the registry.
    """
    if name not in _POLICY_REGISTRY_SPEC:
        raise ValueError(f"Unknown policy: {name!r}. Available: {sorted(_POLICY_REGISTRY_SPEC.keys())}")
    module_path, class_name = _POLICY_REGISTRY_SPEC[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_policy(name: str, **kwargs: Any) -> nn.Module:
    """Create a policy instance by short CLI name.

    Args:
        name: Short identifier for the policy (e.g., "am", "hgs").
        kwargs: Policy-specific initialization arguments.

    Returns:
        nn.Module: Instantiated policy module.
    """
    cls = get_policy_class(name)
    return cls(**kwargs)


__all__ = [
    "ConstructivePolicy",
    "ImprovementPolicy",
    "AttentionModelPolicy",
    "DeepDecoderPolicy",
    "DeepACOPolicy",
    "GFACSPolicy",
    "GLOPPolicy",
    "MDAMPolicy",
    "NARGNNPolicy",
    "PolyNetPolicy",
    "TemporalAMPolicy",
    "PointerNetworkPolicy",
    "SymNCOPolicy",
    "MoEPolicy",
    "NeuOptPolicy",
    "DACTPolicy",
    "N2SPolicy",
    "NonAutoregressivePolicy",
    "get_policy_class",
    "get_policy",
    "VectorizedHGSALNS",
    "IteratedLocalSearchPolicy",
    "VectorizedACOPolicy",
    "VectorizedALNS",
    "VectorizedHGS",
    "VectorizedHVPL",
    "NeuralHeuristicHybrid",
    "HybridTwoStagePolicy",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_name, _, attr = _LAZY[name].partition(":")
        module = importlib.import_module(module_name)
        obj = getattr(module, attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
