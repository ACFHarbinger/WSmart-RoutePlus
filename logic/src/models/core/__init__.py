"""Core Neural Architectures for combinatorial optimization.

This package contains the primary model architectures used for solving
Vehicle Routing Problems, ranging from canonical attention-based seekers to
advanced meta-learning and iterative refinement models.

Attributes:
    AttentionModel: The standard graph attention-based policy.
    MatNet: The matrix-based encoder-decoder model.

"""

from typing import Any

# Model classes load lazily (P-25 / Q-12a): importing the AM must not pull
# MatNet or any other architecture.
_LAZY = {
    "AttentionModel": "logic.src.models.core.attention_model:AttentionModel",
    "MatNet": "logic.src.models.core.matnet:MatNet",
}

__all__ = [
    "AttentionModel",
    "MatNet",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, _, attr = _LAZY[name].partition(":")
        module = importlib.import_module(module_name)
        obj = getattr(module, attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
