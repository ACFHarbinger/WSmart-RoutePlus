"""
Evaluator implementations.

Attributes:
    AugmentationEval: Evaluates a policy using augmentation.
    MultiStartAugmentEval: Evaluates a policy using multi-start and augmentation.
    GreedyEval: Evaluates a policy using greedy search.
    MultiStartEval: Evaluates a policy using multi-start search.
    SamplingEval: Evaluates a policy using sampling.

Example:
    >>> from logic.src.pipeline.features.eval.evaluators import GreedyEval
    >>> evaluator = GreedyEval(config)
    >>> evaluator.evaluate()
"""

from .augmentation import AugmentationEval
from .combined import MultiStartAugmentEval
from .greedy import GreedyEval
from .multi_start import MultiStartEval
from .sampling import SamplingEval

__all__ = [
    "GreedyEval",
    "SamplingEval",
    "AugmentationEval",
    "MultiStartEval",
    "MultiStartAugmentEval",
]
