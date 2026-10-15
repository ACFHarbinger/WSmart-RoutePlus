"""MatNet (Matrix-based Neural Network) components.

This package provides the implementation of MatNet (Kwon et al. 2021), optimized
for combinatorial optimization problems with matrix-form inputs.

Attributes:
    MatNet: The REINFORCE training wrapper.
    MatNetPolicy: The matrix-aware encoder-decoder policy.

"""

from .model import MatNet as MatNet
from .policy import MatNetPolicy as MatNetPolicy

__all__ = ["MatNet", "MatNetPolicy"]
