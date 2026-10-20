"""POMO specific configuration.

Attributes:
    POMOConfig: Configuration for POMO algorithm.

Example:
    pomo_config = POMOConfig(
        num_augment=1,
        num_starts=None,
        augment_fn="dihedral8",
    )
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class POMOConfig:
    """POMO specific configuration.

    Attributes:
        num_augment: Number of augmentations to perform.
        num_starts: Number of starts for the algorithm.
        augment_fn: Function to use for augmentation.
        mandatory_starts_only: If True, restrict multi-start rollouts to mandatory
            nodes only (requires 'mandatory' field in TensorDict). The number of
            starts is determined by the maximum mandatory-node count across the
            batch; instances with fewer mandatory nodes have their last mandatory
            node repeated as padding.  Falls back to normal behaviour when no
            mandatory nodes are present.
    """

    num_augment: int = 1
    num_starts: Optional[int] = None
    augment_fn: str = "dihedral8"
    mandatory_starts_only: bool = False
