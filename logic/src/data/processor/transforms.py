"""
Data augmentation transforms for RL4CO.

Attributes:
    batchify: Repeats the TensorDict along a new dimension.
    dihedral_8_augmentation: Augmentation (x8) for grid-based data (x, y) using Dihedral group D4.
    dihedral_8_augmentation_wrapper: Wrapper for dihedral_8_augmentation.
    symmetric_transform: Symmetric rotation and reflection transform.
    symmetric_augmentation: Augment xy data by `num_augment` times via symmetric transform.
    min_max_normalize: Normalize tensor to [0, 1] range using min-max scaling.
    get_augment_function: Get augmentation function by name or return callable directly.
    StateAugmentation: Augment state by N times via symmetric rotation/reflection transform.

Example:
    batchify(td, num_samples)
    dihedral_8_augmentation(xy)
    dihedral_8_augmentation_wrapper(xy, reduce=True)
    symmetric_transform(x, y, phi, offset=0.5)
    symmetric_augmentation(xy, num_augment=8, first_augment=False)
    min_max_normalize(x)
    get_augment_function("dihedral8")
    state_augmentation = StateAugmentation(num_augment=8, augment_fn="symmetric")
"""

import math
from typing import Any, Callable, List, Optional, Union, cast

import torch
from tensordict import TensorDict

from logic.src.tracking.logging.pylogger import get_pylogger

log = get_pylogger(__name__)


def batchify(td: TensorDict, num_samples: int) -> TensorDict:
    """Repeats the TensorDict along a new dimension.

    Args:
        td: TensorDict to repeat.
        num_samples: Number of times to repeat.

    Returns:
        Repeated TensorDict.
    """
    # Tensordict 0.3.1 might not propagate reshape(-1) to all keys correctly
    # Use repeat_interleave which is more explicit
    try:
        return td.repeat_interleave(num_samples, dim=0)
    except Exception:
        # Fallback for very old versions or nested structures
        # We must set batch_size to the TARGET size first to allow item assignment of different shape
        new_batch_size = torch.Size([td.batch_size[0] * num_samples, *td.batch_size[1:]])
        new_td = TensorDict({}, batch_size=new_batch_size, device=td.device)
        for key, val in td.items():
            if isinstance(val, torch.Tensor):
                new_td[key] = val.repeat_interleave(num_samples, dim=0)
            elif isinstance(val, (TensorDict, dict)):
                # If it's a dict, we still need to try expanding its contents if it has tensors
                if isinstance(val, dict):
                    new_td[key] = {
                        k: (v.repeat_interleave(num_samples, dim=0) if isinstance(v, torch.Tensor) else v)
                        for k, v in val.items()
                    }
                else:
                    new_td[key] = batchify(val, num_samples)
        return new_td


def dihedral_8_augmentation(xy: torch.Tensor) -> torch.Tensor:
    """
    Augmentation (x8) for grid-based data (x, y) using Dihedral group D4.
    Args:
        xy: [batch, graph, 2] tensor of x and y coordinates

    Returns:
        [batch*8, graph, 2] tensor of augmented x and y coordinates
    """
    x, y = xy.split(1, dim=2)
    # Augmentations [batch, graph, 2]
    z0 = torch.cat((x, y), dim=2)
    z1 = torch.cat((1 - x, y), dim=2)
    z2 = torch.cat((x, 1 - y), dim=2)
    z3 = torch.cat((1 - x, 1 - y), dim=2)
    z4 = torch.cat((y, x), dim=2)
    z5 = torch.cat((1 - y, x), dim=2)
    z6 = torch.cat((y, 1 - x), dim=2)
    z7 = torch.cat((1 - y, 1 - x), dim=2)
    # [batch*8, graph, 2]
    aug_xy = torch.cat((z0, z1, z2, z3, z4, z5, z6, z7), dim=0)
    return aug_xy


def dihedral_8_augmentation_wrapper(xy: torch.Tensor, reduce: bool = True, *args, **kw) -> torch.Tensor:
    """Wrapper for dihedral_8_augmentation.

    Args:
        xy: [batch, graph, 2] tensor of x and y coordinates
        reduce: Whether to reduce the batch size by a factor of 8.
        args: Additional arguments.
        kw: Additional keyword arguments.

    Returns:
        [batch*8, graph, 2] tensor of augmented x and y coordinates
    """
    if reduce:
        if xy.shape[0] % 8 != 0:
            # Fallback or warning if batch size is not divisible by 8
            log.warning(f"Batch size {xy.shape[0]} is not divisible by 8 for dihedral8 augmentation. Using full batch.")
        else:
            xy = xy[: xy.shape[0] // 8]
    return dihedral_8_augmentation(xy)


def symmetric_transform(x: torch.Tensor, y: torch.Tensor, phi: torch.Tensor, offset: float = 0.5):
    """Symmetric rotation and reflection transform.

    Args:
        x: x coordinates.
        y: y coordinates.
        phi: Rotation angles.
        offset: Offset for transformation.

    Returns:
        Augmented coordinates.
    """
    x, y = x - offset, y - offset
    # Random rotation
    x_prime = torch.cos(phi) * x - torch.sin(phi) * y
    y_prime = torch.sin(phi) * x + torch.cos(phi) * y
    # Random reflection if phi > 2*pi
    mask = phi > 2 * math.pi
    xy = torch.cat((x_prime, y_prime), dim=-1)
    xy = torch.where(mask, xy.flip(-1), xy)
    return xy + offset


def symmetric_augmentation(
    xy: torch.Tensor,
    num_augment: int = 8,
    first_augment: bool = False,
    generator: Optional[torch.Generator] = None,
    **kwargs,
):
    """Augment xy data by `num_augment` times via symmetric transform.

    Args:
        xy: [batch, graph, 2] tensor of x and y coordinates
        num_augment: Number of augmented copies to generate.
        first_augment: If True, first augmentation is the identity.
        generator: Random number generator.
        kwargs: Additional keyword arguments.

    Returns:
        [batch*num_augment, graph, 2] tensor of augmented x and y coordinates
    """
    if generator is None:
        generator = torch.Generator(device=xy.device)
    total_batch = xy.shape[0]
    phi = torch.rand(total_batch, device=xy.device, generator=generator) * 4 * math.pi
    if first_augment:
        # Identity for the first augmentation of each sample
        # Assuming batchify was used: [S0_A0, S0_A1, ..., S0_AN, S1_A0, ...]
        idx = torch.arange(0, total_batch, num_augment, device=xy.device)
        phi[idx] = 0.0

    x, y = xy.split(1, dim=-1)
    return symmetric_transform(x, y, phi[:, None, None])


def min_max_normalize(x):
    """
    Normalize tensor to [0, 1] range using min-max scaling.

    Args:
        x: Input tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def get_augment_function(augment_fn: Union[str, Callable]):
    """
    Get augmentation function by name or return callable directly.

    Args:
        augment_fn: Either a string ('dihedral8', 'symmetric') or a callable.

    Returns:
        Callable: The augmentation function.

    Raises:
        ValueError: If augment_fn string is not recognized.
    """
    if callable(augment_fn):
        return augment_fn
    if augment_fn == "dihedral8":
        return dihedral_8_augmentation_wrapper
    if augment_fn == "symmetric":
        return symmetric_augmentation
    raise ValueError(f"Unknown augment_fn: {augment_fn}")


class StateAugmentation:
    """
    Augment state by N times via symmetric rotation/reflection transform.

    Attributes:
        num_augment: Number of augmented copies to generate.
        augment_fn: Augmentation function ('symmetric', 'dihedral8', or callable).
        first_aug_identity: If True, first augmentation is the identity.
        normalize: Whether to apply min-max normalization after augmentation.
        feats: List of feature names to augment (default: ['locs']).
    """

    def __init__(
        self,
        num_augment: int = 8,
        augment_fn: Union[str, Callable] = "symmetric",
        first_aug_identity: bool = True,
        normalize: bool = False,
        feats: Optional[List[str]] = None,
    ):
        """
        Initialize StateAugmentation transform.

        Args:
            num_augment: Number of augmented copies to generate.
            augment_fn: Augmentation function ('symmetric', 'dihedral8', or callable).
            first_aug_identity: If True, first augmentation is the identity.
            normalize: Whether to apply min-max normalization after augmentation.
            feats: List of feature names to augment (default: ['locs']).
        """
        self.augmentation = get_augment_function(augment_fn)
        if feats is None:
            self.feats = ["locs"]
        else:
            self.feats = feats
        self.num_augment = num_augment
        self.normalize = normalize
        self.first_aug_identity = first_aug_identity

    def __call__(self, td: TensorDict) -> TensorDict:
        """
        Apply augmentation to a TensorDict.

        Args:
            td: Input TensorDict to augment.

        Returns:
            TensorDict: Augmented TensorDict with num_augment copies.
        """
        # (batch) -> (batch, num_augment) -> (batch * num_augment)
        td_aug = batchify(td, self.num_augment)
        batch_size = td.batch_size[0]
        idx = torch.arange(0, batch_size * self.num_augment, self.num_augment, device=td.device)

        for feat in self.feats:
            init_aug_feat = None
            if self.first_aug_identity:
                init_aug_feat = cast(torch.Tensor, td_aug[feat][idx]).clone()

            # Apply transformation
            aug_feat = cast(Any, self.augmentation)(cast(torch.Tensor, td_aug[feat]), num_augment=self.num_augment)

            if self.normalize:
                aug_feat = min_max_normalize(aug_feat)

            if self.first_aug_identity and init_aug_feat is not None:
                cast(torch.Tensor, aug_feat)[idx] = init_aug_feat

            td_aug[feat] = aug_feat

        return td_aug
