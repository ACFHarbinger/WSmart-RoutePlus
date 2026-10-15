"""
Consistency and symmetry losses for NCO.

This module provides functions to calculate consistency and symmetry
losses for NCO.

Attributes:
    problem_symmetricity_loss: Consistency loss across problem augmentations.
    solution_symmetricity_loss: Consistency loss across solution starts.
    invariance_loss: Loss for invariant representation across augmentations.

Example:
    >>> import losses
    >>> losses.problem_symmetricity_loss(reward, log_likelihood)
    >>> losses.solution_symmetricity_loss(reward, log_likelihood)
    >>> losses.invariance_loss(proj_embed, num_augment)
"""

from typing import Union

import torch
import torch.nn.functional as F


def problem_symmetricity_loss(reward: torch.Tensor, log_likelihood: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Consistency loss across problem augmentations.

    Args:
        reward: Reward tensor.
        log_likelihood: Log-likelihood tensor.
        dim: Dimension to calculate the loss over.

    Returns:
        torch.Tensor: Consistency loss.
    """
    if reward.shape[dim] < 2:
        return torch.tensor(0.0, device=reward.device)
    advantage = reward - reward.mean(dim=dim, keepdim=True)
    loss = -(advantage.detach() * log_likelihood)
    return loss.mean()


def solution_symmetricity_loss(reward: torch.Tensor, log_likelihood: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Consistency loss across solution starts.

    Args:
        reward: Reward tensor.
        log_likelihood: Log-likelihood tensor.
        dim: Dimension to calculate the loss over.

    Returns:
        torch.Tensor: Consistency loss.
    """
    if reward.shape[dim] < 2:
        return torch.tensor(0.0, device=reward.device)
    advantage = reward - reward.mean(dim=dim, keepdim=True)
    loss = -(advantage.detach() * log_likelihood)
    return loss.mean()


def invariance_loss(proj_embed: torch.Tensor, num_augment: int) -> torch.Tensor:
    """
    Loss for invariant representation across augmentations.

    Args:
        proj_embed: Projected embedding tensor.
        num_augment: Number of augmentations.

    Returns:
        torch.Tensor: Invariance loss.
    """
    # proj_embed: [batch * num_augment, graph_size, d] or [batch * num_augment, d]
    # Handle both cases by flattening if needed
    if proj_embed.dim() == 3:
        # [batch * num_augment, graph_size, d] -> [batch * num_augment, graph_size * d]
        proj_embed = proj_embed.view(proj_embed.shape[0], -1)

    bs = proj_embed.shape[0] // num_augment
    pe = proj_embed.view(bs, num_augment, -1)

    # Cosine similarity between first augmentation and others
    similarity: Union[torch.Tensor, float] = 0.0
    ref = pe[:, 0]
    for i in range(1, num_augment):
        similarity += F.cosine_similarity(ref, pe[:, i], dim=-1)

    # We want to maximize similarity, so minimize negative similarity
    # Ensure similarity is a tensor before calling mean
    if isinstance(similarity, float):
        return torch.tensor(-similarity / (num_augment - 1))
    return -(similarity / (num_augment - 1)).mean()
