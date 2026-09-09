"""Visualization helper utilities for the tracking system.

This module provides support tools for model visualization, including
synthetic data generation for testing, a model wrapper for loss landscape
analysis, and model loading functions that recover states from checkpoints.

Attributes:
    get_batch: Generates a random batch of routing data.
    MyModelWrapper: Adapter class for loss-landscapes integration.
    load_model_instance: Reconstructs a model from a saved state.

Example:
    >>> from logic.src.utils.plotting import get_batch
    >>> batch = get_batch(device, size=50)
"""

import torch
from torch import nn

from logic.src.models.core.attention_model import AttentionModel
from logic.src.models.subnets.factories.attention import AttentionComponentFactory
from logic.src.utils.model.problem_factory import load_problem


def get_batch(device, size=50, batch_size=32, temporal_horizon=0):
    """Generates a random batch of VRP-like data for visualization purposes.

    Args:
        device: Device to create tensors on (e.g. 'cpu' or 'cuda').
        size: Number of nodes in the graph. Defaults to 50.
        batch_size: Number of instances in the batch. Defaults to 32.
        temporal_horizon: Time steps for dynamic features. Defaults to 0.

    Returns:
        Dict[str, torch.Tensor]: Batch dictionary containing 'depot', 'loc', etc.
    """
    # TODO: This should ideally use the problem's generate_instance or make_dataset
    all_coords = torch.rand(batch_size, size + 1, 2, device=device)
    depot = all_coords[:, 0, :]
    loc = all_coords[:, 1:, :]
    dist_tensor = torch.cdist(all_coords, all_coords)
    waste = torch.rand(batch_size, size, device=device)
    max_waste = torch.ones(batch_size, device=device)

    batch = {
        "depot": depot,
        "loc": loc,
        "dist": dist_tensor,
        "waste": waste,
        "max_waste": max_waste,
    }

    # Add dummy temporal features if needed
    for i in range(1, temporal_horizon + 1):
        batch[f"fill{i}"] = torch.rand(batch_size, size, device=device)

    return batch


class MyModelWrapper(nn.Module):
    """Wraps a model to conform to the interface expected by loss-landscapes.

    This adapter ensures that the model can be called with a standardized
    forward signature during 3D loss surface perturbations.

    Attributes:
        model (nn.Module): The underlying neural network being wrapped.
    """

    def __init__(self, model):
        """Initializes the wrapper.

        Args:
            model: The neural network model to wrap for visualization.
        """
        super().__init__()
        self.model = model

    def forward(
        self,
        input,
        cost_weights=None,
        return_pi=False,
        pad=False,
        mask=None,
        expert_pi=None,
    ):
        """Forward pass of the model.

        Args:
            input: Input batch data.
            cost_weights: Scaling factors for loss components. Defaults to None.
            return_pi: Whether to return the policy distribution. Defaults to False.
            pad: Whether to pad the input sequences. Defaults to False.
            mask: Optional attention mask. Defaults to None.
            expert_pi: Ground truth policy for comparison. Defaults to None.

        Returns:
            Any: The output of the underlying model.
        """
        return self.model(input, cost_weights, return_pi, pad, mask, expert_pi)


def load_model_instance(model_path, device, size=100, problem_name="wcvrp"):
    """Loads a model for visualization with default architecture parameters.

    Args:
        model_path: Absolute or relative path to the .pt checkpoint.
        device: Device to load the model onto.
        size: Expected problem graph size. Defaults to 100.
        problem_name: Name of the problem environment. Defaults to 'wcvrp'.

    Returns:
        nn.Module: The instantiated and loaded AttentionModel.
    """
    # This is a bit brittle as it assumes specific model args.
    # Ideally should load args from checkpoint or args.json
    problem = load_problem(problem_name)
    factory = AttentionComponentFactory()
    model = AttentionModel(
        embed_dim=128,
        hidden_dim=512,
        problem=problem,
        component_factory=factory,
        n_encode_layers=3,
        mask_inner=True,
        mask_logits=True,
        normalization="instance",
        tanh_clipping=10.0,
        checkpoint_encoder=False,
        shrink_size=None,
        n_heads=8,
        n_encode_sublayers=1,
        n_decode_layers=2,
        predictor_layers=2,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # Handle cases where checkpoint might be nested
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model
