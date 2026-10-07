"""Manager (Neural) selection strategy.

This module provides a neural network-based selection strategy that wraps a
trained MandatoryManager to make collection decisions based on learned
temporal patterns and spatial context.

Attributes:
    ManagerSelector: Neural network-based selection using MandatoryManager.

Example:
    >>> selector = ManagerSelector(threshold=0.5)
    >>> mask = selector.select(fill_levels, locs=locs, waste_history=history)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.models.meta.hrl_manager import MandatoryManager

from .base import VectorizedSelector


class ManagerSelector(VectorizedSelector):
    """Neural network-based selection using MandatoryManager.

    This selector wraps a trained HRL manager to make mandatory collection
    decisions. The manager learns to predict which bins require collection
    based on waste history, spatial distribution, and critical thresholds.

    Attributes:
        threshold: Probability threshold for mandatory decisions.
        device: Computation device.
        manager: The underlying neural manager model.
    """

    def __init__(
        self,
        manager: Optional[MandatoryManager] = None,
        manager_config: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        """Initialize the manager selector.

        Args:
            manager: Pre-instantiated MandatoryManager.
            manager_config: Configuration for creating a manager if not provided.
            threshold: Probability threshold for mandatory decisions.
            device: Computation device ('cpu' or 'cuda').
        """
        self.threshold = threshold
        self.device = device

        if manager is not None:
            self.manager = manager
        else:
            config = manager_config or {}
            self.manager = MandatoryManager(
                hidden_dim=config.get("hidden_dim", 128),
                lstm_hidden=config.get("lstm_hidden", 64),
                input_dim_dynamic=config.get("history_length", 10),
                critical_threshold=config.get("critical_threshold", 0.9),
                device=device,
            )

    def select(
        self,
        fill_levels: torch.Tensor,
        locs: Optional[torch.Tensor] = None,
        waste_history: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select bins using the neural manager.

        Args:
            fill_levels: Current fill levels of bins [B, N].
            locs: Spatial coordinates of nodes [B, N, 2].
            waste_history: Temporal fill history [B, N, T].
            threshold: Override for the default selection threshold.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Boolean mask [B, N] where True indicates collection.
        """
        batch_size, num_nodes = fill_levels.shape
        device = fill_levels.device
        thresh = threshold if threshold is not None else self.threshold

        # Ensure manager is on correct device
        if next(self.manager.parameters()).device != device:
            self.manager = self.manager.to(device)

        # Prepare static features (locations)
        if locs is None:
            locs = torch.zeros(batch_size, num_nodes, 2, device=device)

        # Prepare dynamic features (waste history)
        if waste_history is not None:
            dynamic = waste_history
        else:
            # Expand current fill levels as constant history
            history_len = self.manager.input_dim_dynamic
            dynamic = fill_levels.unsqueeze(-1).expand(-1, -1, history_len)

        # Prepare global features
        current_waste = fill_levels
        critical_mask = (current_waste > self.manager.critical_threshold).float()
        critical_ratio = critical_mask.mean(dim=1, keepdim=True)
        max_waste = current_waste.max(dim=1, keepdim=True)[0]
        global_features = torch.cat([critical_ratio, max_waste], dim=1)

        # Get mandatory mask from manager
        mandatory = self.manager.get_mandatory_mask(locs, dynamic, global_features, threshold=thresh)

        return mandatory

    def load_weights(self, path: str) -> None:
        """Load manager weights from checkpoint.

        Args:
            path: Path to the weight checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "manager_state_dict" in checkpoint:
            self.manager.load_state_dict(checkpoint["manager_state_dict"])
        elif "state_dict" in checkpoint:
            self.manager.load_state_dict(checkpoint["state_dict"])
        else:
            self.manager.load_state_dict(checkpoint)
