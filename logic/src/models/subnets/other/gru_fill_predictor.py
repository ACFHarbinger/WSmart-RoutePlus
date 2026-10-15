"""Gated Recurrent Fill Predictor.

Attributes:
    GatedRecurrentUnitFillPredictor: Predicts bin fill levels using a GRU-based recurrent network.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.helpers.gru_fill_predictor import GatedRecurrentUnitFillPredictor
    >>> model = GatedRecurrentUnitFillPredictor(input_dim=1, hidden_dim=32)
    >>> history = torch.randn(1, 10, 1)  # (batch, seq, feat)
    >>> prediction = model(history)
"""

from typing import List, Optional, Tuple, cast

import torch
from torch import nn

from logic.src.models.subnets.modules import ActivationFunction


class GatedRecurrentUnitFillPredictor(nn.Module):
    """Predicts bin fill levels using a GRU-based recurrent network.

    Attributes:
        hidden_dim (int): Hidden state dimension.
        num_layers (int): Number of GRU layers.
        bidirectional (bool): Whether GRU is bidirectional.
        gru (nn.GRU): Recurrent neural network module.
        predictor (nn.Sequential): MLP for final prediction.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        af_param: float = 1.0,
        threshold: float = 6.0,
        replacement_value: float = 6.0,
        n_params: int = 3,
        uniform_range: Optional[List[float]] = None,
        bidirectional: bool = False,
    ):
        """Initializes the GRF Predictor.

        Args:
            input_dim: Input dimension size. Defaults to 1.
            hidden_dim: Hidden state dimension. Defaults to 64.
            num_layers: Number of GRU layers. Defaults to 2.
            dropout: Dropout probability. Defaults to 0.1.
            activation: Activation function for predictor MLP. Defaults to 'relu'.
            af_param: Activation parameter. Defaults to 1.0.
            threshold: Activation threshold. Defaults to 6.0.
            replacement_value: Replacement value. Defaults to 6.0.
            n_params: Number of parameters. Defaults to 3.
            uniform_range: Range for uniform distribution. Defaults to None.
            bidirectional: Whether GRU is bidirectional. Defaults to False.
        """
        if uniform_range is None:
            uniform_range = [0.125, 1 / 3]
        super(GatedRecurrentUnitFillPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Output dimension will be doubled if bidirectional
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            ActivationFunction(
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                cast(Tuple[float, float], tuple(uniform_range)) if uniform_range else None,
            ),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fill_history: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict next fill level.

        Args:
            fill_history: Sequence of historical fill levels of shape (batch, seq, feat).

        Returns:
            torch.Tensor: Predicted next fill level of shape (batch, 1).
        """
        output, hidden = self.gru(fill_history)
        last_hidden = output[:, -1, :]
        predicted_fill = self.predictor(last_hidden)
        return predicted_fill

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes the hidden state.

        Args:
            batch_size: Size of the input batch.
            device: Device to place the hidden state on.

        Returns:
            torch.Tensor: Initial hidden state of shape (num_layers * num_directions, batch_size, hidden_dim).
        """
        # (num_layers * num_directions, batch_size, hidden_dim)
        directions = 2 if self.bidirectional else 1
        return torch.zeros(self.num_layers * directions, batch_size, self.hidden_dim).to(device)
