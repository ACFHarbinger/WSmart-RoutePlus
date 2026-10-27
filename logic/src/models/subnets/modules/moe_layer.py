"""Mixture of Experts (MoE) layer.

This module provides the MoE layer, which implements sparsely-gated Mixture-of-Experts.
It dynamically selects a subset of expert networks for each input token/node,
allowing for massive parameter scaling while keeping per-token compute constant.

Attributes:
    MoE: Sparsely gated mixture of experts layer.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.moe_layer import MoE
    >>> moe = MoE(input_size=128, output_size=128, num_experts=4, k=2)
    >>> x = torch.randn(1, 10, 128)
    >>> out = moe(x)
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, cast

import torch
from torch import nn
from torch.distributions.normal import Normal

from .moe_dispatcher import SparseDispatcher


class MoE(nn.Module):
    """Sparsely gated Mixture-of-Experts layer with adaptive expert selection.

    Routes input tokens through a gating network to the 'best' `k` experts out
    of a total pool. This allows for significantly increasing model capacity (total
    parameters) without a proportional increase in active computation.

    Attributes:
        noisy_gating (bool): Whether to inject noise into gating logits for load balance.
        num_experts (int): Total number of parallel expert sub-networks.
        output_size (int): Dimensionality of the resulting output features.
        input_size (int): Dimensionality of the input feature space.
        k (int): Exact number of experts to activate for each input sample.
        experts (nn.ModuleList): The collection of expert modules (MLPs or custom).
        w_gate (nn.Parameter): Learnable weights for the master routing gate.
        w_noise (nn.Parameter): Learnable weights for context-aware noise scaling.
        softplus (nn.Softplus): Activation for ensuring positive noise standard deviation.
        softmax (nn.Softmax): Activation for converting logits to routing weights.
        mean (torch.Tensor): Registered buffer for noise distribution mean.
        std (torch.Tensor): Registered buffer for noise distribution standard deviation.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_neurons: Optional[list] = None,
        experts: Optional[nn.ModuleList | list] = None,
        hidden_act: str = "ReLU",
        out_bias: bool = True,
        num_experts: int = 4,
        k: int = 2,
        noisy_gating: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the MoE layer.

        Args:
            input_size: Dimension of incoming individual feature vectors.
            output_size: Target dimension for the combined expert outputs.
            num_neurons: Optional hidden depth/width for generated MLP experts.
            experts: Optional pre-built sequence of expert modules.
            hidden_act: Activation function used in generated MLP experts.
            out_bias: Whether to use bias in the final layer of each expert.
            num_experts: Total count of experts to initialize or manage.
            k: Exact number of top experts to activate per time step.
            noisy_gating: Whether to use stochastic top-k selection for exploration.
            kwargs: Any additional configuration for expert builders.
        """
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k

        # Instantiate experts
        if experts is not None:
            self.experts = experts if isinstance(experts, nn.ModuleList) else nn.ModuleList(experts)
        elif num_neurons is not None and len(num_neurons) > 0:

            def build_mlp(dims: list) -> nn.Sequential:
                """Construct a multi-layer perceptron for an expert.

                Args:
                    dims: List of hidden dimensions.

                Returns:
                    Sequential model representing the MLP.
                """
                layers = []
                in_dim = input_size
                for hidden in dims:
                    layers.append(nn.Linear(in_dim, hidden))
                    layers.append(getattr(nn, hidden_act)())
                    in_dim = hidden
                layers.append(nn.Linear(in_dim, output_size, bias=out_bias))
                return nn.Sequential(*layers)

            self.experts = nn.ModuleList([build_mlp(num_neurons) for _ in range(self.num_experts)])
        else:
            self.experts = nn.ModuleList(
                [nn.Linear(self.input_size, self.output_size, bias=out_bias) for _ in range(self.num_experts)]
            )

        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates the squared coefficient of variation.

        This metric is used as a regularization loss to encourage uniform expert
        usage (load balancing) across the batch.

        Args:
            x: A positive-valued tensor representing counts or loads.

        Returns:
            torch.Tensor: Scalar coefficient of variation squared.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates: torch.Tensor) -> torch.Tensor:
        """Determines the actual activation frequency (load) for each expert.

        Args:
            gates: Sparse gating tensor of shape (batch, num_experts).

        Returns:
            torch.Tensor: Integer load counts per expert.
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self,
        clean_values: torch.Tensor,
        noisy_values: torch.Tensor,
        noise_stddev: torch.Tensor,
        noisy_top_values: torch.Tensor,
    ) -> torch.Tensor:
        """Estimates the probability of inclusion in top-k under noise.

        Enables gradient flow safely through the non-differentiable top-k operation
        by considering the expected inclusion probability.

        Args:
            clean_values: Raw routing logits (batch, num_experts).
            noisy_values: Logits with injected stochastic noise.
            noise_stddev: Contextual noise standard deviation.
            noisy_top_values: Top-k values identified in the current noisy pass.

        Returns:
            torch.Tensor: Probability tensor of shape (batch, num_experts).
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(cast(torch.Tensor, self.mean), cast(torch.Tensor, self.std))
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(
        self, x: torch.Tensor, train: bool, noise_epsilon: float = 1e-2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates routing gates using the noisy top-k mechanism.

        Based on: Shazeer et al. (2017) "Outrageously Large Neural Networks".

        Args:
            x: Flattened input sequence of shape (total_tokens, input_size).
            train: Whether to enable stochasticity (noise injection).
            noise_epsilon: Small stability constant for noise standard deviation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - gates: Sparse weights of shape (total_tokens, num_experts).
                - load: Estimated expert load for balancing losses.
        """
        clean_logits = x @ self.w_gate
        noisy_logits = None
        noise_stddev = None
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # Get topk + 1 for inclusion probability logic
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # Normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)  # Non-topk elements are 0

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits,
                    cast(torch.Tensor, noisy_logits),
                    cast(torch.Tensor, noise_stddev),
                    top_logits,
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x: torch.Tensor, loss_coef: float = 0.0) -> torch.Tensor:
        """Processes the input through the mixture of experts.

        Args:
            x: Sequence or batch tensor of shape (..., input_size).
            loss_coef: Scaling factor for load-balancing auxiliary loss.

        Returns:
            torch.Tensor: Combined expert output of shape (..., output_size).
        """
        output_shape = list(x.size()[:-1]) + [self.output_size]
        x = x.reshape(-1, self.input_size) if x.dim() != 2 else x

        gates, load = self.noisy_top_k_gating(x, self.training)

        # Gating regularization: encourage uniform routing
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        # Efficiently route tokens to their respective experts
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)

        return y.reshape(output_shape)
