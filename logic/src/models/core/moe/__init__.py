"""Mixture of Experts (MoE) architectures and policies.

This package provides models that leverage sparse expert-routing to scale
model capacity. It includes MoE-enabled versions of the standard and temporal
attention models.

Attributes:
    MoEAttentionModel: Static routing model with expert encoder.
    MoETemporalAttentionModel: Time-varying model with expert encoder.
    MoEPolicy: Routing policy using MoE encoding.

"""

from .moe_attention_model import MoEAttentionModel as MoEAttentionModel
from .moe_temporal_attention_model import (
    MoETemporalAttentionModel as MoETemporalAttentionModel,
)
from .policy import MoEPolicy as MoEPolicy

__all__ = ["MoEAttentionModel", "MoETemporalAttentionModel", "MoEPolicy"]
