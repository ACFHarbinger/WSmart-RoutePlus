"""Vectorised (training-time) expert-policy configurations."""

from .aco import ACOConfig
from .alns import VectorALNSConfig
from .hgs import VectorHGSConfig
from .hgs_alns import VectorHGSALNSConfig
from .hvpl import VectorHVPLConfig
from .ils import VectorILSConfig
from .profitable_nn import ProfitableNNConfig
from .rls import RLSConfig

# Hydra _target_ names used by existing checkpoints.
ALNSConfig = VectorALNSConfig
HGSConfig = VectorHGSConfig
ILSConfig = VectorILSConfig
HGSALNSConfig = VectorHGSALNSConfig
HVPLConfig = VectorHVPLConfig

__all__ = [
    "VectorHGSConfig",
    "VectorHVPLConfig",
    "VectorALNSConfig",
    "VectorILSConfig",
    "RLSConfig",
    "ACOConfig",
    "VectorHGSALNSConfig",
    "ProfitableNNConfig",
    "ALNSConfig",
    "HGSConfig",
    "ILSConfig",
    "HGSALNSConfig",
    "HVPLConfig",
]
