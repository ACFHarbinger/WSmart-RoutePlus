"""
Hybrid Genetic Search with Adaptive Diversity Control (HGS-ADC) matheuristic package.

Attributes:
    PolicyHGSADC: The main policy class.
    HGSADCConfig: Parameters for HGS-ADC.
    policy_hgs_adc: The policy module.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control import PolicyHGSADC
    >>> policy = PolicyHGSADC(config)
"""

from logic.src.configs.policies.hgs_adc import HGSADCConfig as HGSADCConfig

from . import policy_hgs_adc as policy_hgs_adc
from .policy_hgs_adc import PolicyHGSADC as PolicyHGSADC

__all__ = ["PolicyHGSADC", "HGSADCConfig", "policy_hgs_adc"]
