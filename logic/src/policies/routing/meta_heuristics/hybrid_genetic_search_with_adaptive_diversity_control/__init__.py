"""
Hybrid Genetic Search with Adaptive Diversity Control (HGS-ADC) matheuristic package.

Attributes:
    PolicyHGSADC: The main policy class.
    HGSADCParams: Parameters for HGS-ADC.
    policy_hgs_adc: The policy module.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control import PolicyHGSADC
    >>> policy = PolicyHGSADC(config)
"""

from . import policy_hgs_adc as policy_hgs_adc
from .params import HGSADCParams as HGSADCParams
from .policy_hgs_adc import PolicyHGSADC as PolicyHGSADC

__all__ = ["PolicyHGSADC", "HGSADCParams", "policy_hgs_adc"]
