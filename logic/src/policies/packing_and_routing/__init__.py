"""Packing and Routing Policies Package.

This package contains algorithms that perform **joint** mandatory-bin selection
and routing in a single integrated optimization loop, rather than
executing the two phases sequentially.

The package exposes the main factory and registry for joint policies, as well
as the specific implementations for Greedy Orienteering, Simulated Annealing,
and NDS-BRKGA.

Attributes:
    BaseJointPolicy: Base class for joint policies.
    JointPolicyFactory: Factory for creating joint policies.
    JointPolicyRegistry: Registry of available joint policies.
    JointGreedyParams: Parameters for JGO.
    JointGreedyPolicy: JGO policy implementation.
    JointSAParams: Parameters for JSA.
    JointSAPolicy: JSA policy implementation.
    NDSBRKGAParams: Parameters for NDS-BRKGA.
    NDSBRKGAPolicy: NDS-BRKGA policy implementation.

Example:
    >>> from logic.src.policies.packing_and_routing import JointPolicyFactory
    >>> policy = JointPolicyFactory.create("nds_brkga", config=cfg)
"""

from .base.base_joint_policy import BaseJointPolicy
from .base.factory import JointPolicyFactory
from .base.registry import JointPolicyRegistry
from .joint_greedy_orienteering import JointGreedyParams, JointGreedyPolicy
from .joint_simulated_annealing import JointSAParams, JointSAPolicy
from .non_dominated_sorting_biased_random_key_genetic_algorithm import (
    NDSBRKGAParams,
    NDSBRKGAPolicy,
)

__all__ = [
    "BaseJointPolicy",
    "JointPolicyRegistry",
    "JointPolicyFactory",
    "NDSBRKGAPolicy",
    "NDSBRKGAParams",
    "JointSAPolicy",
    "JointSAParams",
    "JointGreedyPolicy",
    "JointGreedyParams",
]
