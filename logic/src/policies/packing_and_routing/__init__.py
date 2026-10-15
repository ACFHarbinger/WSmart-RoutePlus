"""Joint packing-and-routing policies. Implementations load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

from .base.factory import JointPolicyFactory
from .base.registry import JointPolicyRegistry

_LAZY = {
    "BaseJointPolicy": "logic.src.policies.packing_and_routing.base.base_joint_policy:BaseJointPolicy",
    "JointGreedyParams": "logic.src.policies.packing_and_routing.joint_greedy_orienteering.params:JointGreedyParams",
    "JointGreedyPolicy": "logic.src.policies.packing_and_routing.joint_greedy_orienteering.policy_jgo:JointGreedyPolicy",
    "JointSAParams": "logic.src.policies.packing_and_routing.joint_simulated_annealing.params:JointSAParams",
    "JointSAPolicy": "logic.src.policies.packing_and_routing.joint_simulated_annealing.policy_jsa:JointSAPolicy",
    "NDSBRKGAParams": "logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.params:NDSBRKGAParams",
    "NDSBRKGAPolicy": "logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga:NDSBRKGAPolicy",
}

__all__ = ["JointPolicyFactory", "JointPolicyRegistry", *list(_LAZY)]


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
