"""Joint Greedy Orienteering Package.

Implements a joint mandatory-packing and routing algorithm
based on greedy orienteering heuristics.

Attributes:
    JointGreedyParams (Type[JointGreedyParams]): Configuration for the greedy policy.
    JointGreedyPolicy (Type[JointGreedyPolicy]): The joint greedy policy implementation.

Example:
    >>> from logic.src.policies.packing_and_routing.joint_greedy_orienteering import JointGreedyPolicy
    >>> policy = JointGreedyPolicy()
"""

from .params import JointGreedyParams
from .policy_jgo import JointGreedyPolicy

__all__ = ["JointGreedyParams", "JointGreedyPolicy"]
