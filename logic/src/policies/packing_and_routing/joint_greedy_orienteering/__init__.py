"""Joint Greedy Orienteering Package.

Implements a joint mandatory-packing and routing algorithm
based on greedy orienteering heuristics.

Attributes:
    JointGreedyConfig (Type[JointGreedyConfig]): Configuration for the greedy policy.
    JointGreedyPolicy (Type[JointGreedyPolicy]): The joint greedy policy implementation.

Example:
    >>> from logic.src.policies.packing_and_routing.joint_greedy_orienteering import JointGreedyPolicy
    >>> policy = JointGreedyPolicy()
"""

from logic.src.configs.policies.jgo import JointGreedyConfig
from .policy_jgo import JointGreedyPolicy

__all__ = ["JointGreedyConfig", "JointGreedyPolicy"]
