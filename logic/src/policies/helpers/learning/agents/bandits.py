"""Multi-Armed Bandit (MAB) agents.

Provides classic multi-armed bandit algorithms for operator selection
without environmental context.

Attributes:
    BanditAgent: Abstract base class for MAB agents.
    EpsilonGreedyBandit: Standard ε-greedy exploration.
    UCBBandit: Upper Confidence Bound (UCB1) implementation.
    SoftmaxBandit: Boltzmann-style action selection.
    ThompsonSamplingBandit: Bayesian posterior sampling.
    DiscountedUCBBandit: UCB for non-stationary environments.
    SlidingWindowUCBBandit: Time-windowed UCB.
    EXP3Agent: Adversarial bandit algorithm.

Example:
    >>> from logic.src.policies.helpers.learning.agents.bandits import UCBBandit
    >>> agent = UCBBandit(n_arms=5)
"""

from .bandits.base import BanditAgent
from .bandits.epsilon_greedy import EpsilonGreedyBandit
from .bandits.exp3 import EXP3Agent
from .bandits.softmax import SoftmaxBandit
from .bandits.thompson import ThompsonSamplingBandit
from .bandits.ucb import DiscountedUCBBandit, SlidingWindowUCBBandit, UCBBandit

__all__ = [
    "BanditAgent",
    "EpsilonGreedyBandit",
    "UCBBandit",
    "SoftmaxBandit",
    "ThompsonSamplingBandit",
    "DiscountedUCBBandit",
    "SlidingWindowUCBBandit",
    "EXP3Agent",
]
