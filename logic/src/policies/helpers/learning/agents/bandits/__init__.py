"""Bandit Agents Module.

This module provides various Multi-Armed Bandit (MAB) implementations, including
standard UCB, Epsilon-Greedy, Softmax, Thompson Sampling, and non-stationary
variants like Discounted and Sliding Window UCB.

Attributes:
    BanditAgent: Abstract base class for all bandit agents.
    EpsilonGreedyBandit: Bandit that selects a random arm with probability epsilon.
    UCBBandit: Upper Confidence Bound bandit.
    SoftmaxBandit: Bandit that selects arms based on a softmax distribution of rewards.
    ThompsonSamplingBandit: Bandit using Thompson Sampling (Bayesian approach).
    DiscountedUCBBandit: UCB variant that discounts old rewards.
    SlidingWindowUCBBandit: UCB variant that only considers recent rewards.
    EXP3Agent: Exponential-weight algorithm for exploration and exploitation.

Example:
    >>> from logic.src.policies.helpers.learning.agents.bandits import UCBBandit
    >>> agent = UCBBandit(n_arms=5)
    >>> action = agent.select_arm()
    >>> agent.update(action, reward=1.0)
"""

from .base import BanditAgent
from .epsilon_greedy import EpsilonGreedyBandit
from .exp3 import EXP3Agent
from .softmax import SoftmaxBandit
from .thompson import ThompsonSamplingBandit
from .ucb import DiscountedUCBBandit, SlidingWindowUCBBandit, UCBBandit

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
