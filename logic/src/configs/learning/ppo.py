"""PPO specific configuration.

Attributes:
    PPOConfig: Configuration for PPO algorithm.

Example:
    ppo_config = PPOConfig(
        epochs=10,
        eps_clip=0.2,
        value_loss_weight=0.5,
        mini_batch_size=0.25,
    )
"""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO specific configuration.

    Attributes:
        inner_epochs: Number of epochs to train.
        eps_clip: Epsilon value for the algorithm.
        value_loss_weight: Coefficient for value loss.
        mini_batch_size: Fraction of the batch size to use for each mini-batch.
    """

    inner_epochs: int = 10
    eps_clip: float = 0.2
    value_loss_weight: float = 0.5
    mini_batch_size: float = 0.25
