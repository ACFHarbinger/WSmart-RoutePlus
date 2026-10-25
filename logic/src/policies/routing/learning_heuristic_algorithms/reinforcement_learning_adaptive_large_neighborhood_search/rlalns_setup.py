"""Destroy/repair operator wiring and RL agent construction for RL-ALNS, mixed into RLALNSSolver as RLALNSSetupMixin."""

from logic.src.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.learning.agents.bandits import (
    DiscountedUCBBandit as DiscountedUCBAgent,
)
from logic.src.policies.helpers.learning.agents.bandits import (
    EXP3Agent,
)
from logic.src.policies.helpers.learning.agents.bandits import (
    SlidingWindowUCBBandit as SlidingWindowUCBAgent,
)
from logic.src.policies.helpers.learning.agents.bandits import (
    ThompsonSamplingBandit as ThompsonSamplingAgent,
)
from logic.src.policies.helpers.learning.agents.bandits import (
    UCBBandit as UCB1Agent,
)
from logic.src.policies.helpers.learning.agents.td_learning import (
    ExpectedSarsaAgent as ExpectedSARSAAgent,
)
from logic.src.policies.helpers.learning.agents.td_learning import (
    QLearningAgent,
)
from logic.src.policies.helpers.learning.agents.td_learning import (
    SarsaAgent as SARSAAgent,
)
from logic.src.policies.helpers.operators import (
    cluster_removal,
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    shaw_profit_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.destroy_ruin.shaw import shaw_removal
from logic.src.policies.helpers.operators.destroy_ruin.string import string_removal


class RLALNSSetupMixin:
    def _init_operators(self) -> None:
        """
        Initialize the portfolio of destroy and repair operators.

        Sets up the destroy_ops and repair_ops lists with lambda wrappers
        that call the underlying operator functions with appropriate arguments.
        Provides a mapping for action indices to operator combinations.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        if use_profit:
            self.destroy_ops = [
                lambda r, n: random_removal(r, n, rng=self.random),
                lambda r, n: worst_profit_removal(r, n, self.dist_matrix, self.wastes, self.R, self.C),
                lambda r, n: cluster_removal(r, n, self.dist_matrix, self.nodes, rng=self.random),
                lambda r, n: shaw_profit_removal(r, n, self.dist_matrix, self.wastes, self.R, self.C),
                lambda r, n: string_removal(r, n, self.dist_matrix, rng=self.random),
            ]
        else:
            self.destroy_ops = [
                lambda r, n: random_removal(r, n, rng=self.random),
                lambda r, n: worst_removal(r, n, self.dist_matrix),
                lambda r, n: cluster_removal(r, n, self.dist_matrix, self.nodes, rng=self.random),
                lambda r, n: shaw_removal(r, n, self.dist_matrix),
                lambda r, n: string_removal(r, n, self.dist_matrix, rng=self.random),
            ]

        self.destroy_names = ["Random", "Worst", "Cluster", "Shaw", "String"]

        if use_profit:
            self.repair_ops = [
                lambda r, n: greedy_profit_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                ),
                lambda r, n: regret_2_profit_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                ),
            ]
        else:
            self.repair_ops = [
                lambda r, n: greedy_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                ),
                lambda r, n: regret_2_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                ),
            ]
        self.repair_names = ["Greedy", "Regret-2"]

        # Calculate total number of possible (destroy, repair) combinations
        self.n_actions = len(self.destroy_ops) * len(self.repair_ops)

    def _create_rl_agent(self):
        """
        Factory method to create the appropriate RL agent based on configuration.

        Supports a wide variety of MAB and TD-learning agents through a unified interface.

        Returns:
            RLAgent: An instance of a bandit or TD-learning agent.

        Raises:
            ValueError: If the configured algorithm is not recognized.
        """
        algo = self.params.rl_algorithm.lower()
        seed = int(self.np_random.integers(1000000))

        if algo == "ucb1":
            return UCB1Agent(n_arms=self.n_actions, c=self.params.ucb_c, seed=seed)

        elif algo == "discounted_ucb":
            return DiscountedUCBAgent(
                n_arms=self.n_actions,
                c=self.params.ucb_c,
                gamma=self.params.ucb_gamma,
                seed=seed,
            )

        elif algo == "sliding_window_ucb":
            return SlidingWindowUCBAgent(
                n_arms=self.n_actions,
                window_size=self.params.ucb_window_size,
                c=self.params.ucb_c,
                seed=seed,
            )

        elif algo == "thompson_sampling":
            return ThompsonSamplingAgent(
                n_arms=self.n_actions,
                alpha_prior=self.params.ts_alpha_prior,
                beta_prior=self.params.ts_beta_prior,
                seed=seed,
            )

        elif algo == "exp3":
            return EXP3Agent(n_arms=self.n_actions, gamma=self.params.exp3_gamma, seed=seed)

        elif algo == "q_learning":
            n_states = 27  # 3x3x3 discretized states
            return QLearningAgent(
                n_states=n_states,
                n_actions=self.n_actions,
                alpha=self.params.alpha,
                gamma=self.params.gamma,
                epsilon=self.params.epsilon,
                epsilon_decay=self.params.epsilon_decay,
                epsilon_min=self.params.epsilon_min,
                seed=seed,
            )

        elif algo == "sarsa":
            n_states = 27
            return SARSAAgent(
                n_states=n_states,
                n_actions=self.n_actions,
                alpha=self.params.alpha,
                gamma=self.params.gamma,
                epsilon=self.params.epsilon,
                epsilon_decay=self.params.epsilon_decay,
                epsilon_min=self.params.epsilon_min,
                seed=seed,
            )

        elif algo == "expected_sarsa":
            n_states = 27
            return ExpectedSARSAAgent(
                n_states=n_states,
                n_actions=self.n_actions,
                alpha=self.params.alpha,
                gamma=self.params.gamma,
                epsilon=self.params.epsilon,
                epsilon_decay=self.params.epsilon_decay,
                epsilon_min=self.params.epsilon_min,
                seed=seed,
            )

        else:
            raise ValueError(f"Unknown RL algorithm: {algo}")
