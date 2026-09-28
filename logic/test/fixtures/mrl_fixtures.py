"""
Fixtures for Meta-Reinforcement Learning (MRL) and Hyperparameter Optimization.
"""

import pytest
import torch
from logic.src.pipeline.rl.meta.contextual_bandits import (
    WeightContextualBandit,
)
from logic.src.pipeline.rl.meta.multi_objective.weight_optimizer import (
    MORLWeightOptimizer,
)
from logic.src.pipeline.rl.meta.td_learning import (
    CostWeightManager,
)
from logic.src.pipeline.rl.meta.weight_optimizer import (
    RewardWeightOptimizer,
)


@pytest.fixture
def hpo_opts():
    """Provide standard HPO options."""
    return {
        "problem": "wcvrp",
        "graph_size": 20,
        "save_dir": "test_save_dir",
        "load_path": "test_load_path",
        "resume": False,
        "val_size": 10,
        "val_dataset": None,
        "area": "Rio Maior",
        "waste_type": "test_waste",
        "distance_method": "test_dist",
        "data_distribution": "test_dist",
        "vertex_method": "test_vertex",
        "edge_threshold": 10,
        "edge_method": "test_edge",
        "focus_graph": None,
        "dm_filepath": None,
        "no_cuda": True,
        "train_time": False,
        "env.graph.start_day": 0,
        "train.env.graph.n_days": 1,
        "eval_batch_size": 2,
        "hop_epochs": 1,
        "eta": 3,
        "indpb": 0.1,
        "tournsize": 3,
        "n_pop": 5,
        "cxpb": 0.5,
        "mutpb": 0.2,
        "n_gen": 1,
        "verbose": 0,
        "n_startup_trials": 1,
        "n_warmup_steps": 1,
        "interval_steps": 1,
        "logging": {"log_output": False},
        "seed": 1234,
        "run_name": "test_run",
        "metric": "loss",
        "cpu_cores": 1,
        "log_dir": "test_log_dir",
        "num_samples": 1,
        "max_tres": 1,
        "max_conc": 1,
        "no_tensorboard": True,
        "device": "cpu",
        "model": "am",
        "temporal_horizon": 0,
        "encoder": "gat",
    }


@pytest.fixture
def bandit_setup():
    """Setup WeightContextualBandit instance."""
    initial_weights = {"w_waste": 1.0, "w_over": 1.0}
    weight_ranges = {"w_waste": (0.1, 5.0), "w_over": (0.1, 5.0)}
    context_features = ["waste", "overflow", "day"]
    dummy_dist_matrix = torch.rand(10, 2)

    bandit = WeightContextualBandit(
        num_days=10,
        distance_matrix=dummy_dist_matrix,
        initial_weights=initial_weights,
        context_features=context_features,
        features_aggregation="avg",
        exploration_strategy="epsilon_greedy",
        exploration_factor=0.5,
        num_weight_configs=5,
        weight_ranges=weight_ranges,
        window_size=5,
    )
    return bandit


@pytest.fixture
def morl_setup():
    """Setup MORLWeightOptimizer instance."""
    initial_weights = {"w_waste": 1.0, "w_over": 1.0, "w_len": 1.0}
    optimizer = MORLWeightOptimizer(
        initial_weights=initial_weights,
        weight_names=["w_waste", "w_over", "w_len"],
        objective_names=["waste_efficiency", "overflow_rate"],
        weight_ranges=[0.01, 5.0],
        history_window=10,
        exploration_factor=0.2,
        adaptation_rate=0.1,
    )
    return optimizer


@pytest.fixture
def cwm_setup():
    """Setup CostWeightManager instance."""
    initial_weights = {"waste": 1.0, "over": 1.0, "len": 1.0}
    manager = CostWeightManager(
        initial_weights=initial_weights,
        learning_rate=0.1,
        decay_rate=0.9,
        weight_ranges=[0.1, 5.0],
        window_size=5,
    )
    return manager


@pytest.fixture
def rwo_setup():
    """Setup RewardWeightOptimizer instance."""

    # Helper mock class for RWO
    class MockModel(torch.nn.Module):
        """Mock Neural Model for RewardWeightOptimizer."""

        def __init__(self, input_size, hidden_size, output_size):
            """Initialize mock model."""
            super().__init__()
            self.layer = torch.nn.Linear(input_size, output_size)

        def forward(self, x):
            """Mock forward pass."""
            return self.layer(x[:, -1, :]), None

    initial_weights = {"w1": 1.0, "w2": 1.0}

    optimizer = RewardWeightOptimizer(
        model_class=MockModel,              # type: ignore[bad-argument-type]
        initial_weights=initial_weights,
        history_length=5,
        hidden_size=10,
        lr=0.01,
        device="cpu",
        meta_batch_size=2,
        min_weights=[0.1, 0.1],
        max_weights=[5.0, 5.0],
        meta_optimizer="adam",
    )
    return optimizer
