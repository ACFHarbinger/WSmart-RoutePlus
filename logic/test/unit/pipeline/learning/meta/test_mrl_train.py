"""Tests for Meta-Reinforcement Learning training pipelines."""

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestWeightContextualBandit:
    """Tests for the Weight Contextual Bandit meta-learner."""

    @pytest.mark.unit
    def test_initialization(self, bandit_setup):
        """Test initialization of bandit parameters."""
        bandit = bandit_setup
        assert bandit.num_configs == 5
        assert bandit.exploration_strategy == "epsilon_greedy"
        assert len(bandit.weight_configs) == 5
        # Ensure initial weights are included
        assert {"w_waste": 1.0, "w_over": 1.0} in bandit.weight_configs

    @pytest.mark.unit
    def test_generate_weight_configs(self, bandit_setup):
        """Test generation of diverse weight configurations."""
        bandit = bandit_setup
        configs = bandit.weight_configs
        assert len(configs) == 5
        for config in configs:
            assert "w_waste" in config
            assert "w_over" in config
            # Check bounding
            assert 0.1 <= config["w_waste"] <= 5.0

    @pytest.mark.unit
    def test_get_context_features(self, bandit_setup):
        """Test extraction of context features from dataset."""
        bandit = bandit_setup

        # Mocking a dataset with instances
        # instance['waste'] -> tensor, 'max_waste' -> tensor
        class MockInstance(dict):
            """Mock dictionary-like instance."""
            pass

        mock_data = []
        for _ in range(5):
            inst = MockInstance()
            inst["waste"] = torch.tensor(50.0)
            inst["max_waste"] = torch.tensor(100.0)
            mock_data.append(inst)

        class MockDataset:
            """Mock dataset class."""

            def __init__(self, data):
                """Initialize mock dataset."""
                self.data = data

            def __getitem__(self, idx):
                """Get item from dataset."""
                return self.data[idx]

            def __len__(self):
                """Return length of dataset."""
                return len(self.data)

        dataset = MockDataset(mock_data)

        # We need to set max_feat_values first to avoid div by zero or similar logic issues in categorization later
        # But _get_context_features returns raw values
        context = bandit._get_context_features(dataset)

        assert "avg_waste" in context
        assert "avg_overflow" in context
        assert "day" in context
        assert context["avg_waste"] == 50.0

    @pytest.mark.unit
    def test_get_current_weights_epsilon_greedy(self, bandit_setup):
        """Test selection of current weights using epsilon-greedy strategy."""
        bandit = bandit_setup
        bandit.set_max_feature_values({"avg_waste": 100.0, "avg_overflow": 1.0})

        # Mock dataset
        class MockDataset:
            """Mock dataset with dummy data."""

            data = [{"waste": torch.tensor(50.0), "max_waste": torch.tensor(100.0)}] * 5

        dataset = MockDataset()

        # Force exploration (epsilon = 1.0 is not directly settable in init easily unless we hack it,
        # but 0.5 is high enough to hit random eventually, or we force it)
        bandit.exploration_factor = 1.0

        weights = bandit.get_current_weights(dataset)
        assert isinstance(weights, dict)
        assert "w_waste" in weights

        # Check history update
        assert len(bandit.history) == 1
        assert bandit.history[0]["selected_config"] == weights

    @pytest.mark.unit
    def test_update_rewards(self, bandit_setup):
        """Test reward update mechanism."""
        bandit = bandit_setup
        reward = 10.0
        context = {"avg_waste": 50.0, "day": 1}
        bandit.set_max_feature_values({"avg_waste": 100.0})

        # Pre-set a config index to update
        bandit.current_config_idx = 0

        stats = bandit.update(reward, {}, context=context)

        assert stats["trials"][0] == 1
        assert bandit.total_trials == 1

        # Verify context reward storage

        context_key = bandit._context_to_key(context)
        assert bandit.context_rewards[context_key][0] == [10.0]


class TestMORLWeightOptimizer:
    """Tests for Multi-Objective RL weight optimizer."""

    @pytest.mark.unit
    def test_initialization(self, morl_setup):
        """Test initialization of MORL optimizer."""
        optimizer = morl_setup
        assert len(optimizer.current_weights) == 3
        assert optimizer.history_window == 10
        assert len(optimizer.performance_history) == 0

    @pytest.mark.unit
    def test_metrics_calculation(self, morl_setup):
        """Test calculation of performance metrics."""
        optimizer = morl_setup
        metrics = {
            "waste_collected": 200.0,
            "tour_length": 100.0,
            "num_overflows": 5,
            "total_bins": 50,
        }

        objs = optimizer._calculate_objectives(metrics)
        assert objs["waste_efficiency"] == 2.0  # 200/100
        assert objs["overflow_rate"] == 0.1  # 5/50

    @pytest.mark.unit
    def test_update_performance_history(self, morl_setup):
        """Test updating of performance history buffers."""
        optimizer = morl_setup
        metrics = {
            "waste_collected": 200.0,
            "tour_length": 100.0,
            "num_overflows": 5,
            "total_bins": 50,
        }
        reward = 10.0

        optimizer.update_performance_history(metrics, reward)

        assert len(optimizer.performance_history) == 1
        assert len(optimizer.pareto_front.solutions) == 1

        # Check values
        sol = optimizer.pareto_front.solutions[0]
        assert sol.reward == 10.0
        assert sol.objectives["waste_efficiency"] == 2.0

    @pytest.mark.unit
    def test_pareto_front_dominance(self, morl_setup):
        """Test Pareto dominance logic."""
        optimizer = morl_setup

        # Solution 1: Reference
        metrics1 = {
            "waste_collected": 100.0,
            "tour_length": 100.0,
            "num_overflows": 10,
            "total_bins": 100,
        }
        # Eff=1.0, Over=0.1
        optimizer.update_performance_history(metrics1, 10.0)

        # Solution 2: Dominated (Worse eff, same overflows)
        metrics2 = {
            "waste_collected": 50.0,
            "tour_length": 100.0,
            "num_overflows": 10,
            "total_bins": 100,
        }
        # Eff=0.5, Over=0.1
        optimizer.update_performance_history(metrics2, 5.0)

        # Only Solution 1 should remain (since 1 dominates 2)

        assert len(optimizer.pareto_front.solutions) == 1
        assert optimizer.pareto_front.solutions[0].reward == 10.0

        # Solution 3: Dominates Sol 1 (Better eff, Less overflows)
        metrics3 = {
            "waste_collected": 200.0,
            "tour_length": 100.0,
            "num_overflows": 5,
            "total_bins": 100,
        }
        # Eff=2.0, Over=0.05
        optimizer.update_performance_history(metrics3, 20.0)

        # Now Sol 3 dominates Sol 1. Sol 1 should be removed, Sol 3 added.
        assert len(optimizer.pareto_front.solutions) == 1
        assert optimizer.pareto_front.solutions[0].reward == 20.0

    @pytest.mark.unit
    def test_update_weights_logic(self, morl_setup):
        """Test weight update logic based on performance."""
        optimizer = morl_setup

        # Force a condition where adjustment is needed (e.g. step check)
        # _need_to_adjust_weights checks history length though.
        # Fill history first
        for _ in range(5):
            metrics = {
                "waste_collected": 100.0,
                "tour_length": 100.0,
                "num_overflows": 10,
                "total_bins": 100,
            }
            optimizer.update_performance_history(metrics, 10.0)

        # Update weights (should trigger exploitation or exploration)
        optimizer.current_weights.copy()

        # Force exploration to be 0 for deterministic adaptation check (or mock random)
        optimizer.exploration_factor = 0.0

        new_weights = optimizer.update_weights(metrics=None, reward=None, day=1, step=5)  # Step 5 triggers modulo check

        assert new_weights is not None
        # Weights might change or stay same depending on trends, but function runs
        assert isinstance(new_weights, dict)


class TestCostWeightManager:
    """Tests for Cost Weight Manager."""

    @pytest.mark.unit
    def test_initialization(self, cwm_setup):
        """Test initialization of cost weight manager."""
        manager = cwm_setup
        assert manager.learning_rate == 0.1
        assert len(manager.weights) == 3

    @pytest.mark.unit
    def test_update_weights_td(self, cwm_setup):
        """Test Time-Difference based weight updates."""
        manager = cwm_setup

        # Step 1: Initial observation to set expected reward
        # Reward = 10. Cost components: High waste (good), Low overflow (good cost?)
        # Wait, cost components values depend on metric.
        # update_cost_weights_td: 'waste' -> positive update if TD>0. Others -> negative update if TD>0.

        cost_components = {"waste": 1.0, "over": 1.0, "len": 1.0}

        # Call 1: Sets expected reward to 10.0
        new_weights = manager.update_weights(10.0, cost_components)
        assert manager.expected_reward == 10.0
        # First step TD error is 0 (10 - 10), so weights shouldn't change
        assert new_weights == {"waste": 1.0, "over": 1.0, "len": 1.0}

        # Call 2: Reward 20.0 (Better). TD error = 20 - 10 = 10.
        # Waste weight should INCREASE (maximize reward).
        # Over/Len weight should DECREASE (minimize cost).
        new_weights_2 = manager.update_weights(20.0, cost_components)

        assert new_weights_2["waste"] > 1.0
        assert new_weights_2["over"] < 1.0
        assert new_weights_2["len"] < 1.0

        # Verify LR decay
        assert manager.learning_rate < 0.1

    @pytest.mark.unit
    def test_weight_bounding(self, cwm_setup):
        """Test weight bounding constraints."""
        manager = cwm_setup
        manager.learning_rate = 100.0  # Huge LR to force bound check

        cost_components = {"waste": 1.0}
        manager.update_weights(10.0, cost_components)
        manager.update_weights(100.0, cost_components)  # TD error massive

        w = manager.get_current_weights()
        assert w["waste"] <= 5.0  # Max bound


class TestRewardWeightOptimizer:
    """Tests for Reward Weight Optimizer."""

    @pytest.mark.unit
    def test_initialization(self, rwo_setup):
        """Test initialization of reward optimizer."""
        optimizer = rwo_setup
        assert optimizer.num_weights == 2
        assert len(optimizer.weight_history) == 0

    @pytest.mark.unit
    def test_update_histories_and_batch_prep(self, rwo_setup):
        """Test history updates and batch preparation."""
        optimizer = rwo_setup

        # Add data
        for i in range(10):
            optimizer.current_weights = torch.tensor([1.0, 1.0])
            perf = [1.0, 2.0, 3.0]  # 3 metrics (2 weights + 1)
            reward = float(i)
            optimizer.update_histories(perf, reward)

        assert len(optimizer.weight_history) == 5  # history_length
        assert len(optimizer.reward_history) == 5

        # Prepare batch
        features, targets = optimizer.prepare_meta_learning_batch()
        assert features is not None
        assert targets is not None
        assert features.shape[0] >= 1  # meta_batch_size might not be full if history is limited
        assert features.shape[1] > 0  # seq_len

    @pytest.mark.unit
    def test_meta_learning_step(self, rwo_setup):
        """Test meta-learning step execution."""
        optimizer = rwo_setup

        # Add sufficient data
        for _i in range(10):
            optimizer.update_histories([1.0, 1.0, 1.0], 1.0)

        loss = optimizer.meta_learning_step()
        assert loss is not None
        assert isinstance(loss, float)
        assert optimizer.meta_step == 1

    @pytest.mark.unit
    def test_recommend_weights(self, rwo_setup):
        """Test weight recommendation logic."""
        optimizer = rwo_setup

        # Not enough history
        w = optimizer.recommend_weights()
        assert torch.equal(w, optimizer.current_weights)

        # Add history
        for _i in range(5):
            optimizer.update_histories([1.0, 1.0, 1.0], 1.0)

        w_new = optimizer.recommend_weights()
        assert w_new.shape == (2,)
        assert w_new.min() >= 0.1
        assert w_new.max() <= 5.0
