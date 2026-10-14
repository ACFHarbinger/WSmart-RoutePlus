

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def test_rl_alns_solver():
    print("\n=== Testing RLALNSSolver ===")
    from logic.src.policies.routing.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.params import (
        RLALNSParams,
    )
    from logic.src.policies.routing.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver import (
        RLALNSSolver,
    )

    # Mock Data
    dist_matrix = np.array([[0, 10, 20], [10, 0, 15], [20, 15, 0]])
    wastes = {1: 10., 2: 20.}
    capacity = 50.0
    R = 1.0
    C = 1.0

    # Test several algorithms
    algos = ["ucb1", "q_learning", "sarsa"]

    for algo in algos:
        print(f"Testing with algorithm: {algo}")
        from logic.src.configs.policies.other import BanditConfig, RLConfig
        rl_config = RLConfig(
            agent_type="bandit",
            bandit=BanditConfig(algorithm=algo)
        )
        params = RLALNSParams(
            rl_config=rl_config,
            max_iterations=5,
            start_temp=100.0,
            cooling_rate=0.9,
            seed=42
        )

        solver = RLALNSSolver(dist_matrix, wastes, capacity, R, C, params)
        routes, profit, cost = solver.solve()

        print(f"  Result: Routes={routes}, Profit={profit:.2f}, Cost={cost:.2f}")
        assert profit is not None
        assert isinstance(routes, list)

if __name__ == "__main__":
    try:
        test_rl_alns_solver()
        print("\nAll RL smoke tests passed!")
    except Exception as e:
        print(f"\nSmoke test failed: {e}")
        import traceback
        traceback.print_exc()


        exit(1)
