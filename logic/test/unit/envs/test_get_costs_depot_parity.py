"""Unit tests verifying PTP and MVPTP get_costs parity across width N and N+1 waste shapes.

Addresses Issue #61 design item:
- PTP.get_costs unconditionally prepended a depot waste column, shifting customer
  indices when dataset was a reset TensorDict (width N+1).
- Prepends only when waste lacks the depot column (width N), ensuring exact parity
  and preventing customer index shift.
"""

import pytest
import torch
from logic.src.envs.routing.ptp import PTPEnv
from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.mvptp import MVPTP
from logic.src.envs.tasks.ptp import PTP


@pytest.mark.unit
@pytest.mark.fast
def test_ptp_get_costs_width_n_vs_n1_parity() -> None:
    """PTP.get_costs must produce identical cost dict and profit for width N and N+1."""
    depot = torch.tensor([[0.0, 0.0]])
    locs_N = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]])
    waste_N = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    dataset_N = {
        "depot": depot,
        "locs": locs_N,
        "waste": waste_N,
        "cost_km": 1.0,
        "revenue_kg": 1.0,
    }

    # Tour visits node 1 (waste 10) and node 3 (waste 30)
    pi = torch.tensor([[0, 1, 3, 0]])

    # Format N+1: prepended with depot (as produced by PTPEnv.reset)
    locs_N1 = torch.cat([depot.unsqueeze(1), locs_N], dim=1)
    waste_N1 = torch.cat([torch.zeros(1, 1), waste_N], dim=1)
    dataset_N1 = {
        "depot": depot,
        "locs": locs_N1,
        "waste": waste_N1,
        "cost_km": 1.0,
        "revenue_kg": 1.0,
    }

    neg_profit_N, c_dict_N, _ = PTP.get_costs(dataset_N, pi, cw_dict=None)
    neg_profit_N1, c_dict_N1, _ = PTP.get_costs(dataset_N1, pi, cw_dict=None)

    # Waste: customer 1 (10) + customer 3 (30) = 40.0
    assert torch.allclose(c_dict_N["waste"], torch.tensor([40.0]))
    assert torch.allclose(c_dict_N1["waste"], torch.tensor([40.0]))

    # Tour length: 0->1 (1.0) + 1->3 (2.0) + 3->0 (3.0) = 6.0
    assert torch.allclose(c_dict_N["length"], torch.tensor([6.0]))
    assert torch.allclose(c_dict_N1["length"], torch.tensor([6.0]))

    # Negative profit: length (6.0) - waste (40.0) = -34.0
    assert torch.allclose(neg_profit_N, torch.tensor([-34.0]))
    assert torch.allclose(neg_profit_N1, torch.tensor([-34.0]))


@pytest.mark.unit
@pytest.mark.fast
def test_ptp_get_costs_no_customer_shift_on_reset_tensordict() -> None:
    """Calling get_costs on a reset TensorDict must not shift customer indices."""
    env = PTPEnv(num_loc=5, batch_size=[1])
    td_raw = env.generator(1)
    # Distinct waste values for each customer: 10, 20, 30, 40, 50
    td_raw["waste"] = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]])
    td_raw["max_waste"] = torch.tensor([100.0])

    td_reset = env.reset(td_raw.clone())

    # Tour visiting specifically customer 1: must collect exactly 10.0 (not depot 0.0)
    pi_c1 = torch.tensor([[0, 1, 0]])
    _, c_dict_c1, _ = PTP.get_costs(td_reset, pi_c1, cw_dict=None)
    assert c_dict_c1["waste"].item() == pytest.approx(10.0)

    # Tour visiting specifically customer 5 (last customer): must collect 50.0 (not customer 4's 40.0)
    pi_c5 = torch.tensor([[0, 5, 0]])
    _, c_dict_c5, _ = PTP.get_costs(td_reset, pi_c5, cw_dict=None)
    assert c_dict_c5["waste"].item() == pytest.approx(50.0)


@pytest.mark.unit
@pytest.mark.fast
def test_ptp_get_costs_batched_parity() -> None:
    """Verify batched parity across width N and N+1 with multiple instances."""
    B, N = 4, 6
    torch.manual_seed(123)
    depot = torch.randn(B, 2)
    locs_N = torch.randn(B, N, 2)
    waste_N = torch.rand(B, N) * 50.0
    dataset_N = {"depot": depot, "locs": locs_N, "waste": waste_N}

    locs_N1 = torch.cat([depot.unsqueeze(1), locs_N], dim=1)
    waste_N1 = torch.cat([torch.zeros(B, 1), waste_N], dim=1)
    dataset_N1 = {"depot": depot, "locs": locs_N1, "waste": waste_N1}

    pi = torch.tensor([[0, 1, 2, 4, 0]] * B)

    c_N = PTP.get_costs(dataset_N, pi, cw_dict=None)
    c_N1 = PTP.get_costs(dataset_N1, pi, cw_dict=None)

    assert torch.allclose(c_N[0], c_N1[0])
    assert torch.allclose(c_N[1]["waste"], c_N1[1]["waste"])
    assert torch.allclose(c_N[1]["length"], c_N1[1]["length"])


@pytest.mark.unit
@pytest.mark.fast
def test_mvptp_get_costs_width_n_vs_n1_parity() -> None:
    """MVPTP.get_costs must produce identical cost dict and trip capacity check on both shapes."""
    depot = torch.zeros(1, 2)
    locs_N = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    waste_N = torch.tensor([[20.0, 30.0]])
    dataset_N = {
        "depot": depot,
        "locs": locs_N,
        "waste": waste_N,
        "capacity": torch.tensor([50.0]),
        "max_waste": torch.tensor([50.0]),
    }

    locs_N1 = torch.cat([depot.unsqueeze(1), locs_N], dim=1)
    waste_N1 = torch.cat([torch.zeros(1, 1), waste_N], dim=1)
    dataset_N1 = {
        "depot": depot,
        "locs": locs_N1,
        "waste": waste_N1,
        "capacity": torch.tensor([50.0]),
        "max_waste": torch.tensor([50.0]),
    }

    pi = torch.tensor([[0, 1, 2, 0]])

    cost_N, c_dict_N, _ = MVPTP.get_costs(dataset_N, pi, cw_dict=None)
    cost_N1, c_dict_N1, _ = MVPTP.get_costs(dataset_N1, pi, cw_dict=None)

    assert torch.allclose(cost_N, cost_N1)
    assert torch.allclose(c_dict_N["waste"], c_dict_N1["waste"])
    assert torch.allclose(c_dict_N["length"], c_dict_N1["length"])


@pytest.mark.unit
@pytest.mark.fast
def test_base_problem_get_tour_length_parity() -> None:
    """BaseProblem.get_tour_length must return identical tour length for customer-only and prepended locs."""
    B, N = 3, 4
    torch.manual_seed(42)
    depot = torch.randn(B, 2)
    locs_N = torch.randn(B, N, 2)
    waste_N = torch.rand(B, N)
    dataset_N = {"depot": depot, "locs": locs_N, "waste": waste_N}

    locs_N1 = torch.cat([depot.unsqueeze(1), locs_N], dim=1)
    waste_N1 = torch.cat([torch.zeros(B, 1), waste_N], dim=1)
    dataset_N1 = {"depot": depot, "locs": locs_N1, "waste": waste_N1}

    pi = torch.tensor([[0, 1, 2, 0]] * B)

    len_N = BaseProblem.get_tour_length(dataset_N, pi)
    len_N1 = BaseProblem.get_tour_length(dataset_N1, pi)

    assert torch.allclose(len_N, len_N1)


@pytest.mark.unit
@pytest.mark.fast
def test_base_problem_get_waste_with_depot_edge_cases() -> None:
    """Verify get_waste_with_depot handles fallbacks and edge cases gracefully."""
    # Case 1: only waste and depot, pi.max() >= waste.shape[-1]
    dataset = {"waste": torch.tensor([[10.0, 20.0]])}
    pi = torch.tensor([[0, 2, 0]])  # Node 2 requires width 3
    w = BaseProblem.get_waste_with_depot(dataset, pi)
    assert w.shape == torch.Size([1, 3])
    assert torch.equal(w, torch.tensor([[0.0, 10.0, 20.0]]))

    # Case 2: explicit schema establishes that waste already includes depot
    dataset_with_zero = {"waste": torch.tensor([[0.0, 10.0, 20.0]]), "num_loc": 2}
    pi2 = torch.tensor([[0, 1, 0]])
    w2 = BaseProblem.get_waste_with_depot(dataset_with_zero, pi2)
    assert torch.equal(w2, torch.tensor([[0.0, 10.0, 20.0]]))

    # Case 3: dataset with 'visited'
    dataset_visited = {
        "waste": torch.tensor([[10.0, 20.0]]),
        "visited": torch.zeros(1, 3, dtype=torch.bool),
    }
    w3 = BaseProblem.get_waste_with_depot(dataset_visited)
    assert torch.equal(w3, torch.tensor([[0.0, 10.0, 20.0]]))


@pytest.mark.unit
@pytest.mark.fast
def test_evaluator_parity_width_n_vs_n1() -> None:
    """Verify training evaluator logic produces exact parity across width N and N+1 datasets."""
    # Simulate the pipeline/features/eval/engine.py get_costs invocation:
    # _, c_dict, _ = model.problem.get_costs(batch_i, seq_tensor, None, ...)
    depot = torch.tensor([[0.5, 0.5]])
    locs_N = torch.tensor([[[0.1, 0.2], [0.8, 0.7], [0.3, 0.9]]])
    waste_N = torch.tensor([[0.4, 0.6, 0.8]])

    # Evaluator instance representation (width N, as loaded from make_dataset)
    instance_N = {
        "depot": depot,
        "locs": locs_N,
        "waste": waste_N,
        "cost_km": 1.0,
        "revenue_kg": 1.0,
    }

    # Reset environment representation (width N+1)
    instance_N1 = {
        "depot": depot,
        "locs": torch.cat([depot.unsqueeze(1), locs_N], dim=1),
        "waste": torch.cat([torch.zeros(1, 1), waste_N], dim=1),
        "cost_km": 1.0,
        "revenue_kg": 1.0,
    }

    seq = torch.tensor([[0, 1, 2, 0]])

    _, c_dict_N, _ = PTP.get_costs(instance_N, seq, None)
    _, c_dict_N1, _ = PTP.get_costs(instance_N1, seq, None)

    # Parity check across all metrics produced for evaluation results
    assert c_dict_N["length"].item() == pytest.approx(c_dict_N1["length"].item())
    assert c_dict_N["waste"].item() == pytest.approx(c_dict_N1["waste"].item())
    assert c_dict_N["total"].item() == pytest.approx(c_dict_N1["total"].item())
