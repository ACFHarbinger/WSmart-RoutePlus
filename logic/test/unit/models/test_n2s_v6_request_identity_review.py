"""Unit tests for N2S immutable request identity, feasible initialization, and precedence preservation."""

import pytest
import torch
from logic.src.envs.tsp_kopt import TSPkoptEnv
from logic.src.models.core.n2s.decoder import N2SDecoder
from logic.src.models.core.n2s.policy import N2SPolicy, create_feasible_pdp_solution, validate_pdp_solution

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_delivery_identity_immutable_across_tour_permutations():
    """Validates that delivery identity is determined strictly by the request schema, never tour order."""
    decoder = N2SDecoder(embed_dim=16, num_heads=4)
    bs, n = 2, 7
    device = torch.device("cpu")
    partners = torch.tensor([[0, 4, 5, 6, 1, 2, 3], [0, 4, 5, 6, 1, 2, 3]])

    # Tour 1: Standard canonical ordering (all pickups before deliveries)
    tour_valid = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [0, 2, 1, 3, 5, 4, 6]])
    # Tour 2: Reordered tour where deliveries precede pickups
    tour_reversed = torch.tensor([[0, 4, 5, 6, 1, 2, 3], [0, 6, 5, 4, 3, 2, 1]])

    deliv_valid, _ = decoder._identify_delivery_nodes(tour_valid, partners, bs, n, device)
    deliv_reversed, _ = decoder._identify_delivery_nodes(tour_reversed, partners, bs, n, device)

    # Delivery nodes must be exactly nodes 4, 5, 6 in both cases
    expected = torch.tensor([[False, False, False, False, True, True, True], [False, False, False, False, True, True, True]])
    assert torch.equal(deliv_valid, expected)
    assert torch.equal(deliv_reversed, expected)
    assert torch.equal(deliv_valid, deliv_reversed), "Delivery roles must not depend on tour sequence"


def test_policy_preserves_valid_initial_solution():
    """Validates that N2SPolicy preserves a caller-supplied feasible PDP solution."""
    env = TSPkoptEnv(num_loc=6)
    policy = N2SPolicy(embed_dim=16, num_heads=2, k_neighbors=3)

    td = env.reset(batch_size=[1])
    # Interleaved feasible tour: 0 -> 1 -> 4 -> 2 -> 5 -> 3 -> 6
    caller_solution = torch.tensor([[0, 1, 4, 2, 5, 3, 6]])
    partners = torch.tensor([[0, 4, 5, 6, 1, 2, 3]])
    td["solution"] = caller_solution.clone()
    td["partner_ids"] = partners

    out = policy(td, env, strategy="greedy", max_steps=1, return_actions=True)
    assert "actions" in out
    # Tour was modified by step 1, but step 1 operated on the caller's initial tour
    # Verify pickup 1 is still before delivery 4, 2 before 5, 3 before 6
    sol_out = td["solution"][0].tolist()
    assert sol_out.index(1) < sol_out.index(4)
    assert sol_out.index(2) < sol_out.index(5)
    assert sol_out.index(3) < sol_out.index(6)


def test_infeasible_initial_solution_rejected_with_clear_error():
    """Validates that an infeasible caller-supplied solution is rejected rather than silently reinterpreted."""
    env = TSPkoptEnv(num_loc=6)
    policy = N2SPolicy(embed_dim=16, num_heads=2, k_neighbors=3)

    td = env.reset(batch_size=[1])
    # Infeasible tour: delivery 4 appears before pickup 1
    infeasible_solution = torch.tensor([[0, 4, 2, 3, 1, 5, 6]])
    partners = torch.tensor([[0, 4, 5, 6, 1, 2, 3]])
    td["solution"] = infeasible_solution
    td["partner_ids"] = partners

    with pytest.raises(ValueError, match="Initial PDP solution violates precedence constraint"):
        policy(td, env, max_steps=1)


def test_unrestricted_reset_initializes_feasible_solution_and_preserves_precedence():
    """Validates that policy initializes a feasible solution on unrestricted TSP reset and preserves precedence across multiple moves."""
    env = TSPkoptEnv(num_loc=10)
    policy = N2SPolicy(embed_dim=32, num_heads=2, k_neighbors=5)

    td = env.reset(batch_size=[2])
    # The default reset on TSPkoptEnv produces arbitrary customer permutations
    out = policy(td, env, strategy="greedy", max_steps=5, return_actions=True)

    assert out["actions"].shape == (2, 5, 4)
    assert torch.isfinite(out["log_likelihood"]).all()

    # For each batch and each request, verify pickup precedes delivery
    sol = td["solution"]
    partners = td["partner_ids"]
    bs, n = sol.shape
    half = (n - 1) // 2

    for b in range(bs):
        row = sol[b].tolist()
        for p in range(1, half + 1):
            d = int(partners[b, p].item())
            assert row.index(p) < row.index(d), f"Batch {b}: pickup {p} at {row.index(p)} after delivery {d} at {row.index(d)}"


def test_create_and_validate_feasible_pdp_solution():
    """Tests the canonical feasible PDP generator and validator."""
    bs, n = 3, 11
    device = torch.device("cpu")
    half = (n - 1) // 2
    partner_ids = torch.empty(bs, n, dtype=torch.long, device=device)
    p = torch.arange(n, device=device)
    p[0] = 0
    p[1 : half + 1] = p[1 : half + 1] + half
    p[half + 1 :] = p[half + 1 :] - half
    partner_ids[:] = p

    node_indices = torch.arange(n, device=device).unsqueeze(0).expand(bs, -1)
    is_depot = partner_ids == node_indices
    is_delivery = (partner_ids < node_indices) & ~is_depot
    is_pickup = ~is_delivery & ~is_depot

    sol = create_feasible_pdp_solution(bs, n, partner_ids, is_pickup, is_delivery, device)
    assert sol.shape == (bs, n)

    # Validating feasible solution must pass without raising
    validate_pdp_solution(sol, partner_ids, is_pickup, is_delivery)

    # Corrupting one batch to violate precedence must raise ValueError
    corrupted = sol.clone()
    # Swap pickup 1 and delivery (1 + half)
    p1 = 1
    d1 = 1 + half
    pos_p1 = int((sol[0] == p1).nonzero().item())
    pos_d1 = int((sol[0] == d1).nonzero().item())
    corrupted[0, pos_p1] = d1
    corrupted[0, pos_d1] = p1

    with pytest.raises(ValueError, match="Initial PDP solution violates precedence constraint"):
        validate_pdp_solution(corrupted, partner_ids, is_pickup, is_delivery)
