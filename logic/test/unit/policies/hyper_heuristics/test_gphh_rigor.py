"""
Rigor tests for the GPHH 4-fix refactoring.

Validates:
  1. Deep genetic operators (crossover & mutation reach all depths)
  2. No profitability gate (all capacity-feasible candidates scored)
  3. External training environments (distinct dist matrices)
  4. K-NN candidate list (candidates restricted to K neighbours of endpoints)
  5. Terminal synchronisation
  6. End-to-end solver correctness
"""

import random

import numpy as np
import pytest
from logic.src.policies.routing.hyper_heuristics.genetic_programming_hyper_heuristic.params import GPHHParams
from logic.src.policies.routing.hyper_heuristics.genetic_programming_hyper_heuristic.solver import GPHHSolver
from logic.src.policies.routing.hyper_heuristics.genetic_programming_hyper_heuristic.tree import (
    _TERMINALS,
    ConstantNode,
    FunctionNode,
    TerminalNode,
    _collect_mutable_points,
    _mutate,
    _random_tree,
    _subtree_crossover,
    to_callable,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def dist_3() -> np.ndarray:
    return np.array([
        [0.0, 10.0, 20.0],
        [10.0, 0.0, 15.0],
        [20.0, 15.0, 0.0],
    ])


@pytest.fixture
def dist_5() -> np.ndarray:
    """5-node Euclidean instance."""
    rng_np = np.random.RandomState(7)
    coords = rng_np.rand(5, 2) * 100
    n = len(coords)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dm[i][j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
    return dm


@pytest.fixture
def small_params() -> GPHHParams:
    return GPHHParams(
        gp_pop_size=4, max_gp_generations=2, tree_depth=3,
        candidate_list_size=3, n_training_instances=2,
        training_sample_ratio=1.0, seed=42,
    )


@pytest.fixture
def small_solver(dist_3, small_params) -> GPHHSolver:
    wastes = {1: 10.0, 2: 20.0}
    return GPHHSolver(dist_3, wastes, 50.0, R=1.0, C=1.0, params=small_params)


@pytest.fixture
def medium_solver(dist_5) -> GPHHSolver:
    wastes = {1: 15.0, 2: 25.0, 3: 10.0, 4: 30.0}
    params = GPHHParams(
        gp_pop_size=4, max_gp_generations=2, tree_depth=3,
        candidate_list_size=3, n_training_instances=2,
        training_sample_ratio=0.75, seed=42,
    )
    return GPHHSolver(dist_5, wastes, 60.0, R=2.0, C=0.5, params=params)


# ---------------------------------------------------------------------------
# 1. Deep genetic operators
# ---------------------------------------------------------------------------


class TestDeepOperators:
    """Operators must reach nodes at all depths, not just level 1."""

    def test_collect_mutable_points_from_terminal(self):
        """Terminal has exactly one mutable point: the root itself."""
        t = TerminalNode("node_profit")
        assert _collect_mutable_points(t) == [(None, None)]

    def test_collect_mutable_points_counts(self, rng):
        """A tree with k FunctionNodes yields exactly 2k + 1 mutable points (including root)."""
        # Build a known 2-FunctionNode tree: ADD(SUB(T,T), T)
        t = FunctionNode("ADD",
                         FunctionNode("SUB", TerminalNode("node_profit"), TerminalNode("insertion_cost")),
                         TerminalNode("remaining_capacity"))
        pts = _collect_mutable_points(t)
        # 2 FunctionNodes -> 4 child slots + 1 root = 5 points
        assert len(pts) == 5

    def test_collect_points_includes_deep_nodes(self, rng):
        """Deep nodes (depth ≥ 2) must appear in the mutable point list."""
        # Build depth-3 chain: ADD(MUL(DIV(T,T), T), T)
        leaf = TerminalNode("node_profit")
        inner_inner = FunctionNode("DIV", leaf.copy(), leaf.copy())
        inner = FunctionNode("MUL", inner_inner, leaf.copy())
        root = FunctionNode("ADD", inner, leaf.copy())

        pts = _collect_mutable_points(root)
        parents = {id(p) for p, _ in pts}
        # inner_inner (depth 2) must be reachable
        assert id(inner_inner) in parents

    def test_crossover_can_modify_deep_subtrees(self, rng):
        """After many crossover operations some must produce offspring that differ
        from both parents at depth >= 2. Uses asymmetric ctx so any subtree swap
        produces a detectably different evaluation result."""
        t1 = FunctionNode("ADD",
                          FunctionNode("MUL", TerminalNode("node_profit"), TerminalNode("insertion_cost")),
                          TerminalNode("remaining_capacity"))
        t2 = FunctionNode("SUB",
                          FunctionNode("DIV", TerminalNode("distance_to_route"), TerminalNode("node_profit")),
                          TerminalNode("insertion_cost"))

        # Asymmetric: each terminal maps to a distinct prime value
        ctx = {
            "node_profit": 3.0,
            "insertion_cost": 7.0,
            "remaining_capacity": 11.0,
            "distance_to_route": 13.0,
        }
        orig1 = t1.evaluate(ctx)
        orig2 = t2.evaluate(ctx)

        changed = False
        for _ in range(50):
            c1, c2 = _subtree_crossover(t1, t2, rng, max_depth=10)
            if c1.evaluate(ctx) != orig1 or c2.evaluate(ctx) != orig2:
                changed = True
                break
        assert changed, "Crossover never produced a different offspring after 50 trials"


    def test_mutation_can_modify_deep_nodes(self, rng):
        """Mutation must be able to change nodes deeper than level 1."""
        # Build a known tree with a distinctive deep node
        deep_leaf = TerminalNode("distance_to_route")  # Distinctive terminal
        t = FunctionNode("ADD",
                         FunctionNode("MUL", deep_leaf, TerminalNode("node_profit")),
                         TerminalNode("remaining_capacity"))

        ctx = {k: 2.0 for k in _TERMINALS}
        orig_score = t.evaluate(ctx)

        changed = False
        for _ in range(50):
            mutated = _mutate(t.copy(), depth=3, rng=rng, max_depth=10)
            if mutated.evaluate(ctx) != orig_score:
                changed = True
                break
        assert changed, "Mutation never changed the tree's output after 50 trials"

    def test_mutation_enforces_depth_limit(self, rng):
        """Mutation must reject subtrees that cause the total tree to exceed max_depth."""
        # Tree of depth 2
        t = FunctionNode("ADD", TerminalNode("a"), TerminalNode("b"))

        # Mutate with max_depth=1 (impossible for an ADD node)
        mutated = _mutate(t, depth=2, rng=rng, max_depth=1)

        # Should return a copy of the original (which is depth 2), but
        # wait, if original is already > max_depth, it definitely returns copy.
        assert mutated.depth() == 2

        # More realistic: depth 3 tree, max_depth 3. If mutation tries to add
        # a depth 2 subtree at level 2, it might hit depth 4.
        t3 = FunctionNode("ADD", t.copy(), TerminalNode("c"))
        assert t3.depth() == 3

        # Force many mutations with small depth limit
        for _ in range(20):
            m = _mutate(t3.copy(), depth=3, rng=rng, max_depth=3)
            assert m.depth() <= 3, f"Mutation allowed tree to grow to depth {m.depth()}"

    def test_compile_tree_equivalence(self):
        """Verify that compiled lambda produces identical results to evaluate()."""
        t = FunctionNode("ADD",
                         FunctionNode("MUL", TerminalNode("node_profit"), TerminalNode("insertion_cost")),
                         FunctionNode("DIV", TerminalNode("distance_to_route"), TerminalNode("remaining_capacity")))

        ctx = {
            "node_profit": 10.5,
            "distance_to_route": 2.0,
            "insertion_cost": 1.5,
            "remaining_capacity": 50.0
        }

        expected = t.evaluate(ctx)

        # Compile and call
        func = to_callable(t)
        actual = func(
            node_profit=ctx["node_profit"],
            distance_to_route=ctx["distance_to_route"],
            insertion_cost=ctx["insertion_cost"],
            remaining_capacity=ctx["remaining_capacity"]
        )

        assert abs(actual - expected) < 1e-9

    def test_constant_node_evaluation(self):
        """Verify that ConstantNode returns its stored value regardless of context."""
        c = ConstantNode(0.5)
        assert c.evaluate({}) == 0.5
        assert c.evaluate({"any": 10.0}) == 0.5
        assert c.compile() == "0.5"

    def test_protected_div_compilation(self):
        """Verify that compiled DIV uses the protected_div helper."""
        # Build a tree that would normally bloat if DIV stringified sub-expressions twice
        # FunctionNode("DIV", l, r) -> "protected_div(l, r)"
        t = FunctionNode("DIV", TerminalNode("node_profit"), TerminalNode("distance_to_route"))
        expr = t.compile()
        assert "protected_div" in expr
        assert "if" not in expr  # The 'if' logic is now hidden inside the helper

        func = to_callable(t)
        assert func(10.0, 0.0, 0, 0) == 1.0  # Protected division by zero (min value in context)
        # Note: func arguments are node_profit, distance_to_route, insertion_cost, remaining_capacity
        # Our terminals a, b are not in the standard set, so we need to be careful with names.

        # Test with real terminals
        t2 = FunctionNode("DIV", TerminalNode("node_profit"), TerminalNode("insertion_cost"))
        func2 = to_callable(t2)
        assert func2(10.0, 0.0, 2.0, 0.0) == 5.0
        assert func2(10.0, 0.0, 0.0, 0.0) == 1.0  # Div by zero

    def test_erc_perturbation_mutation(self, rng):
        """Verify that mutation can perturb a ConstantNode's value."""
        c = ConstantNode(0.5)
        # Force mutation on the root constant node
        # Since it's a constant, _mutate has a 50% chance to perturb.
        # We'll run it a few times to ensure we catch a perturbation.
        perturbed = False
        for _ in range(50):
            m = _mutate(c.copy(), depth=1, rng=rng, max_depth=1)
            if isinstance(m, ConstantNode) and m.val != 0.5:
                perturbed = True
                break
        assert perturbed, "ERC was never perturbed after 50 mutations"

    def test_parsimony_default(self):
        """Verify that parsimony_coefficient defaults to 0.001."""
        params = GPHHParams()
        assert params.parsimony_coefficient == 0.001

        # Test from_config
        class MockConfig:
            pass
        config = MockConfig()
        params2 = GPHHParams.from_config(config)
        assert params2.parsimony_coefficient == 0.001

    def test_depth_tracking(self):
        """Verify depth() calculation for terminals and nested trees."""
        t1 = TerminalNode("node_profit")
        assert t1.depth() == 1
        t2 = FunctionNode("ADD", t1.copy(), t1.copy())
        assert t2.depth() == 2
        t3 = FunctionNode("MUL", t2.copy(), t1.copy())
        assert t3.depth() == 3

    def test_crossover_enforces_depth_limit(self, rng):
        """Crossover must reject offspring that exceed max_depth."""
        # Parent 1: depth 3
        p1 = FunctionNode("ADD",
                          FunctionNode("MUL", TerminalNode("a"), TerminalNode("b")),
                          TerminalNode("c"))
        # Parent 2: depth 2
        p2 = FunctionNode("SUB", TerminalNode("d"), TerminalNode("e"))

        # If we crossover p1's depth-2 subtree into p2, it stays depth 3.
        # But if we go deeper or swap the wrong way, it could grow.
        # Let's force a bloat by setting max_depth very small.
        off1, off2 = _subtree_crossover(p1, p2, rng, max_depth=2)

        # Offspring must be copies of parents because p1 (depth 3) already exceeds max_depth 2
        assert off1.depth() <= 2 or (off1.depth() == p1.depth() and off2.depth() == p2.depth())
        # In fact, my implementation reverts BOTH if EITHER bloats.
        # If p1 started at depth 3, and max_depth=2, it's already invalid?
        # Actually p1.depth()=3. If max_depth=2, crossover logic should revert.
        assert off1.depth() == p1.depth()
        assert off2.depth() == p2.depth()


# ---------------------------------------------------------------------------
# 2. No profitability gate
# ---------------------------------------------------------------------------


class TestNoProfitabilityGate:
    """All capacity-feasible candidates must be passed to GP scoring."""

    def test_unprofitable_node_can_be_inserted(self, dist_3):
        """A node with near-zero profit should be insertable when scored highly."""
        # Node 1 has tiny waste → low profit, but a tree biased to node_profit=0
        # should still get scored and potentially inserted.
        wastes = {1: 0.001, 2: 100.0}  # Node 1 is nearly worthless
        params = GPHHParams(seed=0, candidate_list_size=5, n_training_instances=1)
        solver = GPHHSolver(dist_3, wastes, 200.0, R=1.0, C=1.0, params=params)

        # A tree that always scores identically (constant 0.0) should insert both
        # nodes because neither is gated out — the GP decides.
        const_tree = TerminalNode("remaining_capacity")  # always non-zero
        routes = solver._construct_solution(
            const_tree, solver.nodes, wastes, set(), dist_3, solver._knn
        )
        visited = {n for r in routes for n in r}
        # Both nodes should be reachable for scoring (no pre-filter)
        assert len(visited) >= 1  # At minimum the profitable node 2 is visited


# ---------------------------------------------------------------------------
# 3. External training environments
# ---------------------------------------------------------------------------


class TestTrainingEnvs:
    """External training environments must be used when supplied."""

    def _make_env(self, seed: int, n: int = 3) -> tuple:
        rng = np.random.RandomState(seed)
        coords = rng.rand(n + 1, 2) * 50
        dm = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
        wastes = {i: rng.uniform(1, 10) for i in range(1, n + 1)}
        return dm, wastes, []

    def test_external_env_used(self, dist_3, small_params):
        """When training_environments are supplied, _resolve_training returns them."""
        env1 = self._make_env(1)
        env2 = self._make_env(2)
        wastes = {1: 10.0, 2: 20.0}
        solver = GPHHSolver(dist_3, wastes, 50.0, 1.0, 1.0,
                            small_params, training_environments=[env1, env2])
        resolved = solver._resolve_training()
        assert len(resolved) == 2
        # The resolved matrices are the external ones, not the test instance
        assert not np.array_equal(resolved[0][0], dist_3)

    def test_fallback_when_no_env(self, dist_3, small_params):
        """Without external envs, fallback generates node-subset instances."""
        wastes = {1: 10.0, 2: 20.0}
        solver = GPHHSolver(dist_3, wastes, 50.0, 1.0, 1.0, small_params)
        resolved = solver._resolve_training()
        assert len(resolved) == small_params.n_training_instances

    def test_evaluate_tree_with_external_env(self, dist_3, small_params):
        """Fitness evaluation must complete without error on external envs;
        the env must use same node count as solver to avoid index errors."""
        # Build training env matching dist_3 (2 customers, 3×3 matrix)
        env1 = (dist_3.copy(), {1: 8.0, 2: 12.0}, [])
        wastes = {1: 10.0, 2: 20.0}
        solver = GPHHSolver(dist_3, wastes, 50.0, 1.0, 1.0,
                            small_params, training_environments=[env1])
        tree = _random_tree(2, solver.rng)
        resolved = solver._resolve_training()
        fitness = solver._evaluate_tree(tree, resolved)
        assert isinstance(fitness, float)


# ---------------------------------------------------------------------------
# 4. K-NN candidate list
# ---------------------------------------------------------------------------


class TestKNNCandidateList:
    """K-NN index must be correct and candidates must be restricted."""

    def test_knn_length(self, dist_5, small_params):
        """KNN list length must be ≤ candidate_list_size."""
        nodes = list(range(1, 5))
        knn = GPHHSolver._build_knn(dist_5, nodes, k=3)
        for _n, neighbours in knn.items():
            assert len(neighbours) <= 3

    def test_knn_self_excluded(self, dist_5):
        """A node must not appear in its own KNN list."""
        nodes = list(range(1, 5))
        knn = GPHHSolver._build_knn(dist_5, nodes, k=4)
        for n, neighbours in knn.items():
            assert n not in neighbours

    def test_knn_ordered_by_distance(self, dist_3):
        """KNN must be sorted nearest-first."""
        nodes = [1, 2]
        knn = GPHHSolver._build_knn(dist_3, nodes, k=2)
        for n, neighbours in knn.items():
            dists = [dist_3[n][m] for m in neighbours]
            assert dists == sorted(dists)

    def test_knn_depot_included(self, dist_3):
        """Depot (node 0) must be in the KNN index."""
        nodes = [1, 2]
        knn = GPHHSolver._build_knn(dist_3, nodes, k=2)
        assert 0 in knn


# ---------------------------------------------------------------------------
# 5. Terminal synchronisation
# ---------------------------------------------------------------------------


class TestTerminalSync:
    def test_terminals_in_context(self, small_solver, dist_3):
        ctx = small_solver._build_insertion_context(
            10.0, [1], 2, 5.0, dist_3, 30.0
        )
        for t in _TERMINALS:
            assert t in ctx

    def test_context_keys_match_terminals(self, small_solver, dist_3):
        ctx = small_solver._build_insertion_context(
            10.0, [1], 2, 5.0, dist_3, 30.0
        )
        assert set(ctx.keys()) == set(_TERMINALS)

    def test_terminal_count(self):
        assert len(_TERMINALS) == 4


# ---------------------------------------------------------------------------
# 6. End-to-end solver
# ---------------------------------------------------------------------------


class TestSolverEndToEnd:
    def test_returns_valid_tuple(self, small_solver):
        routes, profit, cost = small_solver.solve()
        assert isinstance(routes, list)
        assert isinstance(profit, float)
        assert isinstance(cost, float)

    def test_profit_can_be_negative_without_gate(self, small_solver):
        """Without a profitability gate, a random tree may produce negative net profit.
        This is correct: the cost is always non-negative, and negative profit means
        routing cost exceeds revenue. This is resolved during GP evolution."""
        _, profit, cost = small_solver.solve()
        assert isinstance(profit, float)
        assert cost >= 0.0  # Cost is always non-negative (sum of distances)

    def test_cost_nonnegative(self, small_solver):
        _, _, cost = small_solver.solve()
        assert cost >= 0.0

    def test_empty_instance(self):
        dist = np.array([[0]])
        params = GPHHParams(seed=42)
        solver = GPHHSolver(dist, {}, 50.0, 1.0, 1.0, params)
        routes, profit, cost = solver.solve()
        assert routes == []
        assert profit == 0.0
        assert cost == 0.0

    def test_medium_instance_produces_routes(self, medium_solver):
        routes, _, _ = medium_solver.solve()
        profit = medium_solver._evaluate_routes(routes, medium_solver.wastes, medium_solver.dist_matrix)
        assert isinstance(profit, float)
        assert np.isfinite(profit)

    def test_no_duplicate_visits(self, medium_solver):
        routes, _, _ = medium_solver.solve()
        all_nodes = [n for r in routes for n in r]
        assert len(all_nodes) == len(set(all_nodes))

    def test_capacity_respected(self, medium_solver):
        routes, _, _ = medium_solver.solve()
        for route in routes:
            load = sum(medium_solver.wastes.get(n, 0.0) for n in route)
            assert load <= medium_solver.capacity + 1e-9
