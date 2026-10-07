"""
Baseline performance benchmarks for classical optimization solvers and heuristics.

Extends benchmark_ls.py to include multi-vehicle solvers and parity checks.
"""

import time
from typing import Any, Dict

import numpy as np
import torch
from tensordict import TensorDict

from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow import (
    run_swc_tcf_optimizer,
)
from logic.src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.cvrp import (
    find_routes,
    find_routes_ortools,
)
from logic.src.policies.vector.random_local_search import (
    RandomLocalSearchPolicy,
)
from logic.src.tracking.logging.structured_logging import log_benchmark_metric


def benchmark_random_local_search(
    batch_size: int = 128,
    num_nodes: int = 50,
    iterations: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Benchmark the Random Local Search policy."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Benchmarking Random LS on {device} (batch={batch_size}, nodes={num_nodes}, iters={iterations})...")

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        locs = torch.rand(batch_size, num_nodes, 2, device=device)
        waste = torch.rand(batch_size, num_nodes, device=device)
    else:
        generator = torch.Generator().manual_seed(seed)
        locs = torch.rand(batch_size, num_nodes, 2, generator=generator)
        waste = torch.rand(batch_size, num_nodes, generator=generator)

    td = TensorDict(
        {
            "locs": locs,
            "waste": waste,
            "capacity": torch.ones(batch_size, device=device),
        },
        batch_size=[batch_size],
        device=device,
    )

    policy = RandomLocalSearchPolicy(env_name="mvptp", n_iterations=iterations, seed=seed).to(device)

    class MockEnv:
        waste_weight = 1.0
        cost_weight = 1.0

    # Warmup
    _ = policy(td, env=MockEnv())

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    _ = policy(td, env=MockEnv())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    total_time = end - start
    time_per_instance = total_time / batch_size

    print(f"    - Total Time: {total_time:.4f}s")
    print(f"    - Time per Instance: {time_per_instance:.6f}s")

    log_benchmark_metric(
        "baseline_rls",
        {"total_time": total_time, "time_per_instance": time_per_instance},
        {
            "policy": "random_ls",
            "batch_size": batch_size,
            "num_nodes": num_nodes,
            "iterations": iterations,
            "device": str(device)
        }
    )

    return {
        "total_time": total_time,
        "time_per_instance": time_per_instance,
        "instances_per_second": 1.0 / time_per_instance if time_per_instance > 0 else 0
    }


def benchmark_vrpp_solvers(num_nodes: int = 20, time_limit: int = 2, seed: int = 42) -> Dict[str, Any]:
    """Benchmark Gurobi vs Hexaly on VRPP."""
    print(f"[*] Benchmarking VRPP Solvers (nodes={num_nodes}, limit={time_limit}s)...")

    # Generate random instance
    np_rng = np.random.default_rng(seed)
    bins = np_rng.random(num_nodes) * 100
    dist_mat = np_rng.random((num_nodes + 1, num_nodes + 1)).tolist()
    binsids = list(range(num_nodes + 1))
    mandatory_nodes = list(range(1, num_nodes // 5))  # 20% critical

    params = {
        "Q": 100.0, "R": 1.0, "B": 1.0, "C": 1.0, "V": 1.0,
        "Omega": 10.0, "delta": 0.05, "psi": 0.8
    }

    results = {}

    for solver in ["gurobi", "hexaly"]:
        try:
            start = time.time()
            tour, profit, cost = run_swc_tcf_optimizer(
                bins=bins,
                distance_matrix=dist_mat,
                values=params,
                binsids=binsids,
                mandatory_nodes=mandatory_nodes,
                number_vehicles=1,
                time_limit=time_limit,
                optimizer=solver
            )
            elapsed = time.time() - start
            results[solver] = {
                "time": elapsed,
                "profit": profit,
                "cost": cost,
                "tour_len": len(tour)
            }
            print(f"    - {solver.capitalize()}: {elapsed:.4f}s, Profit: {profit:.2f}")

            log_benchmark_metric(
                f"baseline_vrpp_{solver}",
                {"time": elapsed, "profit": profit, "cost": cost},
                {"policy": solver, "num_nodes": num_nodes, "time_limit": time_limit}
            )
        except Exception as e:
            print(f"    - {solver.capitalize()} failed: {e}")

    return results


def benchmark_cvrp_solvers(num_nodes: int = 50, n_vehicles: int = 5, seed: int = 42) -> Dict[str, Any]:
    """Benchmark PyVRP vs OR-Tools."""
    print(f"[*] Benchmarking CVRP Solvers (nodes={num_nodes}, vehicles={n_vehicles})...")

    np_rng = np.random.default_rng(seed)
    dist_mat = np_rng.randint(10, 100, size=(num_nodes + 1, num_nodes + 1))
    np.fill_diagonal(dist_mat, 0)
    wastes = np_rng.randint(5, 20, size=num_nodes)
    max_caps = 100
    to_collect = list(range(1, num_nodes + 1))

    results = {}

    # PyVRP
    try:
        start = time.time()
        _ = find_routes(dist_mat, wastes, max_caps, to_collect, n_vehicles)
        elapsed = time.time() - start
        results["pyvrp"] = elapsed
        print(f"    - PyVRP: {elapsed:.4f}s")
        log_benchmark_metric(
            "baseline_multi_pyvrp",
            {"time": elapsed},
            {"policy": "pyvrp", "num_nodes": num_nodes, "n_vehicles": n_vehicles}
        )
    except Exception as e:
        print(f"    - PyVRP failed: {e}")

    # OR-Tools
    try:
        start = time.time()
        _ = find_routes_ortools(dist_mat, wastes, max_caps, to_collect, n_vehicles)
        elapsed = time.time() - start
        results["ortools"] = elapsed
        print(f"    - OR-Tools: {elapsed:.4f}s")
        log_benchmark_metric(
            "baseline_multi_ortools",
            {"time": elapsed},
            {"policy": "ortools", "num_nodes": num_nodes, "n_vehicles": n_vehicles}
        )
    except Exception as e:
        print(f"    - OR-Tools failed: {e}")

    return results


if __name__ == "__main__":
    benchmark_random_local_search(batch_size=64, num_nodes=20)
    print("-" * 40)
    benchmark_vrpp_solvers(num_nodes=15, time_limit=1)
    print("-" * 40)
    benchmark_cvrp_solvers(num_nodes=30, n_vehicles=3)
