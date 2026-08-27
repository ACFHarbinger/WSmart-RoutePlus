import time

import numpy as np
import torch
from tensordict import TensorDict

from logic.src.envs.problems import VRPP
from logic.src.models import AttentionModel
from logic.src.models.subnets.factories import AttentionComponentFactory
from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf import (
    run_swc_tcf_optimizer,
)
from logic.src.policies.vector.random_local_search import RandomLocalSearchPolicy


def get_dummy_model(device="cpu"):
    factory = AttentionComponentFactory()
    model = AttentionModel(
        embed_dim=128,
        hidden_dim=128,
        problem=VRPP,
        component_factory=factory,
        n_heads=8,
        n_encode_layers=3,
        normalization="instance",
    ).to(device)
    model.eval()
    return model


def benchmark_neural_latency(device="cpu", seed=42):
    print(f"\n--- Neural Model Latency (Device: {device}) ---")
    model = get_dummy_model(device)
    batch_sizes = [1, 64, 256]
    n_nodes = 50
    generator = torch.Generator(device=device).manual_seed(seed)
    for bs in batch_sizes:
        td = TensorDict(
            {
                "locs": torch.rand(bs, n_nodes, 2, generator=generator, device=device),
                "depot": torch.rand(bs, 2, generator=generator, device=device),
                "waste": torch.rand(bs, n_nodes, generator=generator, device=device),
                "capacity": torch.ones(bs, device=device),
                "max_waste": torch.ones(bs, device=device),
            },
            batch_size=[bs],
            device=device,
        )

        # Warmup
        model.set_strategy("greedy")
        with torch.no_grad():
            _ = model(td.clone())

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(td.clone())
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        avg_time = (end - start) / 10
        print(f"Batch Size {bs:3d} | Latency: {avg_time*1000:7.2f} ms | Throughput: {bs/avg_time:10.2f} inst/s")


def benchmark_solvers(seed=42):
    print("\n--- OR Solver Performance (N=20) ---")
    n_bins = 20
    np_rng = np.random.default_rng(seed)
    dist_matrix = np_rng.random((n_bins + 1, n_bins + 1)).tolist()
    bins = np_rng.random(n_bins) * 100
    values = {"Q": 100.0, "R": 1.0, "B": 1.0, "C": 0.1, "V": 1.0, "Omega": 0.1, "delta": 0.0, "psi": 0.8}
    binsids = list(range(n_bins + 1))
    mandatory_nodes = [5, 10, 15]

    for backend in ["gurobi", "hexaly"]:
        start = time.time()
        try:
            run_swc_tcf_optimizer(
                bins=bins,
                distance_matrix=dist_matrix,
                values=values,
                binsids=binsids,
                mandatory_nodes=mandatory_nodes,
                optimizer=backend,
                time_limit=10,
                seed=seed,
            )
            end = time.time()
            print(f"{backend:8s} | Solve Time: {end - start:7.4f} s")
        except Exception as e:
            print(f"{backend:8s} | Skipped ({e})")


def benchmark_ls_throughput(device="cpu", seed=42):
    print(f"\n--- Local Search Throughput (Device: {device}) ---")
    bs = 512
    n_nodes = 50
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        locs = torch.rand(bs, n_nodes, 2, device=device)
        waste = torch.rand(bs, n_nodes, device=device)
    else:
        generator = torch.Generator().manual_seed(seed)
        locs = torch.rand(bs, n_nodes, 2, generator=generator)
        waste = torch.rand(bs, n_nodes, generator=generator)

    td = TensorDict(
        {
            "locs": locs,
            "waste": waste,
            "capacity": torch.ones(bs, device=device),
        },
        batch_size=[bs],
        device=device,
    )

    policy = RandomLocalSearchPolicy(env_name="cvrpp", n_iterations=100, seed=seed).to(device)

    class MockEnv:
        waste_weight = 1.0
        cost_weight = 1.0

    # Warmup
    _ = policy(td, env=MockEnv())

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    _ = policy(td, env=MockEnv())
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    print(f"Batch Size {bs:3d} | Total Time: {end - start:7.4f} s | Throughput: {bs/(end-start):10.2f} inst/s")



def run_benchmarks(args):
    """
    Main entry point for the benchmark suite.
    """
    subset = args.get("subset", "all")
    device_arg = args.get("device", "auto")
    output_file = args.get("output")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_arg == "auto" else device_arg

    results = {}  # type: ignore[var-annotated]

    print(f"Running Benchmarks (Subset: {subset}, Device: {device})...")

    if subset in ["all", "neural"]:
        # Capture stdout? Or just run and let it print?
        # For now, let it print.
        benchmark_neural_latency(device)

    if subset in ["all", "ls"]:
        benchmark_ls_throughput(device)

    if subset in ["all", "solvers"]:
        benchmark_solvers()

    if output_file:
        import json
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Simulate args for direct execution
    run_benchmarks({"subset": "all", "device": "auto"})
