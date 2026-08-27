"""
Comprehensive Performance Benchmark Runner.

Executes latency, throughput, and solver benchmarks across neural models,
local search operators, and exact/metaheuristic solvers.

Attributes:
    main: Entry point for running all benchmarks.

Example:
    >>> python -m logic.benchmark.run_all
"""

from __future__ import annotations

import time

import torch
from loguru import logger

from logic.benchmark.baseline_benchmarks import benchmark_random_local_search
from logic.benchmark.benchmark_suite import (
    benchmark_ls_throughput,
    benchmark_neural_latency,
    benchmark_solvers,
)
from logic.benchmark.neural_benchmarks import benchmark_neural_model


def main() -> None:
    """Run all performance benchmarks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"  WSmart+ Route Performance Benchmark Suite (Device: {device})")
    print("=" * 70)

    start_time = time.time()

    # 1. Neural model latency & batch scaling
    try:
        benchmark_neural_latency(device=device)
    except Exception as e:
        logger.warning(f"Neural latency benchmark skipped: {e}")

    # 2. Neural model inference (AM on VRPP)
    try:
        benchmark_neural_model(model_name="am", num_nodes=50, batch_size=128)
    except Exception as e:
        logger.warning(f"Neural model benchmark skipped: {e}")

    # 3. Vectorized Local Search throughput
    try:
        benchmark_ls_throughput(device=device)
    except Exception as e:
        logger.warning(f"Local search throughput benchmark skipped: {e}")

    # 4. Random Local Search baseline
    try:
        benchmark_random_local_search(batch_size=128, num_nodes=50, iterations=50)
    except Exception as e:
        logger.warning(f"Random LS benchmark skipped: {e}")

    # 5. OR Solver performance (small instance)
    try:
        benchmark_solvers()
    except Exception as e:
        logger.warning(f"OR solver benchmark skipped: {e}")

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"  Benchmark Suite Completed in {total_time:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
