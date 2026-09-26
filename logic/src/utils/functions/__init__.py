"""
Mathematical and environment-related helper functions sub-package.

Attributes:
    safe_exp: Compute exponent with overflow/underflow protection.
    run_all_in_pool: Runs a function over a dataset in parallel.
    get_path_until_string: Truncates a path up to a specific directory component.
    is_wc_problem: Check if the problem is a Waste Collection (WC) variant.
    is_vrpp_problem: Check if the problem is a Vehicle Routing Problem with Profits (VRPP) variant.
    is_tsp_problem: Check if the problem is a Traveling Salesperson Problem (TSP) variant.
    ensure_tensordict: Converts various input types to TensorDict.
    sample_many: Samples many solutions by repeated execution.
    move_to: Recursively moves variables to the specified device.
    compute_in_batches: Computes a memory-heavy function in batches.
    do_batch_rep: Replicates a variable n times along the batch dimension.

Example:
    >>> from logic.src.utils.functions import safe_exp, run_all_in_pool, get_path_until_string, is_wc_problem, is_vrpp_problem, is_tsp_problem, ensure_tensordict, sample_many, move_to, compute_in_batches, do_batch_rep
    >>> # Mathematical utility
    >>> safe_exp(1000)
    inf
    >>> # Parallel execution
    >>> def my_func(directory, filename, *args):
    ...     # Process file
    ...     return result
    >>> results, num_cpus = run_all_in_pool(my_func, "data", dataset)
    >>> # Path manipulation
    >>> path = "/home/user/project/src/module.py"
    >>> get_path_until_string(path, "src")
    '/home/user/project/src'
    >>> # Problem type checking
    >>> is_wc_problem("wcvrp")
    True
    >>> is_vrpp_problem("cvrpp")
    True
    >>> is_tsp_problem("tsp")
    True
    >>> # TensorDict conversion
    >>> import torch
    >>> from tensordict import TensorDict
    >>> td = ensure_tensordict({"a": torch.tensor([1, 2])}, device="cuda")
    >>> # Sampling
    >>> minpis, mincosts = sample_many(inner_func, get_cost_func, input)
    >>> # Tensor operations
    >>> moved = move_to(tensor, "cuda")
    >>> result = compute_in_batches(heavy_func, 2, tensor)
    >>> replicated = do_batch_rep(tensor, 3)
    >>> print(replicated.shape)
    torch.Size([6])
"""

from .math import safe_exp
from .parallel import run_all_in_pool
from .path import get_path_until_string
from .problem import is_tsp_problem, is_vrpp_problem, is_wc_problem
from .sampling import sample_many
from .tensors import compute_in_batches, do_batch_rep, move_to

__all__ = [
    "move_to",
    "safe_exp",
    "run_all_in_pool",
    "get_path_until_string",
    "compute_in_batches",
    "do_batch_rep",
    "sample_many",
    "is_wc_problem",
    "is_vrpp_problem",
    "is_tsp_problem",
]
