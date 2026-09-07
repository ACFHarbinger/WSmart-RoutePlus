"""datasets.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import datasets
"""

import os
import random
from typing import Any, Optional

import numpy as np
import torch

from logic.src.configs import Config
from logic.src.configs.envs.graph import GraphConfig
from logic.src.configs.tasks.data import DataConfig
from logic.src.constants import ROOT_DIR
from logic.src.data.generators.builders import VRPInstanceBuilder
from logic.src.data.generators.validators import validate_data_config
from logic.src.pipeline.simulations.repository import set_repository_from_path
from logic.src.utils.data.loader import (
    check_extension,
    save_simulation_dataset,
    save_td_dataset,
)


def generate_datasets(cfg: Config) -> None:
    """
    Generates VRP datasets based on the typed Hydra configuration.

    Args:
        cfg: Root Hydra configuration with ``cfg.data`` containing data
            generation parameters.
    """
    # Validate and sanitize config values
    validate_data_config(cfg)

    # Initialize the filesystem repository for coordinate/depot loading
    set_repository_from_path(str(ROOT_DIR))

    data = cfg.data

    # Set the random seed and execute the program
    random.seed(data.seed)
    np.random.seed(data.seed)
    torch.manual_seed(data.seed)

    gamma_dists = ["gamma1", "gamma2", "gamma3", "gamma4"]
    distributions_per_problem = {
        "vrpp": ["empty", "const", "unif", "dist", "emp", *gamma_dists],
        "wcvrp": ["empty", "const", "unif", "dist", "emp", *gamma_dists],
        "swcvrp": ["empty", "const", "unif", "dist", "emp", *gamma_dists],
    }

    # Define the problem distribution(s)
    if data.problem == "all":
        problems = distributions_per_problem
    else:
        problems = {
            data.problem: (
                distributions_per_problem[data.problem]
                if len(data.data_distributions) == 1 and data.data_distributions[0] == "all"
                else list(data.data_distributions)
            )
        }

    files_generated = 0
    for problem, distributions in problems.items():
        files_generated += _generate_problem_data(problem, distributions, data)


def _generate_problem_data(problem: str, distributions: Any, data: DataConfig) -> int:
    """Helper to generate data for a specific problem and its distributions.

    Args:
        problem: Description of problem.
        distributions: Description of distributions.
        data: Description of data.

    Returns:
        Number of files generated.
    """
    datadir = (
        os.path.join(ROOT_DIR, "data", data.data_dir, problem)
        if data.dataset_type in ["train_time", "train"]
        else os.path.join(ROOT_DIR, "data", data.data_dir)
    )
    try:
        os.makedirs(datadir, exist_ok=True)
    except Exception as e:
        raise Exception("directories to save generated data files do not exist and could not be created") from e

    graphs = list(data.graphs) if data.graphs else []
    file_count = 0

    try:
        for dist in distributions or [None]:
            if graphs:
                for graph_cfg in graphs:
                    file_count += _process_instance_generation(problem, dist, datadir, data, graph_cfg=graph_cfg)
            else:
                print("\n[WARNING] No graphs provided for instance generation. Skipping.")
    except Exception as e:
        has_dists = len(distributions) >= 1 and distributions[0] is not None
        raise Exception(
            "failed to generate data for problem {}{}  due to {}".format(
                problem, f" {distributions}" if has_dists else "", repr(e)
            )
        ) from e

    return file_count


def _process_instance_generation(
    problem: str,
    dist: Any,
    datadir: str,
    data: DataConfig,
    graph_cfg: Optional[GraphConfig] = None,
) -> int:
    """Configure builder and save datasets for a specific configuration.

    Args:
        problem: Description of problem.
        dist: Description of dist.
        datadir: Description of datadir.
        data: Description of data.
        graph_cfg: Description of graph_cfg.

    Returns:
        Number of files generated.
    """
    assert graph_cfg is not None, "graph_cfg must be provided"
    size = graph_cfg.num_loc
    graph = graph_cfg.focus_graph
    area = graph_cfg.area
    waste_type = graph_cfg.waste_type
    vertex_method = graph_cfg.vertex_method
    focus_size = graph_cfg.focus_size if graph_cfg.focus_size is not None else 0
    n_days = max(graph_cfg.n_days - graph_cfg.start_day, 1)
    n_samples = graph_cfg.n_samples if graph_cfg.n_samples is not None else 1
    if data.dataset_type == "train":
        n_days = 1

    if graph is not None and not os.path.isfile(graph):
        sim_dir = os.path.join(ROOT_DIR, "data", "simulator")
        if os.path.isfile(os.path.join(sim_dir, graph)):
            graph = os.path.join(sim_dir, graph)
        elif os.path.isfile(os.path.join(sim_dir, "bins_selection", graph)):
            graph = os.path.join(sim_dir, "bins_selection", graph)

    assert size is not None, "size must be provided"
    assert waste_type is not None, "waste_type must be provided"
    assert area is not None, "area must be provided"

    builder = VRPInstanceBuilder(device="cuda" if torch.cuda.is_available() else "cpu")
    builder.set_dataset_size(n_samples).set_problem_size(size).set_waste_type(waste_type).set_distribution(
        dist or "empty"
    ).set_area(area).set_focus_graph(graph, focus_size).set_method(vertex_method).set_problem_name(
        problem
    ).set_num_days(n_days).set_seed(data.seed)

    _apply_noise_config(builder, problem, data)

    file_count = 0
    if data.dataset_type == "test_simulator":
        file_count = _generate_test_simulator_data(
            builder, n_days, datadir, dist, size, data, area, waste_type, n_samples
        )
    elif data.dataset_type == "train_time":
        file_count = _generate_train_time_data(builder, problem, n_days, datadir, dist, size, data)
    else:
        file_count = _generate_train_data(
            builder, problem, graph_cfg.n_days, graph_cfg.start_day, datadir, dist, size, data
        )

    return file_count


def _apply_noise_config(builder: VRPInstanceBuilder, problem: str, data: DataConfig) -> None:
    """Apply noise parameters to the builder.

    Args:
        builder: Description of builder.
        problem: Description of problem.
        data: Description of data.
    """
    if data.mu is not None:
        sigma = data.sigma
        if isinstance(sigma, list):
            sigma = sigma[0]
        builder.set_noise(data.mu[0] if isinstance(data.mu, list) else data.mu, (sigma**2 if sigma > 0 else 0.0))
    elif problem == "swcvrp":
        sigma = data.sigma
        if isinstance(sigma, list):
            sigma = sigma[0]
        mu = data.mu[0] if isinstance(data.mu, list) and data.mu else 0.0
        builder.set_noise(
            mu if mu is not None else 0.0,
            (sigma**2 if sigma > 0 else 1.0),
        )


def _generate_test_simulator_data(
    builder: VRPInstanceBuilder,
    n_days: int,
    datadir: str,
    dist: Any,
    size: int,
    data: DataConfig,
    area: str,
    waste_type: str,
    n_samples: int,
) -> int:
    """Generate and save test simulator data.

    Args:
        builder: Description of builder.
        n_days: Description of n_days.
        datadir: Description of datadir.
        dist: Description of dist.
        size: Description of size.
        data: Description of data.
        area: Description of area.
        waste_type: Description of waste_type.
        n_samples: Description of n_samples.

    Returns:
        Number of files generated.
    """
    name = data.name or "dataset"
    if data.filename is None:
        filename = os.path.join(
            datadir,
            "{}{}{}_{}{}_N{}_seed{}.npz".format(
                area,
                size,
                "_{}".format(dist) if dist is not None else "",
                name,
                n_days if n_days > 0 else "",
                n_samples,
                data.seed,
            ),
        )
    else:
        filename = check_extension(data.filename, ".npz")

    _verify_and_save(builder, filename, data, is_td=False)
    return 1


def _generate_train_time_data(
    builder: VRPInstanceBuilder, problem: str, n_days: int, datadir: str, dist: Any, size: int, data: DataConfig
) -> int:
    """Generate and save train time data.

    Args:
        builder: Description of builder.
        problem: Description of problem.
        n_days: Description of n_days.
        datadir: Description of datadir.
        dist: Description of dist.
        size: Description of size.
        data: Description of data.

    Returns:
        Number of files generated.
    """
    name = data.name or "dataset"
    if data.filename is None:
        ext = ".td"
        if data.mu is not None:
            mu_val = data.mu[0] if isinstance(data.mu, list) else data.mu
            sigma_val = data.sigma[0] if isinstance(data.sigma, list) else data.sigma
            filename = os.path.join(
                datadir,
                "{}{}{}_{}{}_seed{}_{}_{}{}".format(
                    problem,
                    size,
                    "_{}".format(dist) if dist is not None else "",
                    name,
                    n_days if n_days > 0 else "",
                    data.seed,
                    f"gaussian{mu_val}",
                    f"_{sigma_val}",
                    ext,
                ),
            )
        else:
            filename = os.path.join(
                datadir,
                "{}{}{}_{}{}_seed{}{}".format(
                    problem,
                    size,
                    "_{}".format(dist) if dist is not None else "",
                    name,
                    n_days if n_days > 0 else "",
                    data.seed,
                    ext,
                ),
            )
    else:
        filename = check_extension(data.filename, ".td")

    _verify_and_save(builder, filename, data, is_td=True)
    return 1


def _generate_train_data(
    builder: VRPInstanceBuilder,
    problem: str,
    n_days: int,
    start_day: int,
    datadir: str,
    dist: Any,
    size: int,
    data: DataConfig,
) -> int:
    """Generate and save standard training data.

    Args:
        builder: Description of builder.
        problem: Description of problem.
        datadir: Description of datadir.
        dist: Description of dist.
        size: Description of size.
        data: Description of data.

    Returns:
        Number of files generated.
    """
    name = data.name or "dataset"
    for epoch in range(start_day, n_days):
        print("- Generating epoch {} data".format(epoch))
        if data.filename is None:
            ext = ".td"
            if data.mu is not None:
                mu_val = data.mu[0] if isinstance(data.mu, list) else data.mu
                sigma_val = data.sigma[0] if isinstance(data.sigma, list) else data.sigma
                filename = os.path.join(
                    datadir,
                    "{}{}{}_{}{}_seed{}_{}_{}{}".format(
                        problem,
                        size,
                        ("_{}".format(dist) if dist is not None else ""),
                        name,
                        epoch if n_days > 1 else "",
                        data.seed,
                        f"gaussian{mu_val}",
                        f"_{sigma_val}",
                        ext,
                    ),
                )
            else:
                filename = os.path.join(
                    datadir,
                    "{}{}{}_{}{}_seed{}{}".format(
                        problem,
                        size,
                        ("_{}".format(dist) if dist is not None else ""),
                        name,
                        epoch if n_days > 1 else "",
                        data.seed,
                        ext,
                    ),
                )
        else:
            filename = check_extension(data.filename, ".td")

        _verify_and_save(builder, filename, data, is_td=True)
    return max(n_days - start_day, 0)


def _verify_and_save(builder: VRPInstanceBuilder, filename: str, data: DataConfig, is_td: bool = False) -> None:
    """Verify file existence and save the dataset.

    Args:
        builder: Description of builder.
        filename: Description of filename.
        data: Description of data.
        is_td: Description of is_td.
    """
    ext = ".td" if is_td else ".npz"
    assert data.overwrite or not os.path.isfile(check_extension(filename, ext)), (
        f"File {filename} already exists! Try running with -f option to overwrite."
    )

    if is_td:
        dataset = builder.build_td()
        save_td_dataset(dataset, filename)
    else:
        dataset = builder.build()
        save_simulation_dataset(dataset, filename)

    print(f"Generated {filename}")
