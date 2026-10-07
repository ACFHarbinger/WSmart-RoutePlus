"""
Data Config module.

Attributes:
    DataConfig: Configuration for data generation.

Example:
    data_config = DataConfig(
        name="test",
        filename="test.pkl",
        data_dir="datasets",
        problem="ptp",
        mu=[10.0, 10.0],
        sigma=0.6,
        data_distributions=["gaussian"],
        num_locs=[50],
        penalty_factor=3.0,
        overwrite=False,
        seed=42,
        dataset_type="train",
        graphs=[GraphConfig()],
    )
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.envs.graph import GraphConfig


@dataclass
class DataConfig:
    """Data generation configuration.

    Attributes:
        name: Name to identify dataset. For training data, this will result in .td files.
        filename: Filename of the dataset to create (ignores datadir).
        data_dir: Create datasets in data.
        problem: Problem type selection. Should be 'ptp', 'mvptp', 'tcmvptp'
            or 'all'.
        mu: Mean of Gaussian noise (implies Gaussian noise generation if set).
        sigma: Variance of Gaussian noise.
        data_distributions: Distributions to generate for problems.
        num_locs: Sizes of problem instances.
        penalty_factor: Penalty factor for problems.
        overwrite: Set true to overwrite.
        seed: Random seed.
        dataset_type: Set type of dataset to generate ('train', 'train_time', 'test_simulator').
        graph: Graph/instance configuration.
    """

    name: Optional[str] = None
    filename: Optional[str] = None
    data_dir: str = "datasets"
    problem: str = "all"
    mu: Optional[List[float]] = None
    sigma: Any = 0.6
    data_distributions: List[str] = field(default_factory=lambda: ["all"])
    penalty_factor: float = 3.0
    overwrite: bool = False
    seed: int = 42
    dataset_type: Optional[str] = None
    graphs: List[GraphConfig] = field(default_factory=list)
