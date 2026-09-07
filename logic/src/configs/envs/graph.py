"""
Graph/instance configuration module.

Attributes:
    GraphConfig: Graph configuration.

Example:
    >>> from logic.src.configs.envs import GraphConfig
    >>> config = GraphConfig()
    >>> print(config)
    GraphConfig(num_loc=50, num_nodes=50, num_customers=20, customer_types={'A': 0.25, 'B': 0.5, 'C': 0.25}, area='alpine', demand_distribution='normal', demand_normal_mean=10, demand_normal_std=1, min_demand=1, max_demand=25, capacity_distribution='normal', capacity_normal_mean=300, capacity_normal_std=50, min_capacity=200, max_capacity=400, instance_generator='random', edge_probability=0.3, shuffle_seed=42)
"""

from dataclasses import dataclass, field
from typing import Optional, Union

from .objective import ObjectiveConfig


@dataclass
class GraphConfig:
    """Configuration for problem instances and graph data.

    Attributes:
        area: County area of the bins locations.
        waste_type: Type of waste bins selected for the optimization problem.
        vertex_method: Method to transform vertex coordinates ('mmn', etc.).
        distance_method: Method to compute distance matrix ('ogd', etc.).
        dm_filepath: Path to the distance matrix file.
        edge_threshold: How many of all possible edges to consider.
        edge_method: Method for getting edges ('dist', 'knn', etc.).
        focus_graph: Paths to the files with the coordinates of the graphs to focus on.
        focus_size: Number of focus graphs to include.
        n_samples: Number of samples/instances to generate for this graph.
        start_day: The starting day of the simulation.
        n_days: Number of days to generate for this graph.
        reward: Per-graph objective/reward weights. When set, overrides ``env.reward``
            for this graph during curriculum or evaluation. When ``None``, the
            global ``env.reward`` is used.
    """

    area: str = "riomaior"
    num_loc: int = 50
    waste_type: str = "plastic"
    vertex_method: Optional[str] = "mmn"
    distance_method: str = "ogd"
    dm_filepath: Optional[str] = None
    tm_filepath: Optional[str] = None
    time_matrix_unit: str = "seconds"
    save_updated_dm: Optional[str] = None
    edge_threshold: Union[float, int, str] = "0"
    edge_method: Optional[str] = None
    focus_graph: Optional[str] = None
    focus_size: Optional[int] = None
    load_dataset: Optional[str] = None
    n_samples: int = 1
    start_day: int = 0
    n_days: int = 1
    reward: Optional[ObjectiveConfig] = field(default=None)
