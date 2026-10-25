"""
Environment configuration dataclasses.

Attributes:
    EnvConfig: Environment configuration.
    GraphConfig: Graph configuration.
    ObjectiveConfig: Objective configuration.

Example:
    >>> from logic.src.configs.envs import EnvConfig, GraphConfig, ObjectiveConfig
    >>> env_config = EnvConfig()
    >>> print(env_config)
    EnvConfig(max_nodes=50, n_customers=20, customer_types={'A': 0.25, 'B': 0.5, 'C': 0.25}, n_depots=1, demand_distribution='normal', demand_normal_mean=10, demand_normal_std=1, min_demand=1, max_demand=25, capacity_distribution='normal', capacity_normal_mean=300, capacity_normal_std=50, min_capacity=200, max_capacity=400, instance_generator='random', max_travel_time=24, opening_time=0, closing_time=24, min_waste_nodes_ratio=0.4, max_waste_nodes_ratio=0.6, service_time=1, n_time_bins=1, max_time_bin_length=1, generate_empty_graphs=False, n_edges_factor=3, shuffle_seed=42)
    >>> graph_config = GraphConfig()
    >>> print(graph_config)
    GraphConfig(num_loc=50, num_nodes=50, num_customers=20, customer_types={'A': 0.25, 'B': 0.5, 'C': 0.25}, area='alpine', demand_distribution='normal', demand_normal_mean=10, demand_normal_std=1, min_demand=1, max_demand=25, capacity_distribution='normal', capacity_normal_mean=300, capacity_normal_std=50, min_capacity=200, max_capacity=400, instance_generator='random', edge_probability=0.3, shuffle_seed=42)
    >>> objective_config = ObjectiveConfig()
    >>> print(objective_config)
    ObjectiveConfig(w_capacity=1.0, w_distance=1.0, w_time=0.0, w_penalty=1.0, penalty_threshold=1.0, alpha_distance=1.0)
"""

from .env import EnvConfig
from .graph import GraphConfig
from .objective import ObjectiveConfig

__all__ = [
    "EnvConfig",
    "GraphConfig",
    "ObjectiveConfig",
]
