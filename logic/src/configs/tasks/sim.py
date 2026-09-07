"""
Sim Config module.
Attributes:
    SimConfig: Configuration for simulation pipeline.

Example:
    sim_config = SimConfig(
        policies=["alns", "hgs", "sans", "aco_ks", "aco_hh", "pso", "psoma", "swc_tcf", "na"],
        full_policies=["alns", "hgs", "sans", "aco_ks", "aco_hh", "pso", "psoma", "swc_tcf", "na"],
        data_distribution="gamma1",
        problem="vrpp",
        days=31,
        seed=42,
        output_dir="output",
        checkpoint_dir="temp",
        checkpoint_days=0,
        n_samples=1,
        resume=False,
        cpu_cores=0,
        n_vehicles=1,
        noise_mean=0.0,
        noise_variance=0.0,
        cache_regular=True,
        no_cuda=False,
        server_run=False,
        env_file="vars.env",
        gplic_file=None,
        symkey_name=None,
        gapik_file=None,
        stats_filepath=None,
        data_dir=None,
        policy_configs={},  # To be populated by expand_policy_configs
        config_path={},  # To be populated by expand_policy_configs
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from logic.src.configs.envs.graph import GraphConfig


@dataclass
class SimConfig:
    """Simulation configuration.

    Attributes:
        policies: List of policy configurations to test on the WSR simulator.
        full_policies: Expanded policy names after config expansion (populated at runtime).
        data_distribution: Distribution to generate the bins daily waste fill.
        problem: Problem variant to simulate (e.g. 'vrpp', 'cvrpp', 'wcvrp', 'cwcvrp', 'scwcvrp', 'ctop').
        days: Number of days to run the simulation for.
        seed: Random seed.
        output_dir: Name of WSR simulator test output directory.
        checkpoint_dir: Name of WSR simulator test runs checkpoint directory.
        checkpoint_days: Number of days interval to save simulation checkpoints.
        n_samples: Number of simulation samplings for each policy.
        resume: Resume testing.
        n_vehicles: Number of vehicles.
        noise_mean: Mean of Gaussian noise to inject into observed bin levels.
        noise_variance: Variance of Gaussian noise to inject into observed bin levels.
        cache_regular: Deactivate caching for policy regular (Default True).
        no_cuda: Disable CUDA.
        server_run: Simulation will be executed in a remote server.
        env_file: Name of the file that contains the environment variables.
        gplic_file: Name of the file that contains the license to use for Gurobi.
        symkey_name: Name of the cryptographic key used to access the API keys.
        gapik_file: Name of the file that contains the key to use for the Google API.
        stats_filepath: Path to the file to read the statistics from.
        graph: Graph configuration for simulation trials.
        data_dir: Directory containing the simulation data.
        config_path: Policy configuration paths populated by expand_policy_configs.
    """

    policies: List[Any] = field(default_factory=list)
    full_policies: List[str] = field(default_factory=list)
    data_distribution: str = "gamma1"
    problem: str = "vrpp"
    shift_hours: float = 7.0
    avg_speed_kmh: float = 35.0
    service_time_h: float = 1.5 / 60.0
    days: int = 31
    seed: int = 42
    output_dir: str = "output"
    checkpoint_dir: str = "temp"
    checkpoint_days: int = 0
    n_samples: int = 1
    resume: bool = False

    cpu_cores: int = 0
    n_vehicles: int = 1
    graph: Any = field(default_factory=GraphConfig)
    noise_mean: float = 0.0
    noise_variance: float = 0.0
    cache_regular: bool = True
    no_cuda: bool = False
    server_run: bool = False
    env_file: str = "vars.env"
    gplic_file: Optional[str] = None
    symkey_name: Optional[str] = None
    gapik_file: Optional[str] = None
    stats_filepath: Optional[str] = None
    data_dir: Optional[str] = None
    run_name: Optional[str] = None
    policy_configs: Dict[str, Any] = field(default_factory=dict)
    config_path: Dict[str, Any] = field(default_factory=dict)
