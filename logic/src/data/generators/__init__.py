"""
Data generation tools for creating synthetic VRP instances.

Provides builders, dataset generation scripts, and validators for
PTP, CMPTP and CMPTPTC problem types.

Attributes:
    generate_datasets: Generate datasets based on the provided arguments.
    validate_data_config: Validate the arguments for dataset generation.

Example:
    from logic.src.data.generators import generate_datasets, validate_data_config
    args = validate_data_config(args)
    data = generate_datasets(args)
"""

from .datasets import generate_datasets
from .validators import validate_data_config

__all__ = ["generate_datasets", "validate_data_config"]
