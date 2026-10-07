"""
Data generation tools for creating synthetic VRP instances.

Provides builders, dataset generation scripts, and validators for
VRPP, CVRPP and CTOP problem types.

Attributes:
    generate_datasets: Generate datasets based on the provided arguments.
    validate_gen_data_args: Validate the arguments for dataset generation.

Example:
    from logic.src.data.generators import generate_datasets, validate_gen_data_args
    args = validate_gen_data_args(args)
    data = generate_datasets(args)
"""

from .datasets import generate_datasets
from .validators import validate_gen_data_args

__all__ = ["generate_datasets", "validate_gen_data_args"]
