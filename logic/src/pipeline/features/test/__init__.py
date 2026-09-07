"""
Simulation testing sub-package for WSmart-Route.

Attributes:
    run_simulator_test: Runs the WSR simulator test.
    simulator_testing: Main entry point for simulator testing.

Example:
    >>> from logic.src.pipeline.features.test import run_simulator_test, simulator_testing
    >>> run_simulator_test()
    >>> simulator_testing(config)
"""

from .engine import run_simulator_test
from .orchestrator import simulator_testing

__all__ = ["run_simulator_test", "simulator_testing"]
