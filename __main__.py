#!/usr/bin/env python
"""
Entry Point for Package Execution (Hydra commands only).

Routes CLI commands to the Hydra-based configuration system.
Non-Hydra commands (test_suite, file_system, benchmark) must be
invoked via ``python main.py <command>`` instead.
"""

import sys
import warnings

from logic.controllers.hydra_dispatch import hydra_entry_point

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="jax.tree_util.register_keypaths is deprecated",
)


def main() -> None:
    """Parse the CLI command and dispatch to Hydra."""
    if len(sys.argv) < 2:
        print("Usage: python -m WSmart-Route <task> [overrides...]")
        print("Available tasks: train, evaluation, test_sim, gen_data, hpo, meta_train")
        print("\nFor non-Hydra commands (test_suite, file_system, etc.), use: python main.py <command>")
        sys.exit(1)

    raw_task = sys.argv.pop(1)
    task = {"evaluation": "eval", "sim_hpo": "hpo_sim"}.get(raw_task, raw_task)
    sys.argv.append(f"tasks={task}")
    sys.argv.append(f"task={task}")
    hydra_entry_point()


if __name__ == "__main__":
    main()
