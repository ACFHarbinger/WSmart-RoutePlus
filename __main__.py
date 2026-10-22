#!/usr/bin/env python
"""
Entry Point for Package Execution (Hydra commands only).

Routes CLI commands to the Hydra-based configuration system.
Non-Hydra commands (test_suite, file_system, benchmark) must be
invoked via ``python main.py <command>`` instead.
"""

import sys
import warnings
from pathlib import Path

from logic.controllers.hydra_dispatch import hydra_entry_point

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="jax.tree_util.register_keypaths is deprecated",
)


def _usage() -> str:
    """Usage text listing the tasks whose configs this tree ships (the release ships fewer)."""
    root = Path(__file__).resolve().parent
    tasks = sorted(p.stem for p in (root / "logic" / "configs" / "tasks").glob("*.yaml"))
    text = f"Usage: python . <task> [overrides...]\nAvailable tasks: {', '.join(tasks)}"
    if (root / "main.py").is_file():
        text += "\n\nFor non-Hydra commands (test_suite, file_system, etc.), use: python main.py <command>"
    return text


def main() -> None:
    """Parse the CLI command and dispatch to Hydra."""
    if len(sys.argv) < 2:
        print(_usage())
        sys.exit(1)
    if sys.argv[1] in ("-h", "--help", "help"):
        print(_usage())
        sys.exit(0)

    raw_task = sys.argv.pop(1)
    task = {"evaluation": "eval", "sim_hpo": "hpo_sim"}.get(raw_task, raw_task)
    sys.argv.append(f"tasks={task}")
    sys.argv.append(f"task={task}")
    hydra_entry_point()


if __name__ == "__main__":
    main()
