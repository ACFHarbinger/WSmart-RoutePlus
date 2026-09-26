#!/usr/bin/env python
"""
Main Entry Point for the WSmart-Route Application.

This module serves as the primary gateway for executing various components of the
project, including the WSmart-Route simulator, reinforcement learning training
pipelines, and test suite execution. The graphical interface is the WSmart-Route
Studio Tauri application (see `app/`), which spawns these CLI commands.

It dispatches commands based on arguments parsed by ``logic.controllers.cli``
to the appropriate subsystems (Logic, Test).

Key Functions:
    - main: The central dispatcher that routes the parsed CLI arguments to the
      corresponding execution logic via the Hydra or argparse dispatch systems.
"""

import sys
import typing
import warnings

# Ensure typing.Self compatibility for Python < 3.11
if not hasattr(typing, "Self"):
    try:
        from typing_extensions import Self

        typing.Self = Self  # type: ignore[attr-defined]
    except ImportError:
        # Fallback to Any if typing_extensions is missing
        typing.Self = typing.Any  # type: ignore[attr-defined]

# Ensure torch.multiprocessing.reductions.ForkingPickler compatibility (for Tensordict + Torch 2.11+)
try:
    import torch.multiprocessing.reductions

    if not hasattr(torch.multiprocessing.reductions, "ForkingPickler"):
        import multiprocessing.reduction as reduction

        torch.multiprocessing.reductions.ForkingPickler = reduction.ForkingPickler  # type: ignore[attr-defined]
except ImportError:
    pass

from logic.controllers.cli import parse_params
from logic.controllers.hydra_dispatch import hydra_entry_point
from logic.controllers.parser_dispatch import parser_entry_point

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="jax.tree_util.register_keypaths is deprecated",
)


def main() -> None:
    """
    Dispatch commands to either Hydra-based or legacy argparse-based systems.

    This function routes incoming CLI commands to the appropriate execution path
    based on the command type. It handles the transition between the old argparse
    system and the modern Hydra configuration system.

    System Architecture:
        - Hydra Commands: Configuration-driven, composable, supports sweeps
        - Legacy Commands: Direct argument parsing, simpler but less flexible

    Args:
        None (reads from sys.argv)

    Returns:
        None (calls sys.exit with appropriate code)
    """
    # ========================================================================
    # Dual Dispatch System
    # ========================================================================
    # This application uses TWO command routing systems for historical reasons:
    #
    # 1. HYDRA_COMMANDS (below): Modern commands routed through Hydra/Lightning.
    #    Examples: train, eval, test_sim, gen_data
    #    These bypass parse_params() and go directly to hydra_entry_point().
    #
    # 2. Legacy CLI (parse_params): Traditional argparse-based routing.
    #    Examples: test_suite, file_system
    #    These use the original argument parser and main() function.
    #
    # The sys.argv manipulation below converts user commands into Hydra-compatible
    # overrides (e.g., 'eval' -> 'task=eval').
    # ========================================================================
    task_aliases = {"evaluation": "eval", "sim_hpo": "hpo_sim"}
    hydra_tasks = {"train", "eval", "test_sim", "gen_data", "hpo", "meta_train", "hpo_sim"}
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if first_arg in hydra_tasks or first_arg in task_aliases or first_arg.startswith("tasks="):
            if first_arg in hydra_tasks or first_arg in task_aliases:
                task = task_aliases.get(sys.argv.pop(1), first_arg)
                sys.argv.append(f"tasks={task}")
            hydra_entry_point()
            return

    parser_entry_point(parse_params())


if __name__ == "__main__":
    # Required for multiprocessing (simulator workers) in a frozen executable;
    # a no-op when running from source.
    import multiprocessing

    multiprocessing.freeze_support()
    main()
