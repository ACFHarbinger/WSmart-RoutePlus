"""Simulation output utilities.

Quick start::

    from logic.src.utils.output import PolicyFilter, remove_targeted_runs

    # Remove all 'emp' ALNS runs from a results directory
    remove_targeted_runs(
        results_dir="outputs/simulations/30_days/riomaior_100",
        policy_filter=PolicyFilter(
            distributions=["emp"],
            constructors=["alns"],
        ),
    )

    # Dry-run to preview
    removed = remove_targeted_runs(
        results_dir="outputs/simulations/30_days/riomaior_100",
        policy_filter=PolicyFilter(improvers=["ftsp"]),
        dry_run=True,
    )

CLI usage::

    python -m logic.controllers.cli.out_parser --help

Excel summary::

    from logic.src.utils.output import excel_summary
    df = excel_summary.discover_and_aggregate()

Attributes:
    PolicyFilter: Dataclass specifying which runs to target.
    remove_targeted_runs: Remove matching artefacts from a results directory.
    remove_from_json_file: Remove matching keys from a JSON file.
    remove_from_jsonl_file: Remove matching lines from a JSONL file.
    remove_checkpoint_files: Delete matching checkpoint pkl files.
    remove_fill_history_files: Delete matching XLSX fill-history files.
    slug_matches_filter: Test whether a policy slug matches a filter.
    excel_summary: Utility to aggregate simulation results JSONs into a single Excel log.
"""

from .excel_summary import discover_and_aggregate
from .matcher import PolicyFilter, slug_matches_filter
from .remover import (
    remove_checkpoint_files,
    remove_fill_history_files,
    remove_from_json_file,
    remove_from_jsonl_file,
    remove_targeted_runs,
)

__all__ = [
    "discover_and_aggregate",
    "PolicyFilter",
    "slug_matches_filter",
    "remove_targeted_runs",
    "remove_from_json_file",
    "remove_from_jsonl_file",
    "remove_checkpoint_files",
    "remove_fill_history_files",
]
