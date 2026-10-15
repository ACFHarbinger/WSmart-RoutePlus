"""Policy YAML config-override update utilities.

Quick start::

    from logic.src.utils.target import update_packing, update_route_improvement

    # Switch all standard routing algorithms to service-level packing
    update_packing(
        constructors=["aco_hh", "alns", "bpc", "hgs"],
        pk_yaml="pk_service_level",
        keys=["service_level1", "service_level2"],
    )

    # Switch all standard routing algorithms to fast-TSP route improvement
    update_route_improvement(
        constructors=["aco_hh", "alns", "bpc", "hgs"],
        ri_yaml="ri_ftsp",
        keys=["ftsp"],
    )

    # Dry-run to preview
    update_packing(
        constructors=["sans"],
        pk_yaml="pk_last_minute",
        keys=["last_minute_cf70", "last_minute_cf90"],
        dry_run=True,
    )

CLI usage::

    python -m logic.controllers.cli.tgt_parser ms --help
    python -m logic.controllers.cli.tgt_parser ri --help

Attributes:
    update_packing: Batch-update packing in policy YAML files.
    update_route_improvement: Batch-update route_improvement in policy YAML files.
    list_available_packing_strategies: List available packing algorithm file stems.
    list_available_ri_improvers: List available route-improver file stems.
    list_strategy_keys: List top-level keys defined in a strategy file.
    list_improver_keys: List top-level keys defined in an improver file.
"""

from .policy_link_updater import (
    list_available_packing_strategies,
    list_strategy_keys,
    update_packing,
)
from .policy_link_updater import (
    list_available_ri_improvers,
    list_improver_keys,
    update_route_improvement,
)

__all__ = [
    "update_packing",
    "update_route_improvement",
    "list_available_packing_strategies",
    "list_available_ri_improvers",
    "list_strategy_keys",
    "list_improver_keys",
]
