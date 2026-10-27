"""Parsers for the algorithm-trace and module-catalog commands.

``trace_policy`` writes one execution trace. ``list_policy_modules`` writes
the catalog the Studio visualizer and builder read.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

from logic.controllers.cli.base import ConfigsParser
from logic.src.policies.trace_runner import PROBLEMS


def add_trace_policy_args(parser: ConfigsParser) -> None:
    """Arguments for ``python main.py trace_policy``."""
    parser.add_argument("--problem", default="ptp", choices=list(PROBLEMS), help="Problem id.")
    parser.add_argument("--graph-size", dest="graph_size", type=int, default=5, help="Nodes, including the depot.")
    parser.add_argument("--seed", type=int, default=1, help="Instance seed and ALNS seed.")
    parser.add_argument("--routing", default="alns", help="Routing policy id. This command solves alns.")
    parser.add_argument("--acceptance", default="oi", choices=["oi", "bmc"], help="Acceptance criterion id.")
    parser.add_argument("--packing", default=None, help="Packing id recorded on the trace. Not executed.")
    parser.add_argument(
        "--route-improvement",
        dest="route_improvement",
        default=None,
        help="Route improver id recorded on the trace. Not executed.",
    )
    parser.add_argument("--max-iterations", dest="max_iterations", type=int, default=20, help="ALNS iterations.")
    parser.add_argument("--output", required=True, help="Path of the JSON trace.")


def add_list_policy_modules_args(parser: ConfigsParser) -> None:
    """Arguments for ``python main.py list_policy_modules``."""
    parser.add_argument("--output", default=None, help="Write the catalog here. Default is stdout.")
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Import each registered target. A failed import is stored on that entry.",
    )


def run_trace_policy(opts: Dict[str, Any]) -> None:
    """Solve ALNS and write the trace. The output path is the only stdout line."""
    if opts.get("routing") != "alns":
        print(
            "trace_policy solves alns. Other routing ids stay not_instrumented until their trace issues land.",
            file=sys.stderr,
        )
        sys.exit(2)
    from logic.src.policies.trace_runner import build_alns_trace
    from logic.src.tracking.execution_trace import validate_trace

    trace, _result = build_alns_trace(
        problem=str(opts["problem"]),
        graph_size=int(opts["graph_size"]),
        seed=int(opts["seed"]),
        acceptance=str(opts["acceptance"]),
        packing=opts.get("packing"),
        route_improvement=opts.get("route_improvement"),
        max_iterations=int(opts["max_iterations"]),
    )
    document = trace.to_dict()
    validate_trace(document)
    destination = Path(str(opts["output"]))
    trace.write(destination)
    print(destination)


def run_list_policy_modules(opts: Dict[str, Any]) -> None:
    """Write the module catalog. Exit 0 even when some entries carry ``error``."""
    from logic.src.policies.catalog import build_catalog

    catalog = build_catalog(resolve=bool(opts.get("resolve")))
    text = json.dumps(catalog, indent=2) + "\n"
    output = opts.get("output")
    if output:
        destination = Path(str(output))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        print(destination)
        return
    sys.stdout.write(text)
