"""Policy-slug parsing for simulation log analysis.

Moved from ``logic.src.tracking.logging.modules.analysis``.
"""

from typing import Dict, List, Optional


def _parse_slug(
    slug: str,
    mandatory_prefixes: Optional[List[str]] = None,
    routing_algorithms: Optional[List[str]] = None,
    route_improvers: Optional[List[str]] = None,
    routing_algorithm_engines: Optional[List[str]] = None,
    acceptance_criteria: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Best-effort decomposition of a policy slug into its five named components.

    The expected slug format is::

        {packing}_{routing_algorithm}_{routing_algorithm_engine}_{acceptance_criterion}_{route_improver}

    Any component may be absent.  Parsing proceeds in this order so that each
    vocabulary is applied to the right portion of the slug:

    1. Strip the ``packing`` prefix (longest match from the start).
    2. Strip the ``route_improver`` suffix (longest match from the end).
    3. Strip the ``routing_algorithm`` prefix (longest match from the start).
    4. Strip the ``routing_algorithm_engine`` prefix (longest match from the
       start of the remainder).
    5. Match ``acceptance_criterion`` from the vocabulary (longest prefix match);
       fall back to consuming the whole remainder when no vocabulary entry
       matches.

    Vocabulary lists default to the module-level constants when ``None``
    is passed.  ``_ROUTING_ALGORITHM_ENGINES`` and ``_ACCEPTANCE_CRITERIA``
    default to empty lists, so callers must supply them when their slugs
    contain those components.

    Args:
        slug: Full policy slug, e.g.
            ``"lookahead_aco_hh_ortools_custom_ftsp"``.
        mandatory_prefixes: Ordered packing prefixes (longest-first).
            Defaults to ``_MANDATORY_PREFIXES``.
        routing_algorithms: Ordered routing-algorithm names.  Defaults to
            ``_ROUTING_ALGORITHMS``.
        route_improvers: Ordered route-improver suffixes (longest-first).
            Defaults to ``_ROUTE_IMPROVERS``.
        routing_algorithm_engines: Ordered constructor-engine names
            (longest-first).  Defaults to ``_ROUTING_ALGORITHM_ENGINES``
            (empty — must be supplied when engines appear in slugs).
        acceptance_criteria: Ordered acceptance-criterion names (longest-first).
            Defaults to ``_ACCEPTANCE_CRITERIA`` (empty — whatever remains
            after all other components are stripped becomes the criterion).

    Returns:
        Dict with keys ``packing``, ``routing_algorithm``,
        ``routing_algorithm_engine``, ``acceptance_criterion``, and
        ``route_improver``.  Missing components are empty strings.
    """
    m_prefixes = mandatory_prefixes if mandatory_prefixes is not None else _MANDATORY_PREFIXES
    r_constructors = routing_algorithms if routing_algorithms is not None else _ROUTING_ALGORITHMS
    r_improvers = route_improvers if route_improvers is not None else _ROUTE_IMPROVERS
    r_engines = routing_algorithm_engines if routing_algorithm_engines is not None else _ROUTING_ALGORITHM_ENGINES
    a_criteria = acceptance_criteria if acceptance_criteria is not None else _ACCEPTANCE_CRITERIA

    remainder = slug

    # 1. Strip packing prefix from the start.
    mandatory = ""
    for prefix in m_prefixes:
        if remainder.startswith(prefix + "_"):
            mandatory = prefix
            remainder = remainder[len(prefix) + 1 :]
            break

    # 2. Strip route-improver suffix from the end.
    improver = ""
    for suf in r_improvers:
        if remainder.endswith("_" + suf):
            improver = suf
            remainder = remainder[: -(len(suf) + 1)]
            break

    # 3. Strip routing-algorithm prefix from the start.
    constructor = ""
    for c in r_constructors:
        if remainder.startswith(c):
            constructor = c
            rest = remainder[len(c) :]
            remainder = rest[1:] if rest.startswith("_") else rest
            break

    # 4. Strip routing algorithm engine from the start of what remains.
    engine = ""
    for e in r_engines:
        if remainder.startswith(e):
            engine = e
            rest = remainder[len(e) :]
            remainder = rest[1:] if rest.startswith("_") else rest
            break

    # 5. Match acceptance criterion from vocabulary; fall back to the whole remainder.
    acceptance = ""
    for a in a_criteria:
        if remainder.startswith(a):
            rest = remainder[len(a) :]
            # Only consume if it accounts for the whole remainder or is followed by '_'.
            if rest == "" or rest.startswith("_"):
                acceptance = a
                remainder = rest[1:] if rest.startswith("_") else ""
                break
    if not acceptance:
        acceptance = remainder

    return {
        "packing": mandatory,
        "routing_algorithm": constructor,
        "routing_algorithm_engine": engine,
        "acceptance_criterion": acceptance,
        "route_improver": improver,
    }


_MANDATORY_PREFIXES: List[str] = [
    "last_minute_cf70",
    "last_minute_cf80",
    "last_minute_cf90",
    "last_minute_cf",
    "service_level1",
    "service_level2",
    "service_level",
    "last_minute",
    "regular_tr_lvl5_delayed",
    "regular_tr_lvl4_delayed",
    "regular_tr_lvl3_delayed",
    "regular_lvl5_delayed",
    "regular_lvl4_delayed",
    "regular_lvl3_delayed",
    "regular_tr_lvl5",
    "regular_tr_lvl4",
    "regular_tr_lvl3",
    "regular_lvl5",
    "regular_lvl4",
    "regular_lvl3",
    "regular_lvl2",
    "regular_lvl1",
    "regular",
    "lookahead",
]

_ROUTE_IMPROVERS: List[str] = [
    "tsp_fast_tsp",
    "cvrp_ortools",
    "ftsp",
    "tsp",
]

_ROUTING_ALGORITHMS: List[str] = [
    "vrpp_gurobi",
    "pg_clns",
    "swc_tcf",
    "sans_new",
    "ks_aco",
    "aco_hh",
    "alns",
    "psoma",
    "sisr",
    "hvpl",
    "gurobi",
    "ortools",
    "sans",
    "hgs",
    "bpc",
]

_ROUTING_ALGORITHM_ENGINES: List[str] = []

_ACCEPTANCE_CRITERIA: List[str] = []
