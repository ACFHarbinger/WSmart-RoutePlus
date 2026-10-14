"""Policy slug parsing and filter matching for targeted simulation run removal.

Policy slugs follow the pattern:
    ``{ms_strategy}_{constructor}[_{variant}]_{improver}_{distribution}``

Examples:
    ``lookahead_aco_hh_custom_ftsp_emp``
    ``lookahead_bpc_custom_rls_gamma3``
    ``last_minute_cf70_sans_new_ftsp_gamma1``
    ``lookahead_swc_tcf_gurobi_ftsp_emp``

Attributes:
    DISTRIBUTIONS: Known distribution tags.
    MS_STRATEGIES: Known packing algorithm prefixes.
    IMPROVERS: Known route-improver tags.
    PolicyFilter: Dataclass for specifying filter criteria.
    slug_matches_filter: Test whether a policy slug matches a filter.
    display_name_matches_filter: Test a human-readable display name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

# ---------------------------------------------------------------------------
# Known vocabulary sets (lower-case, no leading/trailing underscores)
# ---------------------------------------------------------------------------

DISTRIBUTIONS: List[str] = ["emp", "gamma1", "gamma2", "gamma3", "gamma4"]

MS_STRATEGIES: List[str] = [
    "lookahead",
    "last_minute",
    "last_minute_cf70",
    "last_minute_cf80",
    "last_minute_cf90",
    "regular",
    "none",
]

IMPROVERS: List[str] = [
    "ftsp",
    "fast_tsp",
    "rls",
    "rds",
    "random_local_search",
    "random_descent_search",
    "two_opt",
    "or_opt",
    "none",
]


@dataclass
class PolicyFilter:
    """Criteria for matching policy slugs or display names.

    All fields are optional; an unset field (``None`` or empty list) matches
    *anything*.  Matching is case-insensitive substring-based by default, but
    ``exact_match`` switches to whole-word token equality.

    Attributes:
        distributions: Accept only runs whose distribution tag is one of these
            (e.g. ``["emp", "gamma3"]``).
        constructors: Accept only runs whose routing-algorithm segment contains
            one of these tokens (e.g. ``["alns", "hgs", "aco_hh"]``).
        ms_strategies: Accept only runs whose packing prefix matches
            one of these (e.g. ``["lookahead", "last_minute"]``).
        improvers: Accept only runs whose route-improver tag is one of these
            (e.g. ``["ftsp", "none"]``).
        exact_match: When ``True`` compare whole tokens; when ``False`` (default)
            allow substring matches.
    """

    distributions: List[str] = field(default_factory=list)
    constructors: List[str] = field(default_factory=list)
    ms_strategies: List[str] = field(default_factory=list)
    improvers: List[str] = field(default_factory=list)
    exact_match: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ok(self, accepted: Sequence[str], value: str) -> bool:
        """Return True if *value* matches any entry in *accepted*, or if
        *accepted* is empty (no constraint).
        """
        if not accepted:
            return True
        value_low = value.lower()
        for token in accepted:
            t = token.lower()
            if self.exact_match:
                if value_low == t:
                    return True
            else:
                if t in value_low:
                    return True
        return False


def _parse_slug(slug: str) -> dict:
    """Decompose a policy slug into its constituent parts.

    Args:
        slug: Raw policy key, e.g. ``"lookahead_aco_hh_custom_ftsp_emp"``.

    Returns:
        Dict with keys ``distribution``, ``improver``, ``ms_strategy``,
        ``constructor`` (the remainder).
    """
    s = slug.lower()

    # 1. Strip trailing distribution tag
    distribution = ""
    for dist in sorted(DISTRIBUTIONS, key=len, reverse=True):
        if s == dist or s.endswith(f"_{dist}"):
            distribution = dist
            s = s[: -(len(dist) + 1)] if s.endswith(f"_{dist}") else ""
            break

    # 2. Strip trailing improver tag
    improver = ""
    for imp in sorted(IMPROVERS, key=len, reverse=True):
        if s == imp or s.endswith(f"_{imp}"):
            improver = imp
            s = s[: -(len(imp) + 1)] if s.endswith(f"_{imp}") else ""
            break

    # 3. Strip leading MS strategy
    ms_strategy = ""
    for ms in sorted(MS_STRATEGIES, key=len, reverse=True):
        if s == ms or s.startswith(f"{ms}_"):
            ms_strategy = ms
            s = s[len(ms) + 1 :] if s.startswith(f"{ms}_") else ""
            break

    # 4. Everything remaining is the routing algorithm (may include variant like "custom")
    constructor = s

    return {
        "distribution": distribution,
        "improver": improver,
        "ms_strategy": ms_strategy,
        "constructor": constructor,
        "raw": slug,
    }


def slug_matches_filter(slug: str, f: PolicyFilter) -> bool:
    """Return ``True`` when *slug* satisfies all non-empty criteria in *f*.

    Args:
        slug: Policy slug string, e.g. ``"lookahead_aco_hh_custom_ftsp_emp"``.
        f: The :class:`PolicyFilter` to test against.

    Returns:
        ``True`` if every populated filter field matches.
    """
    parsed = _parse_slug(slug)

    if not f._ok(f.distributions, parsed["distribution"]):
        return False
    if not f._ok(f.ms_strategies, parsed["ms_strategy"]):
        return False
    if not f._ok(f.improvers, parsed["improver"]):
        return False
    return not (f.constructors and not f._ok(f.constructors, parsed["constructor"]))


def display_name_matches_filter(display_name: str, f: PolicyFilter) -> bool:
    """Match the human-readable display name used in JSONL headers.

    Display names look like:
    ``"Lookahead + ACO_HH_CUSTOM + Ftsp (Emp)"``

    The matching strategy checks the full string for each filter token,
    which is intentionally permissive and distribution-agnostic for the
    display format.

    Args:
        display_name: Human-readable name from the JSONL ``GUI_DAY_LOG_START`` line.
        f: The :class:`PolicyFilter` to test against.

    Returns:
        ``True`` if every non-empty filter field has at least one match.
    """
    # Normalise to lowercase, strip punctuation for easy matching
    dn_low = re.sub(r"[^a-z0-9_ ]", " ", display_name.lower())

    def _any_in(tokens: Sequence[str]) -> bool:
        if not tokens:
            return True
        return any(t.lower().replace("_", " ") in dn_low or t.lower() in dn_low.replace(" ", "_") for t in tokens)

    if not _any_in(f.distributions):
        return False
    if not _any_in(f.ms_strategies):
        return False
    if not _any_in(f.constructors):
        return False
    return _any_in(f.improvers)
