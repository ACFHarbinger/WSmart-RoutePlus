"""
Export the paper's simulation results and policy space as JSON for the website.

The interactive site must not repeat the paper's old mistake of describing an
experiment nobody ran, and it must not disagree with the paper on the numbers
that survive integrity filtering. This generator is therefore a thin wrapper
over the *same* integrity machinery that produces ``paper.tex``'s tables and
figures: it imports ``gen_paper_latex`` and reuses its ``split_degenerate`` /
``drop_affected_cells`` / ``balance_marginal`` / ``improver_pairs`` /
``pareto_front`` so that the website, the markdown reports and the paper share
one definition of "clean data" and one set of exclusions. No number below is
transcribed by hand.

It emits four JSON documents into ``docs/website/public/data/``:

  pipeline.json   the policy configuration space, enumerated from registry
                  decorators in ``logic/src/policies/``, with the benchmarked
                  subset marked. Counts describe implementation files rather
                  than backward-compatibility aliases.

  results.json    the balanced 30-day aggregates (constructors, selection
                  variants, improver pairs, Pareto front, scenario effects) and
                  the paired 30-vs-90-day horizon table.

  bin_fills.json  real per-bin, per-day fill levels (percent 0-100) for each of
                  the three network sizes, read from the simulator's
                  ``fill_history`` spreadsheets, for the selection-strategy
                  widget.

  routes.json     per-day tours and collection metrics for one representative
                  run per network size, with a 2-D node layout obtained by
                  classical multidimensional scaling of the repository's real
                  road-distance matrices (``gmaps_distmat.csv`` for Rio Maior,
                  ``osm_distmat.csv`` for Figueira da Foz).

Raw per-bin coordinates are not checked into the repository (they live under the
gitignored ``data/`` tree), so the spatial view embeds the distance matrix into
two dimensions rather than plotting lat/lng. The embedding preserves pairwise
road distances up to a 2-D projection; it is a layout, not a map, and the
website labels it as such.

Usage
-----
    uv run python logic/gen/export_website_data.py [--force]

Idempotent: skips existing output unless --force is passed.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import gen_paper_latex as gpl
import numpy as np
import pandas as pd
from report_utils import load_json

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGIC_SRC = REPO_ROOT / "logic" / "src" / "policies"
OUT_DIR = REPO_ROOT / "docs" / "website" / "public" / "data"
OUTPUT_DIR = REPO_ROOT / "assets" / "output" / "30days"

#: The one representative policy run per network whose day-by-day tours we
#: export for the 4-D routing view. ACO-HH under the Last-Minute CF70 strategy
#: with Classical Local Search, on the heavier (Gamma-3) demand process -- the
#: "runtime/quality sweet spot" configuration the paper highlights.
REPRESENTATIVE = {"constructor": "aco_hh", "strategy": "last_minute_cf70", "improver": "cls"}

SCENARIOS = [
    {"city": "Rio Maior", "key": "riomaior100_plastic", "N": 100, "dm": "gmaps_distmat.csv"},
    {"city": "Rio Maior", "key": "riomaior170_plastic", "N": 170, "dm": "gmaps_distmat.csv"},
    {"city": "Figueira da Foz", "key": "figueiradafoz350_plastic", "N": 350, "dm": "osm_distmat.csv"},
]

FAMILY_LABELS = {
    "exact_and_decomposition_solvers": "Exact & Decomposition",
    "hyper_heuristics": "Hyper-heuristics",
    "learning_algorithms": "Learning",
    "learning_heuristic_algorithms": "Learning Heuristics",
    "learning_matheuristic_algorithms": "Learning Matheuristics",
    "matheuristics": "Matheuristics",
    "meta_heuristics": "Meta-heuristics",
    "other_algorithms": "Other",
}

#: Prettified names for the algorithm keys the site is most likely to show.
#: Anything not in the map is title-cased from its snake_case key.
NAME_OVERRIDES = {
    # selection
    "last_minute": "Last-Minute",
    "lookahead": "Look-Ahead",
    "service_level": "Service-Level",
    "regular": "Regular",
    "staggered_regular": "Staggered Regular",
    "profit_per_km": "Profit per km",
    "multi_day_prob": "Multi-day Overflow Probability",
    "set_cover": "Set Cover",
    "submodular_greedy": "Submodular Greedy",
    "supermodular_greedy": "Supermodular Greedy",
    "fptas_knapsack": "FPTAS Knapsack",
    "mip_knapsack": "MIP Knapsack",
    "fractional_knapsack": "Fractional Knapsack",
    "kmeans_sector": "K-Means Sector",
    "bernoulli_random": "Bernoulli Random",
    "stochastic_regret": "Stochastic Regret",
    "wasserstein": "Wasserstein Robust",
    "spatial_synergy": "Spatial Synergy",
    "filter_and_fan": "Filter-and-Fan",
    "dispatcher_portfolio": "Portfolio Dispatcher",
    "dispatcher_thompson": "Thompson Dispatcher",
    # construction
    "aco_hh": "ACO-HH",
    "pg_clns": "PG-CLNS",
    "swc_tcf": "SWC-TCF",
    "psoma": "PSOMA",
    "sans": "SANS",
    "alns": "ALNS",
    "bpc": "BPC",
    "hgs": "HGS",
    # improvement
    "fast_tsp": "Fast-TSP",
    "local_search": "Classical Local Search",
    "classical_local_search": "Classical Local Search",
    "dp_route_reopt": "DP Route Re-optimisation",
    "lkh": "LKH",
    "lkh2": "LKH-2",
    "lkh3": "LKH-3",
    "or_opt": "Or-opt",
    "or_opt_steepest": "Or-opt (steepest)",
    "node_exchange_steepest": "Node Exchange (steepest)",
    "steepest_two_opt": "2-opt (steepest)",
    "random_local_search": "Random Local Search",
    "guided_local_search": "Guided Local Search",
    "simulated_annealing": "Simulated Annealing",
    "ruin_recreate": "Ruin-and-Recreate",
    "set_partitioning": "Set Partitioning",
    "set_partitioning_polish": "Set Partitioning (polish)",
    "set_covering": "Set Covering",
    "fix_and_optimize": "Fix-and-Optimise",
    "mip_lns": "MIP LNS",
    "multi_phase": "Multi-phase",
    "adaptive_ensemble": "Adaptive Ensemble",
    "cross_exchange": "Cross Exchange",
    "profitable_detour": "Profitable Detour",
    "cheapest_insertion": "Cheapest Insertion",
    "regret_k_insertion": "Regret-k Insertion",
    "neural_selector": "Neural Selector",
    "learned": "Learned",
    "branch_and_price": "Branch-and-Price",
    "adaptive_large_neighborhood_search": "Adaptive LNS",
}


def prettify(key: str) -> str:
    """Human-readable algorithm name from a snake_case registry key."""
    if key in NAME_OVERRIDES:
        return NAME_OVERRIDES[key]
    return " ".join(w.capitalize() for w in key.replace("-", "_").split("_") if w)


def registry_keys(path: Path, registry: str) -> list[str]:
    """Return literal keys used to register classes in one source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == registry
                and decorator.func.attr == "register"
                and decorator.args
            ):
                continue
            key = ast.literal_eval(decorator.args[0])
            if isinstance(key, str):
                keys.append(key)
    return keys


def implementation_key(path: Path, registry: str, filename_key: str) -> str:
    """Choose the canonical registry key for an implementation source file."""
    keys = registry_keys(path, registry)
    if filename_key in keys:
        return filename_key
    if not keys:
        raise ValueError(f"No {registry}.register decorator in {path}")
    return keys[0]


# --------------------------------------------------------------------------- #
# Policy space enumeration
# --------------------------------------------------------------------------- #


def enumerate_pipeline() -> dict:
    """Enumerate the three registries straight from the source tree."""
    sel_dir = LOGIC_SRC / "mandatory_selection"
    construction_root = LOGIC_SRC / "route_construction"
    improvement_dir = LOGIC_SRC / "route_improvement"

    selection_names = {}
    for path in sel_dir.glob("selection_*.py"):
        filename_key = path.stem.split("_", 1)[1]
        key = implementation_key(path, "MandatorySelectionRegistry", filename_key)
        selection_names[key] = prettify(key)
    selection = [
        {"key": k, "name": v}
        for k, v in sorted(selection_names.items(), key=lambda item: item[1])
    ]

    families = []
    constructors_total = 0
    for fam_dir in sorted(construction_root.iterdir()):
        if not fam_dir.is_dir() or fam_dir.name == "base":
            continue
        policies = sorted(fam_dir.rglob("policy_*.py"))
        if not policies:
            continue
        constructors = []
        for p in policies:
            filename_key = p.stem[len("policy_"):]
            key = implementation_key(p, "RouteConstructorRegistry", filename_key)
            # The algorithm identity is the subdirectory, not the policy file
            # stem: e.g. meta_heuristics/simulated_annealing_neighborhood_search/
            # policy_sans.py is "Simulated Annealing Neighborhood Search", key
            # "sans". Benchmark keys keep their short paper names.
            parent = p.parent.name if p.parent != fam_dir else key
            name = NAME_OVERRIDES.get(key, prettify(parent))
            constructors.append({"key": key, "name": name})
        # Deduplicate keys (one policy file per algorithm, but be defensive).
        seen: set[str] = set()
        deduped = []
        for c in constructors:
            if c["key"] in seen:
                continue
            seen.add(c["key"])
            deduped.append(c)
        deduped.sort(key=lambda c: c["name"])
        constructors_total += len(deduped)
        families.append(
            {
                "family": FAMILY_LABELS.get(fam_dir.name, prettify(fam_dir.name)),
                "key": fam_dir.name,
                "constructors": deduped,
            }
        )
    families.sort(key=lambda f: f["family"])

    improvement = []
    for path in improvement_dir.glob("*.py"):
        if path.stem == "__init__":
            continue
        key = implementation_key(path, "RouteImproverRegistry", path.stem)
        improvement.append({"key": key, "name": prettify(key)})
    improvement.sort(key=lambda item: item["name"])

    return {
        "selection": selection,
        "construction": families,
        "improvement": improvement,
        "counts": {
            "selection": len(selection),
            "constructors": constructors_total,
            "families": len(families),
            "improvers": len(improvement),
        },
        "benchmark": {
            "selection": ["Last-Minute (CF70)", "Last-Minute (CF90)", "Look-Ahead",
                          "Service-Level (SL1)", "Service-Level (SL2)"],
            "constructors": ["ALNS", "BPC", "HGS", "ACO-HH", "PG-CLNS", "PSOMA",
                             "SANS", "SWC-TCF"],
            "improvers": ["Classical Local Search", "Fast-TSP"],
        },
    }


# --------------------------------------------------------------------------- #
# Results (reuse gen_paper_latex integrity machinery)
# --------------------------------------------------------------------------- #


def _clean_frame(horizons: dict[int, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = gpl.load_horizons(horizons)
    clean, degenerate = gpl.split_degenerate(raw)
    clean = gpl.drop_affected_cells(clean, degenerate)
    return clean, degenerate


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    return value


def _round(v: float, nd: int) -> float:
    return float(np.round(v, nd))


def build_results(clean: pd.DataFrame, degenerate: pd.DataFrame, cfg: dict) -> dict:
    """Balanced aggregates the website charts need, mirroring the paper tables."""
    horizon = 30

    def display(c: str) -> str:
        return cfg["constructor_labels"].get(c, c)

    # Constructors (balanced over the constructor marginal).
    sub = gpl.balance_marginal(clean[clean.horizon == horizon], "constructor")
    ctor_colors = cfg.get("constructor_colors", {})
    rows = []
    for name, grp in sub.groupby("constructor"):
        rows.append(
            {
                "name": display(name),
                "color": ctor_colors.get(name, ctor_colors.get(display(name))),
                "n": int(len(grp)),
                "kgkm": _round(grp.kgkm.mean(), 2),
                "kgkm_med": _round(grp.kgkm.median(), 2),
                "overflows": _round(grp.overflows.mean(), 1),
                "overflows_med": _round(grp.overflows.median(), 1),
                "km": _round(grp.km.mean(), 0),
                "time": _round(grp.time.mean(), 0),
            }
        )
    rows.sort(key=lambda r: r["kgkm"], reverse=True)

    # Selection variants (balanced over the variant marginal).
    vs = gpl.balance_marginal(clean[clean.horizon == horizon], "variant").copy()
    vs["variant"] = vs.strategy + vs.cf + vs.sl_var
    order = [v for v in cfg["variant_order"] if v in set(vs.variant)]
    strategies = []
    for v in order:
        grp = vs[vs.variant == v]
        strategies.append(
            {
                "key": v,
                "name": cfg["variant_labels"].get(v, v),
                "n": int(len(grp)),
                "kgkm": _round(grp.kgkm.mean(), 2),
                "overflows": _round(grp.overflows.mean(), 1),
                "km": _round(grp.km.mean(), 0),
                "time": _round(grp.time.mean(), 0),
            }
        )

    # Improvers (paired), with the conditional losses at N=350.
    pairs = gpl.improver_pairs(clean, horizon)
    delta_kgkm = pairs[("kgkm", "CLS")] - pairs[("kgkm", "FTSP")]
    ties = int(np.isclose(delta_kgkm, 0.0).sum())
    wins = int((delta_kgkm > 0).sum())
    losses = int((delta_kgkm < 0).sum())
    # Which constructor does each loss belong to?
    loss_mask = delta_kgkm < 0
    loss_by_ctor = {}
    if loss_mask.any():
        loss_deltas = delta_kgkm[loss_mask]
        constructors = loss_deltas.index.get_level_values("constructor")
        for c, d in zip(constructors, loss_deltas.tolist(), strict=True):
            loss_by_ctor.setdefault(c, []).append(float(d))
    loss_breakdown = sorted(
        (
            {"constructor": display(c), "n": len(v), "mean": _round(np.mean(v), 2)}
            for c, v in loss_by_ctor.items()
        ),
        key=lambda d: d["mean"],
    )
    improvers = {
        "pairs": len(pairs),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_delta_kgkm": _round(delta_kgkm.mean(), 2),
        "mean_km_cls": _round(pairs[("km", "CLS")].mean(), 0),
        "mean_km_ftsp": _round(pairs[("km", "FTSP")].mean(), 0),
        "loss_breakdown": loss_breakdown,
    }

    # Pareto points (per-constructor means) and the non-dominated front.
    agg = sub.groupby("constructor").agg(kgkm=("kgkm", "mean"), overflows=("overflows", "mean"))
    pareto_points = [
        {"name": display(c), "kgkm": _round(r.kgkm, 2), "overflows": _round(r.overflows, 1)}
        for c, r in agg.iterrows()
    ]
    front_names = set(gpl.pareto_front(agg).index)
    for p in pareto_points:
        p["front"] = p["name"] in {display(c) for c in front_names}

    # Scenario effects: distribution and network size (30d).
    scenario = {
        "by_dist": [],
        "by_N": [],
    }
    primary = clean[clean.horizon == horizon]
    balanced_dist = gpl.balance_marginal(primary, "dist")
    for dist, grp in balanced_dist.groupby("dist"):
        scenario["by_dist"].append(
            {
                "dist": cfg.get("dist_labels", {}).get(dist, dist),
                "kgkm": _round(grp.kgkm.mean(), 2),
                "overflows": _round(grp.overflows.mean(), 1),
                "kg_lost": _round(grp.kg_lost.mean(), 0),
            }
        )
    balanced_network = gpl.balance_marginal(primary, "network")
    for n, grp in balanced_network.groupby("N"):
        scenario["by_N"].append(
            {
                "N": int(n),
                "kgkm": _round(grp.kgkm.mean(), 2),
                "km": _round(grp.km.mean(), 0),
                "time": _round(grp.time.mean(), 0),
                "overflows": _round(grp.overflows.mean(), 1),
            }
        )

    # Constructor x scenario-cell heatmap (mean efficiency), for the website's
    # "where does each constructor win/lose" view. Cells are the six
    # network-distribution combinations; per the paper, Gamma-3 and Empirical
    # must not be pooled, so they are shown as distinct columns.
    hsub = gpl.balance_marginal(balanced_dist, "network").copy()
    dist_short = {"Empirical": "Emp", "Gamma-3": "Gamma-3"}
    hsub["cell"] = (
        hsub.city.map(cfg.get("city_short", {}))
        + "-"
        + hsub.N.astype(str)
        + "/"
        + hsub.dist.map(dist_short)
    )
    pivot = hsub.pivot_table(index="constructor", columns="cell", values="kgkm", aggfunc="mean")
    heatmap = {
        "columns": [str(c) for c in pivot.columns],
        "rows": [
            {"constructor": display(c), "values": [_round(v, 2) for v in row]}
            for c, row in pivot.iterrows()
        ],
    }

    # Paired horizon (30 vs 90) per constructor.
    paired = gpl.paired_horizon_frame(clean)
    horizon_rows = []
    total_pos = 0
    total_pairs = 0
    if not paired.empty:
        delta = paired[("kgkm", 90)] - paired[("kgkm", 30)]
        total_pos = int((delta > 0).sum())
        total_pairs = len(paired)
        for ctor, grp in paired.groupby("constructor"):
            horizon_rows.append(
                {
                    "constructor": display(ctor),
                    "pairs": int(len(grp)),
                    "kgkm_30": _round(grp[("kgkm", 30)].mean(), 2),
                    "kgkm_90": _round(grp[("kgkm", 90)].mean(), 2),
                    "overflows_30": _round(grp[("overflows", 30)].mean(), 1),
                    "overflows_90": _round(grp[("overflows", 90)].mean(), 1),
                }
            )
        horizon_rows.sort(key=lambda r: r["kgkm_30"], reverse=True)

    excluded = [
        {
            "constructor": display(r.constructor),
            "horizon": int(r.horizon),
            "city": r.city,
            "N": int(r.N),
            "policy": f"{r.strategy}{r.cf}{r.sl_var}/{r.improver}",
            "kg": _round(r.kg, 0),
            "shortfall": _round(r.shortfall, 2),
            "overflows": _round(r.overflows, 0),
        }
        for _, r in degenerate.sort_values("shortfall", ascending=False).iterrows()
    ]

    return {
        "meta": {
            "primary_horizon": horizon,
            "retained_runs": int(len(clean[clean.horizon == horizon])),
            "constructor_n": int(sub.groupby("constructor").size().min()),
        },
        "constructors": rows,
        "strategies": strategies,
        "improvers": improvers,
        "pareto": pareto_points,
        "scenario": scenario,
        "heatmap": heatmap,
        "horizon": {
            "pairs": total_pairs,
            "positive_pairs": total_pos,
            "mean_delta_kgkm": _round(float((paired[("kgkm", 90)] - paired[("kgkm", 30)]).mean()), 2)
            if not paired.empty else None,
            "by_constructor": horizon_rows,
        },
        "excluded": excluded,
        "colors": {
            "constructors": {display(k): v for k, v in ctor_colors.items()},
            "variants": cfg.get("variant_colors", {}),
            "improvers": cfg.get("improver_colors", {}),
        },
    }


# --------------------------------------------------------------------------- #
# Bin fills (selection widget)
# --------------------------------------------------------------------------- #


def _fill_history_path(scenario: dict) -> Path:
    """One fill_history spreadsheet per scenario. Demand is identical across
    constructors within a cell, so any run's fill history represents the cell."""
    base = OUTPUT_DIR / scenario["key"] / "gamma3"
    for strat_dir in ("lm_ftsp", "lm_cls"):
        fh = base / strat_dir / "fill_history"
        if fh.is_dir():
            files = sorted(fh.glob("*.xlsx"))
            if files:
                return files[0]
    raise SystemExit(f"No fill_history found for {scenario['key']}")


def _read_fill_history(path: Path) -> tuple[list[int], list[list[float]]]:
    """Return (bin_ids, fills[bin][day]) from a fill_history spreadsheet."""
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("openpyxl is required to read fill_history; run `uv sync`") from exc
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    n_bins = len(rows)
    fills = [[float(c) for c in row] for row in rows]
    # The spreadsheet has no bin-id column: it is a plain n_bins x n_days grid.
    # Bin identities are recovered from the distance matrix order in build_routes.
    return list(range(n_bins)), fills


def _daily_increments(fills: list[list[float]], n_days: int) -> list[list[float]]:
    """Recover the policy-independent daily waste increment per bin.

    The fill-history spreadsheet records the *realised* level after each day's
    collection, but the waste arriving each day is fixed by the demand
    realisation and does not depend on which bins were collected (the paper
    re-seeds waste generation per policy-and-day, so every policy faces the
    identical increment sequence). We can therefore recover the increments from
    the level trajectory: a drop between consecutive days marks a collection
    (reset to zero), so the next day's level is exactly that day's increment;
    otherwise the increment is the level rise.
    """
    inc = [[0.0] * n_days for _ in fills]
    for b, row in enumerate(fills):
        prev = row[0]
        inc[b][0] = row[0]
        for d in range(1, n_days):
            cur = row[d]
            if cur >= prev:
                inc[b][d] = cur - prev
            else:
                inc[b][d] = cur
            prev = cur
    return inc


def build_bin_fills() -> dict:
    """Real per-bin, per-day fill levels (percent 0-100) for the three networks."""
    scenarios = []
    for sc in SCENARIOS:
        path = _fill_history_path(sc)
        n_days = 30
        _, fills = _read_fill_history(path)
        # Trim to 30 columns and round to 0.1% -- plenty for a visual and it
        # keeps the wire payload small for the 350-bin scenario.
        grid = [[_round(v, 1) for v in row[:n_days]] for row in fills]
        increments = [[_round(v, 1) for v in row] for row in _daily_increments([r[:n_days] for r in fills], n_days)]
        scenarios.append(
            {
                "city": sc["city"],
                "N": sc["N"],
                "capacity": 100.0,
                "days": n_days,
                "fills": grid,  # fills[bin][day]
                "increments": increments,  # policy-independent daily waste
            }
        )
    return {"capacity": 100.0, "scenarios": scenarios}


# --------------------------------------------------------------------------- #
# Routes (4-D view)
# --------------------------------------------------------------------------- #


def _rep_run_path(scenario: dict) -> Path:
    rep = REPRESENTATIVE
    base = OUTPUT_DIR / scenario["key"] / "gamma3"
    strat_dir = "lm_cls" if rep["improver"] == "cls" else "lm_ftsp"
    log = base / strat_dir / f"log_{rep['strategy']}_{rep['constructor']}_custom_{rep['improver']}_1N.json"
    if not log.exists():
        # Fall back to any available representative-config run in that scenario.
        cands = sorted((base / strat_dir).glob(f"log_{rep['strategy']}_{rep['constructor']}_*.json"))
        if not cands:
            raise SystemExit(f"No representative run for {scenario['key']}")
        log = cands[0]
    return log


def build_routes() -> dict:
    """Per-day tours + MDS layout for one representative run per network."""
    scenarios = []
    for sc in SCENARIOS:
        # Several copies of each matrix are stored and the oversized ones do not
        # begin with the canonical block, so the copy must be chosen rather than
        # globbed for -- see gen_paper_latex.find_distance_matrix and issue #48.
        node_ids, dist = gpl.read_distance_matrix(gpl.find_distance_matrix(sc))
        coords = gpl.classical_mds(dist, ndim=2)
        # Normalise to a unit square for a stable canvas, preserving aspect.
        coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-12)

        log = _rep_run_path(sc)
        daily = json.loads(log.read_text(encoding="utf-8"))["daily"]["0"]

        depot_id = 0
        nodes = []
        for i, nid in enumerate(node_ids):
            nodes.append({"id": nid, "x": _round(coords[i, 0], 4), "y": _round(coords[i, 1], 4), "depot": nid == depot_id})
        # Index lookup for tour decoding.
        idx = {nid: i for i, nid in enumerate(node_ids)}

        days = []
        for d in range(len(daily["tour"])):
            tour = [int(x) for x in daily["tour"][d]]
            mandatory = [int(x) for x in daily["mandatory_nodes"][d]]
            days.append(
                {
                    "day": int(daily["day"][d]) if "day" in daily else d + 1,
                    "tour": [idx.get(x, x) for x in tour],
                    "mandatory": [idx.get(x, x) for x in mandatory],
                    "km": _round(daily["km"][d], 1),
                    "kg": _round(daily["kg"][d], 0),
                    "overflows": _round(daily["overflows"][d], 0),
                    "kgkm": _round(daily["kg/km"][d], 2),
                    "ncol": int(daily["ncol"][d]),
                }
            )

        scenarios.append(
            {
                "city": sc["city"],
                "N": sc["N"],
                "constructor": prettify(REPRESENTATIVE["constructor"]),
                "strategy": prettify(REPRESENTATIVE["strategy"].replace("last_minute_", "")),
                "improver": prettify(REPRESENTATIVE["improver"]),
                "layout": "classical-MDS embedding of the road-distance matrix",
                "nodes": nodes,
                "days": days,
            }
        )
    return {"scenarios": scenarios}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _write(path: Path, payload: object, force: bool) -> None:
    if path.exists() and not force:
        print(f"  Skipped (exists, use --force): {path.name}")
        return
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"  Wrote: {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="destination for the JSON files")
    args = ap.parse_args()

    cfg = load_json("paper_latex_config.json")
    sm = load_json("simulation_metadata.json")
    cfg["dist_labels"] = sm.get("dist_map", {})
    cfg["city_short"] = sm.get("city_short", {})
    cfg["constructor_colors"] = sm.get("constructor_colors", {})
    cfg["variant_colors"] = sm.get("variant_colors", {})
    cfg["improver_colors"] = sm.get("improver_colors", {})

    args.out_dir.mkdir(parents=True, exist_ok=True)

    horizons = {
        30: REPO_ROOT / "docs/private/global/simulation/simulation_summary.csv",
        90: REPO_ROOT / "docs/private/global/simulation/simulation_summary_90d.csv",
    }

    print("Enumerating policy space from the source tree:")
    _write(args.out_dir / "pipeline.json", enumerate_pipeline(), args.force)

    print("Building results from the simulation summaries:")
    clean, degenerate = _clean_frame(horizons)
    _write(args.out_dir / "results.json", build_results(clean, degenerate, cfg), args.force)

    print("Extracting bin fill histories:")
    _write(args.out_dir / "bin_fills.json", build_bin_fills(), args.force)

    print("Embedding road distances and exporting representative routes:")
    _write(args.out_dir / "routes.json", build_routes(), args.force)

    print("Done.")


if __name__ == "__main__":
    main()
