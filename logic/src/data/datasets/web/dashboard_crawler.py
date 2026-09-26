"""
Crawler for WSmart+ Route HTML dashboards.

Extracts the "Todos os Locais" table from a dashboard HTML file or URL and
exposes two usage modes:
  - Export to CSV / Excel for offline use.
  - Return (data_df, coordinates_df) tuples compatible with SimulationRepository.

Columns extracted (all except "Viagem / Origem"):
  Local, % vol. atual, % vol. média, Acum. (%/dia), Volume (kg), Nº Cont.,
  Latitude, Longitude
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError as exc:
    raise ImportError("beautifulsoup4 is required: uv add beautifulsoup4") from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EXCLUDED_COL = "Viagem / Origem"

_COLUMN_RENAME: Dict[str, str] = {
    "Local": "ID",
    "% vol. atual": "Fill_Pct",
    "% vol. média": "Fill_Avg_Pct",
    "Acum. (%/dia)": "Acum_Rate_Pct",
    "Volume (kg)": "Volume_kg",
    "Nº Cont.": "N_Containers",
    "Latitude": "Lat",
    "Longitude": "Lng",
}


def _load_html(source: str) -> str:
    """Return raw HTML from a file path or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        try:
            import urllib.request

            with urllib.request.urlopen(source, timeout=30) as resp:  # noqa: S310
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch URL '{source}': {exc}") from exc
    path = os.path.expanduser(source)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HTML file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _find_todos_os_locais_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    Locate the 'Todos os Locais' table inside the dashboard HTML.

    Strategy: find the flow-section whose tab-bar contains a button with the
    text 'Todos os Locais', then pick the tab-content inside that section
    whose table header includes 'Viagem / Origem'.
    """
    for section in soup.find_all(class_="flow-section"):
        tab_bar = section.find(class_="tab-bar")
        if tab_bar is None:
            continue
        if not any("Todos os Locais" in btn.get_text() for btn in tab_bar.find_all("button")):
            continue
        # Found the right section — look for the tab-content with the full table
        for tc in section.find_all(class_="tab-content"):
            table = tc.find("table")
            if table is None:
                continue
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            if _EXCLUDED_COL in headers:
                return table
    return None


def _parse_table(table: BeautifulSoup) -> pd.DataFrame:
    """Convert a BeautifulSoup table into a cleaned DataFrame."""
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    keep_indices = [i for i, h in enumerate(headers) if h != _EXCLUDED_COL]
    keep_headers = [headers[i] for i in keep_indices]

    rows: List[List[str]] = []
    for tr in table.find_all("tr")[1:]:  # skip header row
        cells = tr.find_all("td")
        if not cells:
            continue
        row = [cells[i].get_text(strip=True) if i < len(cells) else "" for i in keep_indices]
        rows.append(row)

    df = pd.DataFrame(rows, columns=keep_headers)

    # Strip "%" suffixes and cast numerics
    for col in ["% vol. atual", "% vol. média"]:
        if col in df.columns:
            df[col] = df[col].str.replace(r"\s*%", "", regex=True).str.strip()

    # Normalize Portuguese-locale comma decimal separators to dots before
    # numeric conversion (e.g. "39,156337" → "39.156337").
    numeric_cols = ["% vol. atual", "% vol. média", "Acum. (%/dia)", "Volume (kg)", "Nº Cont.", "Latitude", "Longitude"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Local" in df.columns:
        df["Local"] = pd.to_numeric(df["Local"], errors="coerce").astype("Int64")

    return df.rename(columns=_COLUMN_RENAME)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_dataframe(source: str) -> pd.DataFrame:
    """
    Parse the dashboard and return the 'Todos os Locais' table as a DataFrame.

    Columns: ID, Fill_Pct, Fill_Avg_Pct, Acum_Rate_Pct, Volume_kg,
             N_Containers, Lat, Lng

    Args:
        source: Path to an HTML file or a URL.

    Returns:
        DataFrame with one row per location.
    """
    html = _load_html(source)
    soup = BeautifulSoup(html, "html.parser")
    table = _find_todos_os_locais_table(soup)
    if table is None:
        raise ValueError(
            "Could not find the 'Todos os Locais' table in the dashboard. "
            "Make sure the source is a valid WSmart+ Route dashboard HTML."
        )
    return _parse_table(table)


def to_csv(source: str, output_path: str, sep: str = ",") -> str:
    """Export 'Todos os Locais' data to a CSV file.

    Args:
        source: Path to an HTML file or a URL.
        output_path: Destination CSV file path.
        sep: Column separator (default: comma).

    Returns:
        Absolute path to the written file.
    """
    df = extract_dataframe(source)
    df.to_csv(output_path, index=False, sep=sep)
    return os.path.abspath(output_path)


def to_excel(source: str, output_path: str, sheet_name: str = "Todos os Locais") -> str:
    """Export 'Todos os Locais' data to an Excel file.

    Args:
        source: Path to an HTML file or a URL.
        output_path: Destination .xlsx file path.
        sheet_name: Sheet name inside the workbook.

    Returns:
        Absolute path to the written file.
    """
    df = extract_dataframe(source)
    df.to_excel(output_path, index=False, sheet_name=sheet_name)
    return os.path.abspath(output_path)


def to_simulation_data(
    source: str,
    n_bins: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (data_df, coordinates_df) ready for use with SimulationRepository.

    The tuple mirrors what FileSystemRepository / DatasetRepository return from
    ``get_simulator_data``:
      - ``data_df``: columns [ID, Stock, Accum_Rate]
          * Stock      = Fill_Pct / 100  (current fill level in [0, 1])
          * Accum_Rate = Acum_Rate_Pct / 100  (daily fill increment in [0, 1])
      - ``coordinates_df``: columns [ID, Lat, Lng]

    Args:
        source: Path to an HTML file or a URL.
        n_bins: If provided, randomly sample this many bins (reproducible seed).

    Returns:
        (data_df, coordinates_df)
    """
    df = extract_dataframe(source)

    if n_bins is not None:
        if n_bins > len(df):
            raise ValueError(f"n_bins={n_bins} exceeds available locations ({len(df)}).")
        df = df.sample(n=n_bins, random_state=42).reset_index(drop=True)

    data_df = (
        pd.DataFrame(
            {
                "ID": df["ID"],
                "Stock": df["Fill_Pct"] / 100.0,
                "Accum_Rate": df["Acum_Rate_Pct"] / 100.0,
            }
        )
        .sort_values("ID")
        .reset_index(drop=True)
    )

    coordinates_df = (
        pd.DataFrame(
            {
                "ID": df["ID"],
                "Lat": df["Lat"],
                "Lng": df["Lng"],
            }
        )
        .sort_values("ID")
        .reset_index(drop=True)
    )

    return data_df, coordinates_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dashboard_crawler",
        description="Extract 'Todos os Locais' data from a WSmart+ Route dashboard.",
    )
    p.add_argument("source", help="Path to HTML file or dashboard URL")
    sub = p.add_subparsers(dest="command", required=True)

    csv_p = sub.add_parser("csv", help="Export to CSV")
    csv_p.add_argument("output", help="Output CSV file path")
    csv_p.add_argument("--sep", default=",", help="Column separator (default: ',')")

    xls_p = sub.add_parser("excel", help="Export to Excel (.xlsx)")
    xls_p.add_argument("output", help="Output .xlsx file path")
    xls_p.add_argument("--sheet", default="Todos os Locais", help="Sheet name")

    info_p = sub.add_parser("info", help="Print extracted data summary")
    info_p.add_argument("--n-bins", type=int, default=None, help="Limit to N bins")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "csv":
        out = to_csv(args.source, args.output, sep=args.sep)
        print(f"Saved CSV → {out}")

    elif args.command == "excel":
        out = to_excel(args.source, args.output, sheet_name=args.sheet)
        print(f"Saved Excel → {out}")

    elif args.command == "info":
        data_df, coords_df = to_simulation_data(args.source, n_bins=args.n_bins)
        print(f"Locations : {len(data_df)}")
        print(f"Stock range     : [{data_df['Stock'].min():.3f}, {data_df['Stock'].max():.3f}]")
        print(f"Accum_Rate range: [{data_df['Accum_Rate'].min():.4f}, {data_df['Accum_Rate'].max():.4f}]")
        print(f"Lat range : [{coords_df['Lat'].min():.6f}, {coords_df['Lat'].max():.6f}]")
        print(f"Lng range : [{coords_df['Lng'].min():.6f}, {coords_df['Lng'].max():.6f}]")
        print("\ndata_df (first 5):")
        print(data_df.head())
        print("\ncoordinates_df (first 5):")
        print(coords_df.head())


if __name__ == "__main__":
    main()
