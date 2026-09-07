"""Sidebar control panel widgets for the dashboard.

This module provides reusable UI components for mode switching, filtering,
and data ingestion orchestration. It serves as the primary navigation
and control hub for all dashboard modes.

Example:
    mode = render_mode_selector()
    if mode == "training":
        render_training_monitor()

Attributes:
    jinja_env: Shared Jinja2 environment for HTML templates.
    template_dir: Path to UI templates directory.
"""

import os
from typing import Any, Dict, List, Tuple

import jinja2
import streamlit as st

from logic.src.constants import ROOT_DIR
from logic.src.ui import __version__
from logic.src.ui.services.data_loader import (
    discover_simulation_logs,
    discover_training_runs,
)

# Set up Jinja environment
template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))


def render_mode_selector() -> str:
    """Renders the main mode selector in the sidebar.

    Returns:
        str: Internal mode key (e.g., 'training', 'simulation', 'benchmark').
    """
    st.sidebar.title("🎛️ Control Tower")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "📊 Dashboard Mode",
        options=[
            "Training Monitor",
            "Simulation Digital Twin",
            "Simulation Summary",
            "Benchmark Analysis",
            "Data Explorer",
            "Experiment Tracker",
            "Algorithms Registry",
            "Hyperparameter Optimization Tracker",
        ],
        index=1,  # Default to Simulation
        help="Switch between training metrics, simulation visualization, simulation summary, benchmark analysis, data exploration, experiment tracking, and algorithms registry",
    )

    mode_map = {
        "Training Monitor": "training",
        "Simulation Digital Twin": "simulation",
        "Simulation Summary": "simulation_summary",
        "Benchmark Analysis": "benchmark",
        "Data Explorer": "data_explorer",
        "Experiment Tracker": "experiment_tracker",
        "Algorithms Registry": "algorithms",
        "Hyperparameter Optimization Tracker": "hpo_tracker",
    }
    return mode_map.get(mode, "simulation")


def render_auto_refresh_toggle() -> Tuple[bool, int]:
    """Renders auto-refresh controls for real-time monitoring.

    Returns:
        Tuple[bool, int]: (is_enabled, refresh_interval_seconds).
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Auto-Refresh")

    enabled = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
    interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=2,
        max_value=30,
        value=5,
        disabled=not enabled,
    )

    return enabled, interval


def render_training_controls(
    available_runs: List[str],
    available_metrics: List[str],
) -> Dict[str, Any]:
    """Renders controls for training monitor mode.

    Args:
        available_runs: List of available training run names.
        available_metrics: List of available metric names.

    Returns:
        Dict[str, Any]: Selected runs, metrics, and smoothing settings.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Training Controls")

    # Multi-select for runs
    selected_runs = st.sidebar.multiselect(
        "Select Runs to Compare",
        options=available_runs,
        default=available_runs[:2] if len(available_runs) >= 2 else available_runs,
        help="Select multiple runs to compare their metrics",
    )

    # Primary metric
    primary_metric = st.sidebar.selectbox(
        "Primary Metric (Left Y-Axis)",
        options=available_metrics,
        index=0 if available_metrics else 0,
    )

    # Secondary metric
    use_secondary = st.sidebar.checkbox("Show Secondary Metric", value=False)
    secondary_metric = None
    if use_secondary:
        remaining_metrics = [m for m in available_metrics if m != primary_metric]
        if remaining_metrics:
            secondary_metric = st.sidebar.selectbox(
                "Secondary Metric (Right Y-Axis)",
                options=remaining_metrics,
                index=0,
            )

    # X-axis selection
    x_axis = st.sidebar.radio(
        "X-Axis",
        options=["epoch", "step"],
        index=0,
        horizontal=True,
    )

    # Smoothing
    smoothing = st.sidebar.slider(
        "Smoothing (Moving Average)",
        min_value=1,
        max_value=50,
        value=1,
        help="Apply moving average smoothing to curves",
    )

    return {
        "selected_runs": selected_runs,
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "x_axis": x_axis,
        "smoothing": smoothing,
    }


def _render_filter_controls(available_logs: List[str], policies: List[str], samples: List[int]) -> Dict[str, Any]:
    """Renders log, policy, and sample selection controls.

    Args:
        available_logs: List of discovered log file names.
        policies: List of policy IDs found in the current log.
        samples: List of sample IDs found in the current log.

    Returns:
        Dict[str, Any]: Selected log file and filter constraints.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗺️ Simulation Controls")

    # Log file selector (always visible)
    selected_log = st.sidebar.selectbox(
        "Select Log File",
        options=available_logs,
        index=0 if available_logs else 0,
        help="Choose a simulation output file to visualize",
    )

    # Policy and sample filters in expander
    selected_policy = None
    selected_sample = None

    with st.sidebar.expander("Filters", expanded=False):
        if policies:
            selected_policy = st.selectbox(
                "Filter by Policy",
                options=["All"] + policies,
                index=0,
            )
            if selected_policy == "All":
                selected_policy = None

        if samples:
            sample_options = ["All"] + [str(s) for s in samples]
            selected_sample_str = st.selectbox(
                "Select Sample",
                options=sample_options,
                index=0,
            )
            if selected_sample_str != "All":
                selected_sample = int(selected_sample_str)

    return {
        "selected_log": selected_log,
        "selected_policy": selected_policy,
        "selected_sample": selected_sample,
    }


def _render_playback_controls(day_range: Tuple[int, int]) -> Dict[str, Any]:
    """Renders playback mode and temporal day slider controls.

    Args:
        day_range: (min_day, max_day) tuple from simulation data.

    Returns:
        Dict[str, Any]: Playback state ('is_live', 'selected_day').
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("▶️ Playback")

    playback_mode = st.sidebar.radio(
        "Mode",
        options=["Live (Latest)", "Replay (Slider)"],
        index=1,
        horizontal=True,
    )

    is_live = playback_mode == "Live (Latest)"

    # Day slider for replay mode
    selected_day = day_range[1]  # Default to latest
    if not is_live and day_range[0] < day_range[1]:
        selected_day = st.sidebar.slider(
            "Day",
            min_value=day_range[0],
            max_value=day_range[1],
            value=day_range[1],
        )

    return {"is_live": is_live, "selected_day": selected_day}


def _render_matrix_loader(distance_strategy: str) -> Dict[str, Any]:
    """Renders controls for loading custom distance matrices from disk.

    Args:
        distance_strategy: The active distance calculation strategy.

    Returns:
        Dict[str, Any]: Selected matrix and index files.
    """
    selected_matrix_file = None
    selected_index_file = None

    if distance_strategy == "load_matrix":
        with st.sidebar.expander("Distance Matrix", expanded=False):
            matrix_dir = os.path.join(ROOT_DIR, "data", "simulator", "distance_matrix")

            # Helper to recursively find files
            matrix_files = []
            if os.path.exists(matrix_dir):
                for root, _dirs, files in os.walk(matrix_dir):
                    for file in files:
                        if file.endswith((".csv", ".xlsx", ".txt")):
                            rel_path = os.path.relpath(os.path.join(root, file), matrix_dir)
                            matrix_files.append(rel_path)

            selected_matrix_file = st.selectbox(
                "Select Matrix File",
                options=sorted(matrix_files),
                index=0 if matrix_files else 0,
                help="Select the distance matrix file to load.",
            )

            # Bin Index File Selector
            bins_selection_dir = os.path.join(ROOT_DIR, "data", "simulator", "bins_selection")
            index_files = []
            if os.path.exists(bins_selection_dir):
                for root, _dirs, files in os.walk(bins_selection_dir):
                    for file in files:
                        if file.endswith(".json"):
                            rel_path = os.path.relpath(os.path.join(root, file), bins_selection_dir)
                            index_files.append(rel_path)

            selected_index_file = st.selectbox(
                "Select Bin Index File (Optional)",
                options=["None"] + sorted(index_files),
                index=0,
                help="Select a JSON file containing bin indices to subset the matrix.",
            )
            if selected_index_file == "None":
                selected_index_file = None

    return {
        "selected_matrix_file": selected_matrix_file,
        "selected_index_file": selected_index_file,
    }


def render_simulation_controls(
    available_logs: List[str],
    policies: List[str],
    samples: List[int],
    day_range: Tuple[int, int],
) -> Dict[str, Any]:
    """Renders controls for the simulation visualizer (Digital Twin) mode.

    Args:
        available_logs: List of available log file names.
        policies: List of policy names in the log.
        samples: List of sample IDs.
        day_range: (min_day, max_day) tuple.

    Returns:
        Dict[str, Any]: Aggregated selection options for filters and rendering.
    """
    # 1. Filter Controls
    filter_opts = _render_filter_controls(available_logs, policies, samples)

    # 2. Playback Controls
    playback_opts = _render_playback_controls(day_range)

    # 3. Display Options (in expander)
    show_route = True
    show_stats = True
    distance_strategy = "load_matrix"

    with st.sidebar.expander("Display Options", expanded=False):
        show_route = st.checkbox("Show Route Lines", value=True)
        show_stats = st.checkbox("Show Statistics", value=True)

        display_options = {
            "Haversine (Spherical)": "hsd",
            "Geodesic (WGS84)": "gdsc",
            "Euclidean (Planar)": "ogd",
            "Load Matrix (Custom)": "load_matrix",
        }

        strategy_label = st.selectbox(
            "Distance Strategy",
            options=list(display_options.keys()),
            index=3,
            help="Choose method for calculating edge distances.\n'Load Matrix' allows selecting a custom file.",
        )

        distance_strategy = display_options[strategy_label]

    # 4. Matrix Loader (Conditional, in expander)
    matrix_opts = _render_matrix_loader(distance_strategy)

    # Merge all options
    return {
        **filter_opts,
        **playback_opts,
        "show_route": show_route,
        "show_stats": show_stats,
        "distance_strategy": distance_strategy,
        **matrix_opts,
    }


def render_about_section() -> None:
    """Renders an 'About' section with system stats and version info."""
    st.sidebar.markdown("---")

    # Quick stats
    n_runs = len(discover_training_runs())
    n_logs = len(discover_simulation_logs())

    stats_parts = []
    if n_runs > 0:
        stats_parts.append(f"{n_runs} training run{'s' if n_runs != 1 else ''}")
    if n_logs > 0:
        stats_parts.append(f"{n_logs} simulation log{'s' if n_logs != 1 else ''}")

    # Render HTML from template
    template = jinja_env.get_template("sidebar_about.html")
    about_html = template.render(stats_parts=stats_parts, version=__version__)

    # Inject the final rendered HTML into Streamlit
    st.sidebar.markdown(about_html, unsafe_allow_html=True)
