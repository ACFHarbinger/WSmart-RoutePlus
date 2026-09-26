"""Geospatial visualization and route mapping components.

This module provides specialized tools for rendering interactive Leaflet
maps (via Folium) for waste collection routes. It includes support for
custom distance matrices, tour reconstruction, real-time bin status
indicators, and fill-level heatmaps for spatial analysis.

Example:
    render_map_view(display_entry, controls)

Attributes:
    load_custom_matrix: Loads and slices distance matrices from the filesystem.
    reconstruct_tour: Transforms raw node IDs into geospatial coordinates.
    render_map_view: Main UI component for rendering the interactive route map.
    render_bin_heatmap: Visualizes bin fill states as a spatial heatmap.
"""

import json
import os
from typing import Any, Dict, List, Optional

import folium
import jinja2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from logic.src.constants import ROOT_DIR
from logic.src.ui.components.maps import create_bin_heatmap, create_simulation_map

# Set up Jinja environment
template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "templates")
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))


def load_custom_matrix(controls: Dict[str, Any]) -> Any:
    """Loads and slices a custom distance matrix based on user configuration.

    Args:
        controls: User filter settings containing 'distance_strategy',
            'selected_matrix_file', and 'selected_sample'.

    Returns:
        Optional[pd.DataFrame]: The sliced distance matrix or None on failure.
    """
    dist_matrix = None
    if controls.get("distance_strategy") == "load_matrix":
        selected_file = controls.get("selected_matrix_file")
        if selected_file:
            matrix_path = os.path.join(ROOT_DIR, "data", "simulator", "distance_matrix", selected_file)
            if os.path.isfile(matrix_path):
                try:
                    if matrix_path.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(matrix_path, header=None)
                    else:
                        df = pd.read_csv(matrix_path, header=None)

                    loaded_data = df.to_numpy()
                    sliced_data = loaded_data[1:, 1:].astype(float)

                    selected_index_file = controls.get("selected_index_file")
                    if selected_index_file:
                        index_path = os.path.join(
                            ROOT_DIR,
                            "data",
                            "simulator",
                            "bins_selection",
                            selected_index_file,
                        )
                        if os.path.isfile(index_path):
                            with open(index_path, "r") as f:
                                indices_list = json.load(f)

                            sample_idx = controls.get("selected_sample", 0)
                            if 0 <= sample_idx < len(indices_list):
                                idx_list = indices_list[sample_idx]
                                target_indices = np.array([-1] + idx_list) + 1
                                max_idx = sliced_data.shape[0] - 1
                                valid_indices = target_indices[target_indices <= max_idx]
                                if len(valid_indices) < len(target_indices):
                                    st.warning(f"Some indices were out of bounds. Max idx: {max_idx}")
                                sliced_data = sliced_data[valid_indices[:, None], valid_indices]
                            else:
                                st.warning(f"Sample idx {sample_idx} out of range.")

                    dist_matrix = pd.DataFrame(sliced_data)

                except Exception as e:
                    st.error(f"Failed to load matrix {selected_file}: {e}")
    return dist_matrix


def reconstruct_tour(tour: List[Any], all_bin_coords: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Transforms a sequence of node identifiers into coordinate dictionaries.

    Args:
        tour: Sequence of node integers or existing coordinate dicts.
        all_bin_coords: Master list of all available bin coordinates.

    Returns:
        List[Dict[str, Any]]: Sequence of point dictionaries with lat/lng.
    """
    if not tour or isinstance(tour[0], dict):
        return tour

    if not all_bin_coords:
        return [{"id": int(ds_id)} for ds_id in tour]

    points_by_ds_id = {}
    for p in all_bin_coords:
        try:
            ds_id = p.get("dataset_id")
            ds_id_int = int(ds_id) if ds_id is not None else int(p.get("id", -100))
            points_by_ds_id[ds_id_int] = p
        except (ValueError, TypeError):
            continue
    return [points_by_ds_id.get(int(ds_id), {"id": int(ds_id)}) for ds_id in tour]


def render_map_view(display_entry: Any, controls: Dict[str, Any]) -> None:
    """Renders the comprehensive interactive route map.

    Args:
        display_entry: The simulation telemetry record for the current day.
        controls: User filter settings including routing and display options.
    """
    st.subheader("Route Map")

    data = display_entry.data
    tour = data.get("tour", [])
    bin_states = data.get("bin_state_c", [])
    collected = data.get("bin_state_collected", [])
    mandatory = data.get("mandatory")
    tour_indices = data.get("tour_indices")
    all_bin_coords = data.get("all_bin_coords")

    if not tour:
        st.info("No tour data available for this entry.")
        return

    dist_matrix = load_custom_matrix(controls)

    sim_map = create_simulation_map(
        tour=tour,
        bin_states=bin_states,
        served_indices=tour_indices,
        mandatory=mandatory,
        all_bin_coords=all_bin_coords,
        collected=collected if collected else None,
        vehicle_id=0,
        show_route=controls["show_route"],
        zoom_start=13,
        distance_matrix=dist_matrix,
        dist_strategy=controls.get("distance_strategy", "hsd"),
    )

    # Render HTML from template
    template = jinja_env.get_template("map_legend.html")
    legend_html = template.render()
    sim_map.get_root().html.add_child(folium.Element(legend_html))  # type: ignore
    st_folium(sim_map, width=None, height=1080, returned_objects=[])


def render_bin_heatmap(display_entry: Any) -> None:
    """Renders a spatial heatmap representing bin fill levels.

    Args:
        display_entry: The simulation telemetry record for the current day.
    """
    data = display_entry.data
    tour = data.get("tour", [])
    bin_states = data.get("bin_state_c", [])

    if not tour or not bin_states:
        st.info("No bin state or tour data available for heatmap.")
        return

    bin_locations: List[Dict[str, Any]] = []
    matched_states: List[float] = []

    for point in tour:
        if "lat" not in point or "lng" not in point:
            continue
        try:
            bin_id = int(point["id"])
        except (ValueError, TypeError):
            continue
        if bin_id == 0:
            continue

        fill = 50.0
        if 0 <= bin_id < len(bin_states):
            fill = bin_states[bin_id]
        elif 0 <= bin_id - 1 < len(bin_states):
            fill = bin_states[bin_id - 1]

        bin_locations.append(point)
        matched_states.append(fill)

    if not bin_locations:
        st.info("No bin locations found in tour data.")
        return

    heatmap = create_bin_heatmap(bin_locations, matched_states)
    st_folium(heatmap, width=None, height=400, returned_objects=[])
