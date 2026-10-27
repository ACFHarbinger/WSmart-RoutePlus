"""
Data Processing and Transformation Pipeline.

This package provides coordinate formatting, DataFrame manipulation,
and model-input preparation for the WSmart+ Route simulator.

Submodules
----------
- ``formatting``:  Coordinate normalization methods (mmn, wmp, hdp, …)
- ``mapper``:      ``SimulationDataMapper`` class
- ``dataframes``:  DataFrame helpers (sort, sample, convert, export)
- ``processing``:  Data transform wrappers (process_data, process_coordinates, process_model_data)
- ``setup``:       Simulation bootstrap (setup_basedata, setup_dist_path_tup)

Attributes:
    dataframes: DataFrame helpers (sort, sample, convert, export)
    formatting: Coordinate normalization methods (mmn, wmp, hdp, …)
    mapper:      ``SimulationDataMapper`` class
    processing:  Data transform wrappers (process_data, process_coordinates, process_model_data)
    setup:       Simulation bootstrap (setup_basedata, setup_dist_path_tup)

Example:
    >>> from logic.src.data.processor.dataframes import convert_to_dict
    >>> convert_to_dict(pd.DataFrame({"Lng": [0, 1], "Lat": [0, 1]}))
    {'Lng': [0, 1], 'Lat': [0, 1]}
"""

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "convert_to_dict": "logic.src.data.processor.dataframes:convert_to_dict",
    "create_dataframe_from_matrix": "logic.src.data.processor.dataframes:create_dataframe_from_matrix",
    "get_df_types": "logic.src.data.processor.dataframes:get_df_types",
    "process_indices": "logic.src.data.processor.dataframes:process_indices",
    "sample_df": "logic.src.data.processor.dataframes:sample_df",
    "save_matrix_to_excel": "logic.src.data.processor.dataframes:save_matrix_to_excel",
    "setup_df": "logic.src.data.processor.dataframes:setup_df",
    "sort_dataframe": "logic.src.data.processor.dataframes:sort_dataframe",
    "format_coordinates": "logic.src.data.processor.formatting:format_coordinates",
    "SimulationDataMapper": "logic.src.data.processor.mapper:SimulationDataMapper",
    "process_coordinates": "logic.src.data.processor.processing:process_coordinates",
    "process_data": "logic.src.data.processor.processing:process_data",
    "process_model_data": "logic.src.data.processor.processing:process_model_data",
    "setup_basedata": "logic.src.data.processor.setup:setup_basedata",
    "setup_dist_path_tup": "logic.src.data.processor.setup:setup_dist_path_tup",
}

__all__ = [
    "convert_to_dict",
    "get_df_types",
    "process_indices",
    "sample_df",
    "save_matrix_to_excel",
    "setup_df",
    "sort_dataframe",
    "format_coordinates",
    "SimulationDataMapper",
    "process_coordinates",
    "process_data",
    "process_model_data",
    "setup_basedata",
    "setup_dist_path_tup",
]


def __getattr__(name):
    return lazy_getattr(globals(), _LAZY, name)
