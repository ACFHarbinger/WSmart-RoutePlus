"""
Data utilities module.

Attributes:
    check_extension: Check if the file extension is supported.
    collate_fn: Collate list of TensorDicts or dicts into batched TensorDict or dict.
    load_dataset: Load a dataset from a file.
    load_focus_coords: Load focus coordinates from a file.
    load_grid_base: Load grid base from a file.
    load_simulation_dataset: Load a simulation dataset from a file.
    load_td_dataset: Load a TensorDict dataset from a file.
    save_dataset: Save a dataset to a file.
    save_simulation_dataset: Save a simulation dataset to a file.
    save_td_dataset: Save a TensorDict dataset to a file.
    TensorDictStateWrapper: A wrapper for TensorDict that exposes methods expected by legacy model components (Decoders).

Example:
    >>> from logic.src.utils.data import (
    ...     check_extension,
    ...     collate_fn,
    ...     load_dataset,
    ...     load_focus_coords,
    ...     load_grid_base,
    ...     load_simulation_dataset,
    ...     load_td_dataset,
    ...     save_dataset,
    ...     save_simulation_dataset,
    ...     save_td_dataset,
    ...     TensorDictStateWrapper,
    ... )
"""

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "check_extension": "logic.src.utils.data.loader:check_extension",
    "collate_fn": "logic.src.utils.data.loader:collate_fn",
    "load_dataset": "logic.src.utils.data.loader:load_dataset",
    "load_focus_coords": "logic.src.utils.data.loader:load_focus_coords",
    "load_grid_base": "logic.src.utils.data.loader:load_grid_base",
    "load_simulation_dataset": "logic.src.utils.data.loader:load_simulation_dataset",
    "load_td_dataset": "logic.src.utils.data.loader:load_td_dataset",
    "save_dataset": "logic.src.utils.data.loader:save_dataset",
    "save_simulation_dataset": "logic.src.utils.data.loader:save_simulation_dataset",
    "save_td_dataset": "logic.src.utils.data.loader:save_td_dataset",
    "TensorDictStateWrapper": "logic.src.utils.data.td_state_wrapper:TensorDictStateWrapper",
}



__all__ = list(_LAZY)  # pyrefly: ignore[unresolvable-dunder-all]  # lazy re-exports; resolved through __getattr__


def __getattr__(name):
    return lazy_getattr(globals(), _LAZY, name)
