"""
Dataset classes for WSmart-Route.

Attributes:
    BaselineDataset: BaseLine dataset for training.
    ExtraKeyDataset: Extra Key dataset for training.
    TensorDictDatasetFastGeneration: TensorDict dataset for fast generation.
    FastTdDataset: Fast TensorDict dataset for training.
    GeneratorDataset: Generator dataset for training.
    TensorDictDataset: TensorDict dataset for training.
    GenerativeDataset: Generative dataset for simulation.
    NumpyDictDataset: NumpyDict dataset for simulation.
    NumpyPickleDataset: NumpyPickle dataset for simulation.
    PandasCsvDataset: PandasCsv dataset for simulation.
    PandasExcelDataset: PandasExcel dataset for simulation.
    HtmlSimulationDataset: HtmlSimulationDataset dataset for simulation.
    SimulationDataset: Simulation dataset for simulation.
    td_kwargs: Generator dataset for training.
    tensordict_collate_fn: Generator dataset for training.

Example:
    >>> from logic.src.data.datasets import td_kwargs, tensordict_collate_fn, BaselineDataset
    >>> td_kwargs = td_kwargs()
    >>> tensordict_collate_fn = tensordict_collate_fn()
    >>> dataset = BaselineDataset(**td_kwargs)
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=tensordict_collate_fn)
    >>> for batch in dataloader:
    ...     print(batch)
    ...     break
"""

from typing import Any

# Import from utils.data.td_utils, not from a local td_utils
from logic.src.utils.data.td_utils import td_kwargs, tensordict_collate_fn

from .pytorch.baseline_dataset import BaselineDataset
from .pytorch.extra_key_dataset import ExtraKeyDataset
from .pytorch.fast_gen_dataset import TensorDictDatasetFastGeneration
from .pytorch.fast_td_dataset import FastTdDataset
from .pytorch.generator_dataset import GeneratorDataset
from .pytorch.td_dataset import TensorDictDataset
from .simulation.gen_dataset import GenerativeDataset
from .simulation.np_pkl_dataset import NumpyPickleDataset
from .simulation.npz_dataset import NumpyDictDataset
from .simulation.pd_csv_dataset import PandasCsvDataset
from .simulation.pd_xlsx_dataset import PandasExcelDataset
from .simulation.sim_dataset import SimulationDataset

__all__ = [
    "td_kwargs",
    "tensordict_collate_fn",
    # PyTorch datasets
    "BaselineDataset",
    "ExtraKeyDataset",
    "TensorDictDatasetFastGeneration",
    "FastTdDataset",
    "GeneratorDataset",
    "TensorDictDataset",
    # Simulation datasets
    "SimulationDataset",
    "NumpyDictDataset",
    "NumpyPickleDataset",
    "PandasExcelDataset",
    "PandasCsvDataset",
    "HtmlSimulationDataset",
    "GenerativeDataset",
]


def __getattr__(name: str) -> Any:
    """Load the dashboard HTML dataset only when it is requested.

    ``HtmlSimulationDataset`` pulls in the optional beautifulsoup4 crawler.
    Importing it at module level made ``Bins`` and the pytest plugin chain
    fail in environments that never parse dashboards.
    """
    if name == "HtmlSimulationDataset":
        from .web.html_sim_dataset import HtmlSimulationDataset

        return HtmlSimulationDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
