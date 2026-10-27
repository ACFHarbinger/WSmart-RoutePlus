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

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "td_kwargs": "logic.src.utils.data.td_utils:td_kwargs",
    "tensordict_collate_fn": "logic.src.utils.data.td_utils:tensordict_collate_fn",
    "BaselineDataset": "logic.src.data.datasets.pytorch.baseline_dataset:BaselineDataset",
    "ExtraKeyDataset": "logic.src.data.datasets.pytorch.extra_key_dataset:ExtraKeyDataset",
    "TensorDictDatasetFastGeneration": "logic.src.data.datasets.pytorch.fast_gen_dataset:TensorDictDatasetFastGeneration",
    "FastTdDataset": "logic.src.data.datasets.pytorch.fast_td_dataset:FastTdDataset",
    "GeneratorDataset": "logic.src.data.datasets.pytorch.generator_dataset:GeneratorDataset",
    "TensorDictDataset": "logic.src.data.datasets.pytorch.td_dataset:TensorDictDataset",
    "GenerativeDataset": "logic.src.data.datasets.simulation.gen_dataset:GenerativeDataset",
    "NumpyPickleDataset": "logic.src.data.datasets.simulation.np_pkl_dataset:NumpyPickleDataset",
    "NumpyDictDataset": "logic.src.data.datasets.simulation.npz_dataset:NumpyDictDataset",
    "PandasCsvDataset": "logic.src.data.datasets.simulation.pd_csv_dataset:PandasCsvDataset",
    "PandasExcelDataset": "logic.src.data.datasets.simulation.pd_xlsx_dataset:PandasExcelDataset",
    "SimulationDataset": "logic.src.data.datasets.simulation.sim_dataset:SimulationDataset",
    "HtmlSimulationDataset": "logic.src.data.datasets.web.html_sim_dataset:HtmlSimulationDataset",
}

__all__ = [
    "td_kwargs",
    "tensordict_collate_fn",
    "BaselineDataset",
    "TensorDictDataset",
    "GenerativeDataset",
    "NumpyPickleDataset",
    "NumpyDictDataset",
    "PandasCsvDataset",
    "PandasExcelDataset",
    "SimulationDataset",
    "HtmlSimulationDataset",
]


def __getattr__(name):
    return lazy_getattr(globals(), _LAZY, name)
