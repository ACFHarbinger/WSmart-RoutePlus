# Data Management, Generation & Transformation

> **Status (Q-22):** narrative guide. The API reference is generated into
> `docs/api/` by `tools/gen_api_docs.py`; this file is not the source of
> truth for signatures and module inventories.


**Module**: `logic/src/data`
**Purpose**: Comprehensive technical specification of the WSmart-Route data lifecycle—from procedural generation to high-performance dataset management.
**Version**: 3.0
**Last Updated**: February 2026

---

## 1. Overview

The `data` module is responsible for the entire lifecycle of VRP data—from spatial modeling and procedural generation to dataset management and geometric augmentation. It ensures that problem instances are standardized, reproducible, and optimized for high-performance neural training.

---

## Table of Contents

1.  [**Overview**](#1-overview)
2.  [**Instance Builders**](#2-instance-builders-builderspy)
3.  [**Data Transformations**](#3-data-transformations-transformspy)
4.  [**Dataset Management**](#4-dataset-management-datasets)
5.  [**Spatial & Statistical Distributions**](#5-spatial--statistical-distributions-distributions)
6.  [**Data Generation Pipeline**](#6-data-generation-pipeline-generators)
7.  [**Technical Summary**](#7-technical-summary-table)
8.  [**Usage Example**](#8-usage-example-integrated-workflow)

---

## 2. Instance Builders (`builders.py`)

The `builders.py` module provides the `VRPInstanceBuilder` class, a powerful tool for generating standardized Vehicle Routing Problem (VRP) instances programmatically.

### `VRPInstanceBuilder` Class

The builder utilizes a fluent interface, allowing for complex configuration in a readable, chained format.

#### Initialization: `__init__`

```python
def __init__(self, data=None, depot_idx=0, vehicle_cap=100.0, customers=None, dimension=0, coords=None)
```

- **`data`**: Any raw dictionary or dataframe containing problem-specific entries.
- **`depot_idx`**: The integer index designating the depot (starting and ending point). Defaults to `0`.
- **`vehicle_cap`**: The maximum capacity for all vehicles. Crucial for WCVRP and CWC problems.
- **`customers`**: Explicit list of node indices designated as customers.
- **`dimension`**: Final node count (depot + customers).
- **`coords`**: Initial coordinate array if pre-defined; otherwise, these are generated during the `.build()` phase.

#### Configuration Methods (Fluent API)

| Method             | Parameters        | Description                                                                                          |
| :----------------- | :---------------- | :--------------------------------------------------------------------------------------------------- |
| `set_problem_size` | `size: int`       | Sets the number of customers (`num_loc`). The final dimension will be `size + 1`.                    |
| `set_distribution` | `dist: str`       | Sets the waste fill-rate distribution (e.g., `gamma1`-`4`, `unif`, `const`, `dist`, `emp`).          |
| `set_area`         | `area: str`       | Sets the geographic context (e.g., `"Rio Maior"`) which influences depot mapping and bounding boxes. |
| `set_waste_type`   | `waste_type: str` | Configures attributes specific to a waste category (e.g., `"Glass"`, `"Organic"`).                   |
| `set_focus_graph`  | `graph, size`     | Experimental: Sets a high-density "focus" graph for generating clustered urban environments.         |

#### Core Execution: `build`

```python
def build(self, batch_size: int) -> TensorDict
```

This is the terminal method of the builder. It performs the following sequence:

1. **Coordinate Generation**: Samples `(x, y)` locations based on the selected `area` and spatial logic.
2. **Waste Generation**: Calculates initial node waste/profits using the specified waste distribution.
3. **Internal Normalization**: Ensures all coordinates and costs are scaled to the standard range (usually `0.0` to `1.0`).
4. **TensorDict Preparation**: Aggregates all tensors into a single `TensorDict` with keys: `locs`, `depot`, `waste`, `capacity`, and problem-specific metadata.

---

## 3. Data Transformations (`transforms.py`)

This module is responsible for geometric data augmentation and feature normalization, ensuring that models are robust to spatial variances.

### `AugmentTransform` Class

Used typically as a wrapper or inside a `DataLoader` to expand the training/inference variety.

#### Parameters:

- **`num_augment`**: The expansion factor (e.g., `8` for complete dihedral symmetry).
- **`augment_fn`**: The strategy to use:
  - `"symmetric"`: Simple reflection.
  - `"dihedral8"`: Full 8-way rotations and reflections (D4 group).
- **`normalize`**: If `True`, applies `min_max_normalize` after the transformation happens.
- **`feats`**: List of dictionary keys to augment (Default: `["locs"]`).

#### Execution: `__call__`

```python
def __call__(self, td: TensorDict) -> TensorDict:
```

1. **Batchification**: Uses `batchify` to repeat the input `td` by `num_augment`.
2. **Symmetry Application**: Applies the chosen geometric function to the target features.
3. **Re-Normalization**: (Optional) Ensures augmented coordinates fit within the unit square.
4. **Return**: A batch expanded by `num_augment`.

### Functional Utilities

#### `min_max_normalize(tensor: torch.Tensor) -> torch.Tensor`

Scales any input tensor such that its minimum value is `0.0` and its maximum is `1.0`. Essential for ensuring that generated coordinates stay within the expected unit box.

#### `batchify(td: TensorDict, n: int) -> TensorDict`

Calls `td.unsqueeze(1).expand(-1, n, ...).flatten(0, 1)` to efficiently create `n` copies of each batch item. This is much faster than manual looping in Python.

#### `get_augment_function(name: str)`

Maps string identifiers (`symmetric`, `dihedral8`) to their respective implementation functions.

---

## 4. Dataset Management (`datasets/`)

This sub-module provides high-performance PyTorch `Dataset` wrappers designed to handle `TensorDict` objects and complex training metadata.

### Standard Storage Datasets

#### `TensorDictDataset` (`td_dataset.py`)

The primary dataset class. It wraps a single `TensorDict` and treats its first dimension as the batch/sample axis.

- **Methods**:
  - `load(path: str)`: Static factory to load `.td` or `.pkl` files using `torch.load`.
  - `save(path: str)`: Persists the dataset to disk.
  - `__getitem__(index)`: Returns a slice of the internal `TensorDict`.

#### `BaselineDataset` (`baseline_dataset.py`)

Used for training Reinforcement Learning models with precomputed baselines.

- **Logic**: Combines a standard dataset with a `baseline` tensor.
- **Output**: `{"data": dataset[index], "baseline": baseline[index]}`.

#### `ExtraKeyDataset` (`extra_key_dataset.py`)

A flexible extension for appending arbitrary tensors (like auxiliary labels or indices) to a dataset.

- **Input**: A dictionary of extra tensors.
- **Length Validation**: Strictly enforces that all extra tensors have the same length as the base dataset.

### ⚡ Performance & On-the-fly Datasets

#### `GeneratorDataset` (`generator_dataset.py`)

A "virtual" dataset that generates unique samples every time an index is accessed.

- **Why?**: Prevents overfitting because the model never sees the same instance twice.
- **Constraint**: Ignores the `index` parameter in `__getitem__`.

#### `FastGeneratorDataset` (`fast_gen_dataset.py`)

An optimized variant of the `GeneratorDataset` that uses internal buffering. Instead of calling the builder for every single sample, it generates small chunks of data and yields them, significantly reducing the overhead of Python-to-C++ tensor dispatch.

#### `FastTensorDictDataset` (`fast_td_dataset.py`)

Optimizes memory access patterns for massive on-disk datasets by using more efficient slicing methods than the standard `TensorDictDataset`.

---

## 5. Spatial & Statistical Distributions (`distributions/`)

This module defines the "physics" and "topography" of the VRP environment, handling how node locations and bin attributes are sampled.

### Spatial Topography (Node Locations)

These classes generate `(x, y)` coordinate tensors of shape `(batch, num_loc, 2)`.

#### `GaussianMixture` (`spatial_gaussian_mixture.py`)

- **Philosophy**: Simulates nodes clustered around certain "neighborhoods" or urban centers.
- **Parameters**: `n_components` (number of blobs) and `std` (tightness).

#### `ClusterUniform` (`spatial_cluster.py`)

- **Philosophy**: Models a realistic city where some areas are dense (clusters) and others are sparsely populated (uniform noise).
- **Operation**: Places a fixed number of nodes uniformly, then "attracts" a subset of them toward randomly chosen centroids.

#### `SpatialMixed` / `SpatialMixMulti` (`spatial_mixed.py`)

- **Philosophy**: Designed for Meta-RL and Robust RL. It mixes different spatial patterns (Uniform, Gaussian, Cluster) within a single batch so the model must learn a generalizable strategy.

### Statistical Physics (Bin Attributes)

These classes define the profits, and waste fill-rates.

#### `Gamma` (`statistical.py`)

Samples from a Gamma distribution using `alpha` (shape) and `beta` (rate).

- **Significance**: Gamma distributions are highly accurate for modeling recycling bin fill rates, as most bins stay at low levels for long periods before suddenly reaching a critical state.

#### `Empirical` (`empirical.py`)

Uses real-world data points to simulate environment states.

- **Sources**: Can sample from a `Bins` simulation object or a historic data file.
- **Workflow**: If using a `Bins` object, it calls `stochasticFilling()` to simulate the temporal progression of bin levels.

---

## 6. Data Generation Pipeline (`generators/`)

The generators orchestrate the previous components to produce massive benchmarking datasets (train/val/test).

### `generate_datasets` (`datasets.py`)

The main entry point for the CLI `gen_data` command.

1. **Reproduction**: Sets the global random seeds for `random`, `numpy`, and `torch` to facilitate identical results across runs.
2. **Complexity Orchestration**: Loops through every requested problem type (`vrpp`, `wcvrp`, `swcvrp`), node size (`20`, `50`, `100`), and distribution.
3. **Parallel Logic**: (Where applicable) manages the distribution of work to leverage multiple CPU cores.
4. **Naming Convention**: Implements a strict directory structure (e.g., `data/vrpp/train_50_gamma1.pkl`) to ensure other components can find the data.

### `validate_dataset_args` (`validators.py`)

A critical security and logic gate that runs before generation starts.

- **Parameter Collision**: Alerts the user if they try to set a fixed `filename` while requesting multiple datasets.
- **Completeness Check**: Verifies that stochastic problems (`swcvrp`) have their required `mu` and `sigma` parameters defined.
- **Registry Lookup**: Checks input area names (e.g., `"Rio Maior"`) and waste types against internal mappings to prevent "KeyError" later in the process.
- **Sanitization**: Uses regex to clean string inputs, ensuring compatibility with the file system and registries.

---

## 7. Technical Summary Table

| Feature               | Implementation Component           | Outcome                                  |
| :-------------------- | :--------------------------------- | :--------------------------------------- |
| **Data Construction** | `builders.VRPInstanceBuilder`      | Standardized `TensorDict` output         |
| **Spatial Modeling**  | `distributions.GaussianMixture`    | Clustered city topologies                |
| **Fill-rate Logic**   | `distributions.Gamma`              | Realistic stochastic fill profiles       |
| **Robust Training**   | `transforms.AugmentTransform`      | 8x data expansion (Dihedral)             |
| **High Throughput**   | `datasets.FastGeneratorDataset`    | Unlimited data with mini-batch buffering |
| **Reliability**       | `generators.validate_dataset_args` | Prevention of malformed datasets         |

> [!CAUTION]
> **Performance Warning**: Generating very large datasets (e.g., 1 Million samples for size 100) can consume significant RAM. Use `FastGeneratorDataset` or write directly to disk in chunks to avoid OOM issues.

---

## 8. Usage Example: Integrated Workflow

```python
from logic.src.data.builders import VRPInstanceBuilder
from logic.src.data.datasets import FastGeneratorDataset
from logic.src.data.transforms import AugmentTransform

# 1. Setup the Generator logic
builder = VRPInstanceBuilder().set_problem_size(50).set_distribution("gamma2")

# 2. Wrap in a Fast Dataset (virtual size 10,000)
dataset = FastGeneratorDataset(generator=builder.build, size=10000)

# 3. Setup Augmentation
augmentor = AugmentTransform(num_augment=8)

# 4. Fetch and Transform
sample_td = dataset[0] # Generates 1 instance
augmented_batch = augmentor(sample_td) # Now have 8 instances (D4 symmetry)
```
