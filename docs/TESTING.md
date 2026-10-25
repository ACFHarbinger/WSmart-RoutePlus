# WSmart+ Route Testing Guide

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Gurobi](https://img.shields.io/badge/Gurobi-11.0-ED1C24?logo=gurobi&logoColor=white)](https://www.gurobi.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MyPy](https://img.shields.io/badge/MyPy-checked-2f4f4f.svg)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/pytest-testing-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-50%25-green.svg)](https://coverage.readthedocs.io/)
[![CI](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml/badge.svg)](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml)

> **Version**: 2.0 (Comprehensive Edition)
> **Target Audience**: Developers, QA Engineers, Contributors

This document provides a comprehensive guide to the WSmart-Route testing infrastructure, including test organization, execution strategies, fixture patterns, and best practices for writing and maintaining tests.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Test Architecture](#2-test-architecture)
3. [Running Tests](#3-running-tests)
4. [Test Categories and Markers](#4-test-categories-and-markers)
5. [Fixture System](#5-fixture-system)
6. [Writing Tests](#6-writing-tests)
7. [Coverage Requirements](#7-coverage-requirements)
8. [Continuous Integration](#8-continuous-integration)
9. [Test Data Management](#9-test-data-management)
10. [Debugging Failed Tests](#10-debugging-failed-tests)
11. [Performance Testing](#11-performance-testing)
12. [Mutation Testing](#12-mutation-testing)
13. [Advanced Performance Benchmarking](#13-advanced-performance-benchmarking)
14. [Solver Contract Tests](#14-solver-contract-tests)
15. [ELK Stack Logging](#15-elk-stack-logging)

---

## 1. Overview

WSmart-Route utilizes `pytest` as the primary testing framework with comprehensive support for:

- **Unit Tests**: Isolated component testing
- **Integration Tests**: Cross-module interaction testing
- **Simulation Tests**: End-to-end policy evaluation

### Coverage Requirements

| Metric           | Minimum | Target |
| ---------------- | ------- | ------ |
| Overall Coverage | 50%     | 70%    |
| Critical Paths   | 70%     | 90%    |
| New Code         | 60%     | 75%    |

---

## 2. Test Architecture

### Directory Structure

```
WSmart-Route/
├── logic/
│   └── test/
│       ├── conftest.py              # Shared fixtures and configuration
│       ├── fixtures/                # Fixture modules
│       │   ├── arg_fixtures.py      # CLI argument fixtures
│       │   ├── data_fixtures.py     # Data generation fixtures
│       │   ├── eval_fixtures.py     # Evaluation fixtures
│       │   ├── file_system_fixtures.py
│       │   ├── integration_fixtures.py
│       │   ├── input_fixtures.py
│       │   ├── model_fixtures.py    # Neural model fixtures
│       │   ├── mrl_fixtures.py      # Meta-RL fixtures
│       │   ├── policy_aux_fixtures.py
│       │   ├── policy_fixtures.py   # Policy fixtures
│       │   ├── sim_fixtures.py      # Simulator fixtures
│       │   └── vectorized_policy_fixtures.py
│       ├── test_arg_parser.py       # CLI argument parsing
│       ├── test_definitions.py      # Project constants
│       ├── test_eval.py             # Model evaluation
│       ├── test_file_system.py      # File operations
│       ├── test_generate_data.py    # Data generation
│       ├── test_hp_optim.py         # Hyperparameter optimization
│       ├── test_il_train.py         # Imitation learning
│       ├── test_integration.py      # End-to-end integration
│       ├── test_io.py               # I/O utilities
│       ├── test_models.py           # Neural architectures
│       ├── test_modules.py          # Neural components
│       ├── test_mrl_train.py        # Meta-reinforcement learning
│       ├── test_policies.py         # Classical policies
│       ├── test_policies_aux.py     # Policy utilities
│       ├── test_problems.py         # Problem environments
│       ├── test_simulator.py        # Simulation engine
│       ├── test_selection_action.py # Selection strategies (Packing/composed)
│       ├── test_subnets.py          # Encoders and decoders
│       ├── test_suite.py            # Test runner wrapper
│       ├── test_train.py            # Training pipeline
│       ├── test_utils.py            # Utility functions
│       ├── test_vectorized_policies.py
│       └── test_visualize.py        # Visualization utilities
```

### Test File Naming Conventions

| Pattern     | Description              |
| ----------- | ------------------------ |
| `test_*.py` | Standard test modules    |
| `*_test.py` | Alternative test modules |
| `test_*`    | Test function prefix     |
| `Test*`     | Test class prefix        |

---

## 3. Running Tests

### Basic Commands

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov

# Run with detailed coverage report
uv run pytest --cov --cov-report=html
```

### Using the CLI Wrapper

WSmart-Route provides a custom test runner:

```bash
# Run all tests via CLI
python main.py test_suite

# Run specific module
python main.py test_suite --module test_models

# Run specific test class
python main.py test_suite --class TestAttentionModel

# Run specific test function
python main.py test_suite --test test_forward_pass

# Run tests with specific markers
python main.py test_suite --markers "unit and not slow"
```

### Selective Test Execution

```bash
# Run only logic tests
uv run pytest logic/test/

# Run tests matching a pattern
uv run pytest -k "attention"

# Run tests in a specific file
uv run pytest logic/test/test_models.py

# Run a specific test class
uv run pytest logic/test/test_models.py::TestAttentionModel

# Run a specific test method
uv run pytest logic/test/test_models.py::TestAttentionModel::test_forward_pass
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
uv run pytest -n auto

# Run with specific number of workers
uv run pytest -n 4
```

---

## 4. Test Categories and Markers

### Available Markers

Tests are categorized using pytest markers for selective execution:

| Marker                     | Description            | Usage                          |
| -------------------------- | ---------------------- | ------------------------------ |
| `@pytest.mark.slow`        | Long-running tests     | Exclude with `-m "not slow"`   |
| `@pytest.mark.fast`        | Quick tests            | Select with `-m "fast"`        |
| `@pytest.mark.unit`        | Isolated unit tests    | Select with `-m "unit"`        |
| `@pytest.mark.integration` | Cross-module tests     | Select with `-m "integration"` |
| `@pytest.mark.validation`  | Input validation tests | Select with `-m "validation"`  |
| `@pytest.mark.edge_case`   | Edge case coverage     | Select with `-m "edge_case"`   |
| `@pytest.mark.parametrize` | Parametrized tests     | Select with `-m "parametrize"` |

### Command-Specific Markers

| Marker                     | Description                       |
| -------------------------- | --------------------------------- |
| `@pytest.mark.train`       | Training pipeline tests           |
| `@pytest.mark.mrl_train`   | Meta-RL training tests            |
| `@pytest.mark.hp_optim`    | Hyperparameter optimization tests |
| `@pytest.mark.eval`        | Evaluation command tests          |
| `@pytest.mark.test_sim`    | Simulator tests                   |
| `@pytest.mark.gen_data`    | Data generation tests             |
| `@pytest.mark.file_system` | File system operation tests       |
| `@pytest.mark.model`       | Model functionality tests         |
| `@pytest.mark.data`        | Data processing tests             |
| `@pytest.mark.arg_parser`  | Argument parser tests             |

### Special Markers

| Marker                           | Description               |
| -------------------------------- | ------------------------- |
| `@pytest.mark.skip`              | Skip unconditionally      |
| `@pytest.mark.skipif(condition)` | Skip if condition is true |
| `@pytest.mark.xfail`             | Expected to fail          |

### Using Markers

```bash
# Run only fast unit tests
uv run pytest -m "fast and unit"

# Run all tests except slow ones
uv run pytest -m "not slow"

# Run integration tests
uv run pytest -m "integration"

# Run model and training tests
uv run pytest -m "model or train"
```

---

## 5. Fixture System

### Fixture Plugins

The test suite uses a modular fixture system with plugins registered in `conftest.py`:

```python
pytest_plugins = [
    "logic.test.fixtures.arg_fixtures",
    "logic.test.fixtures.data_fixtures",
    "logic.test.fixtures.sim_fixtures",
    "logic.test.fixtures.policy_fixtures",
    "logic.test.fixtures.mrl_fixtures",
    "logic.test.fixtures.model_fixtures",
    "logic.test.fixtures.integration_fixtures",
    "logic.test.fixtures.eval_fixtures",
    "logic.test.fixtures.file_system_fixtures",
    "logic.test.fixtures.input_fixtures",
    "logic.test.fixtures.policy_aux_fixtures",
    "logic.test.fixtures.vectorized_policy_fixtures",
]
```

### Global Fixtures

Available to all tests:

| Fixture                  | Scope    | Description                 |
| ------------------------ | -------- | --------------------------- |
| `mock_torch_device`      | function | CPU torch device            |
| `temp_output_dir`        | function | Temporary output directory  |
| `temp_data_dir`          | function | Temporary data directory    |
| `temp_log_dir`           | function | Temporary log directory     |
| `temp_model_dir`         | function | Temporary model directory   |
| `mock_sys_argv`          | function | Mock sys.argv for CLI tests |
| `mock_environment`       | function | Mock environment variables  |
| `setup_test_environment` | function | Auto-setup test environment |
| `disable_wandb`          | function | Disable W&B logging         |
| `cleanup_test_root`      | session  | Cleanup after session       |

### Using Fixtures

```python
import pytest

class TestMyFeature:
    def test_with_temp_dir(self, temp_output_dir):
        """Test using temporary directory fixture."""
        output_path = Path(temp_output_dir) / "output.txt"
        # Test logic here

    def test_with_device(self, mock_torch_device):
        """Test using CPU device fixture."""
        tensor = torch.zeros(10, device=mock_torch_device)
        assert tensor.device.type == "cpu"
```

### Custom Fixtures

Create custom fixtures in the appropriate fixture module:

```python
# logic/test/fixtures/my_fixtures.py
import pytest

@pytest.fixture
def my_custom_fixture():
    """Create custom test data."""
    data = create_test_data()
    yield data
    cleanup(data)

@pytest.fixture(scope="module")
def expensive_fixture():
    """Fixture shared across module."""
    resource = setup_expensive_resource()
    yield resource
    teardown_expensive_resource(resource)
```

---

## 6. Writing Tests

### Test Structure

Follow this structure for consistent, readable tests:

```python
import pytest
import torch
from logic.src.models.attention_model import AttentionModel


class TestAttentionModel:
    """Test suite for AttentionModel."""

    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return AttentionModel(
            problem="vrpp",
            embed_dim=128,
            n_encode_layers=3
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input batch."""
        batch_size, graph_size = 10, 20
        return {
            "loc": torch.rand(batch_size, graph_size, 2),
            "waste": torch.rand(batch_size, graph_size),
            "depot": torch.rand(batch_size, 2),
        }

    def test_forward_pass(self, model, sample_input):
        """Test model forward pass produces valid output."""
        # Arrange - already done via fixtures

        # Act
        output = model(sample_input)

        # Assert
        assert output is not None
        assert not torch.isnan(output).any()

    def test_output_shape(self, model, sample_input):
        """Test output tensor has correct shape."""
        output = model(sample_input)
        batch_size = sample_input["loc"].shape[0]
        graph_size = sample_input["loc"].shape[1]

        assert output.shape[0] == batch_size

    @pytest.mark.parametrize("batch_size", [1, 16, 64])
    def test_variable_batch_size(self, model, batch_size):
        """Test model handles different batch sizes."""
        input_data = {
            "loc": torch.rand(batch_size, 20, 2),
            "waste": torch.rand(batch_size, 20),
            "depot": torch.rand(batch_size, 2),
        }
        output = model(input_data)
        assert output.shape[0] == batch_size

    @pytest.mark.slow
    def test_training_step(self, model, sample_input):
        """Test complete training step (slow test)."""
        optimizer = torch.optim.Adam(model.parameters())

        output = model(sample_input, return_pi=True)
        loss = output[1].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)
```

### Naming Conventions

| Convention  | Example              | Description                    |
| ----------- | -------------------- | ------------------------------ |
| Test class  | `TestAttentionModel` | PascalCase with `Test` prefix  |
| Test method | `test_forward_pass`  | snake*case with `test*` prefix |
| Fixture     | `sample_input`       | Descriptive snake_case         |
| Helper      | `_create_batch`      | Private underscore prefix      |

### Best Practices

1. **One assertion per test (when practical)**

   ```python
   def test_output_is_tensor(self, model, sample_input):
       output = model(sample_input)
       assert isinstance(output, torch.Tensor)

   def test_output_has_correct_shape(self, model, sample_input):
       output = model(sample_input)
       assert output.shape[0] == sample_input["loc"].shape[0]
   ```

2. **Use parametrize for variations**

   ```python
   @pytest.mark.parametrize("graph_size,expected", [
       (20, True),
       (50, True),
       (100, True),
   ])
   def test_supports_graph_sizes(self, graph_size, expected):
       model = AttentionModel(problem="vrpp")
       input_data = create_input(graph_size)
       result = model(input_data) is not None
       assert result == expected
   ```

3. **Test edge cases explicitly**

   ```python
   @pytest.mark.edge_case
   def test_empty_input_raises(self, model):
       with pytest.raises(ValueError):
           model({})

   @pytest.mark.edge_case
   def test_single_node_graph(self, model):
       input_data = create_input(graph_size=1)
       output = model(input_data)
       assert output is not None
   ```

4. **Use appropriate markers**
   ```python
   @pytest.mark.slow
   @pytest.mark.integration
   def test_full_training_loop(self):
       # Long-running integration test
       pass
   ```

---

## 7. Coverage Requirements

### Running Coverage

```bash
# Basic coverage report
uv run pytest --cov

# HTML report (opens in browser)
uv run pytest --cov --cov-report=html
open htmlcov/index.html

# XML report for CI
uv run pytest --cov --cov-report=xml

# Terminal report with missing lines
uv run pytest --cov --cov-report=term-missing
```

### Coverage Configuration

Coverage settings are defined in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["."]
omit = [
    "*/test/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/site-packages/*",
    "*/.*/*",
    "setup.py",
]
```

### Coverage Targets

| Component             | Minimum | Target |
| --------------------- | ------- | ------ |
| `logic/src/models/`   | 70%     | 85%    |
| `logic/src/policies/` | 60%     | 80%    |
| `logic/src/pipeline/` | 60%     | 75%    |
| `logic/src/envs/`     | 70%     | 85%    |
| `logic/src/utils/`    | 50%     | 70%    |

### Improving Coverage

1. **Identify uncovered code**

   ```bash
   uv run pytest --cov --cov-report=term-missing logic/src/models/
   ```

2. **Focus on critical paths**
   - State transitions in `envs/`
   - Forward passes in `models/`
   - Core algorithms in `policies/`

3. **Add missing test cases**
   ```python
   # Cover error handling
   def test_invalid_input_raises_error(self, model):
       with pytest.raises(ValueError, match="Invalid input"):
           model(None)
   ```

---

## 8. Continuous Integration

### GitHub Actions Workflow

Tests run automatically on:

- Every push to `main`
- Every pull request to `main`

The CI configuration (`.github/workflows/ci.yml`) includes:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.9"
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync
      - name: Run tests
        run: uv run pytest --cov --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

### Pre-commit Hooks

Set up pre-commit hooks for local testing:

```bash
# Install pre-commit
uv pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### CI Best Practices

1. **Run fast tests first**

   ```bash
   uv run pytest -m "fast" && uv run pytest -m "not fast"
   ```

2. **Cache dependencies**

   ```yaml
   - uses: actions/cache@v3
     with:
       path: .venv
       key: venv-${{ hashFiles('pyproject.toml') }}
   ```

3. **Fail fast on critical tests**
   ```bash
   uv run pytest -x -m "critical"
   ```

---

## 9. Test Data Management

### Test Data Location

```
WSmart-Route/
├── data/
│   └── vrpp/
│       ├── vrpp20_val_seed1234.pkl
│       └── vrpp20_test_seed1234.pkl
├── assets/
│   └── test_output/          # Temporary test outputs (auto-cleaned)
└── logic/
    └── test/
        └── fixtures/
            └── test_data/    # Static test data files
```

### Creating Test Data

```python
@pytest.fixture
def sample_vrpp_instance():
    """Create a minimal VRPP instance for testing."""
    return {
        "loc": torch.tensor([
            [0.5, 0.5],  # depot
            [0.2, 0.3],
            [0.8, 0.7],
        ]),
        "waste": torch.tensor([0.0, 0.3, 0.5]),
        "depot": torch.tensor([0]),
    }
```

### Using Test Data Files

```python
import pickle
from pathlib import Path

@pytest.fixture
def test_dataset():
    """Load test dataset from file."""
    data_path = Path(__file__).parent / "fixtures/test_data/small_vrpp.pkl"
    with open(data_path, "rb") as f:
        return pickle.load(f)
```

### Generating Test Data Programmatically

```bash
# Generate small datasets for testing
python main.py generate_data test --problem vrpp --graph_sizes 10 --seed 42 --n_samples 100
```

---

## 10. Debugging Failed Tests

### Verbose Output

```bash
# Maximum verbosity
uv run pytest -vvv

# Show captured output
uv run pytest --capture=no

# Show local variables
uv run pytest --showlocals
```

### Debugging with PDB

```bash
# Drop into debugger on failure
uv run pytest --pdb

# Drop into debugger on first failure
uv run pytest --pdb -x
```

### Using pytest-debug

```python
def test_complex_logic(self):
    import pdb; pdb.set_trace()  # Manual breakpoint
    result = complex_function()
    assert result
```

### Common Failure Patterns

| Symptom               | Likely Cause              | Solution                        |
| --------------------- | ------------------------- | ------------------------------- |
| `ModuleNotFoundError` | Virtual env not activated | `source .venv/bin/activate`     |
| `CUDA error`          | GPU memory exhausted      | Use `mock_torch_device` fixture |
| `FileNotFoundError`   | Missing test data         | Generate or create fixture      |
| `AssertionError`      | Test logic error          | Review assertion conditions     |
| Random failures       | Non-determinism           | Set random seeds                |

### Setting Random Seeds

```python
@pytest.fixture(autouse=True)
def set_random_seeds():
    """Ensure reproducible tests."""
    import random
    import numpy as np
    import torch

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

## 11. Performance Testing

### Benchmarking Tests

```python
import pytest
import time

class TestPerformance:
    @pytest.mark.slow
    def test_forward_pass_performance(self, model):
        """Benchmark forward pass time."""
        input_data = create_large_input(batch_size=256, graph_size=100)

        # Warmup
        for _ in range(5):
            model(input_data)

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            model(input_data)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 100
        assert avg_time < 0.1, f"Forward pass too slow: {avg_time:.3f}s"
```

### Memory Testing

```python
import torch

def test_no_memory_leak(self, model, sample_input):
    """Verify no GPU memory leak."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()

    for _ in range(100):
        output = model(sample_input)
        del output

    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated()

    # Allow small variance
    assert final_memory - initial_memory < 1e6, "Memory leak detected"
```

---

## 12. Mutation Testing

Mutation testing identifies gaps in the test suite by injecting small bugs (mutations) into the source code and checking if existing tests fail.

- **Tool**: `mutmut`
- **Configuration**: `mutmut_config.py` (filters core logic in `logic/src/`)
- **Commands**:
  - `just mutation-test`: Run mutation tests.
  - `just mutation-report`: View detailed results.

---

## 13. Advanced Performance Benchmarking

A formalized suite to track solver latency, throughput, and solution quality, separate from unit-level performance tests.

- **Scripts**: `logic/benchmark/run_all.py` (centralized entry point)
  - Individual benchmarks: `baseline_benchmarks.py`, `neural_benchmarks.py`, `benchmark_policies.py`, etc.
- **Command**: `just benchmark`

---

## 14. Solver Contract Tests

Ensures parity and robustness across different optimization engines.

- **Location**: `logic/test/integration/test_solver_contracts.py`
- **Features**:
  - **Parity checks**: Ensures Gurobi and Hexaly produce similar results formatted identically.
  - **Interface validation**: Validates input/output schemas of solver functions.
  - **Edge cases**: Tests stability with empty instances and invalid IDs.

---

## 15. ELK Stack Logging

Structured logging infrastructure for visualizing test metrics and benchmarking results.

- **Infrastructure**: Docker Compose in `infra/global/docker/elk/`.
- **Structured Logs**: `logic/src/utils/structured_logging.py` provides JSON formatting for Logstash.
- **Usage**:
  ```python
  from logic.src.utils.structured_logging import log_test_metric
  log_test_metric("inference_latency", 12.5)
````

- **Kibana**: Dashboard template available at `infra/global/docker/elk/kibana_dashboard.json`.

To start the ELK stack:

```bash
cd docker/elk
docker-compose up -d
```

The dashboard will be available at `http://localhost:5601`.

---

## Quick Reference

### Common Commands

| Command                        | Description           |
| ------------------------------ | --------------------- |
| `uv run pytest`                | Run all tests         |
| `uv run pytest -v`             | Verbose output        |
| `uv run pytest --cov`          | With coverage         |
| `uv run pytest -m "unit"`      | Run unit tests only   |
| `uv run pytest -m "not slow"`  | Skip slow tests       |
| `uv run pytest -k "attention"` | Match pattern         |
| `uv run pytest -x`             | Stop on first failure |
| `uv run pytest --pdb`          | Debug on failure      |
| `uv run pytest -n auto`        | Parallel execution    |

### Test CLI Commands

| Command                                          | Description     |
| ------------------------------------------------ | --------------- |
| `python main.py test_suite`                      | Run all tests   |
| `python main.py test_suite --module test_models` | Specific module |
| `python main.py test_suite --class TestAM`       | Specific class  |
| `python main.py test_suite --test test_forward`  | Specific test   |
| `python main.py test_suite --markers "fast"`     | By marker       |

---

**Remember**: Good tests are documentation, safety nets, and design tools. Write tests that future developers (including yourself) will thank you for.
