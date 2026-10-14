# WSmart+ Route Field Repair Manual: Troubleshooting Guide

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
> **Philosophy**: Diagnosis > Guesswork
> This document maps "Symptoms" to "Root Causes" and "Cures" for WSmart+ Route.

This comprehensive troubleshooting guide is organized by symptom category. Use the table of contents to quickly navigate to your issue.

---

## Table of Contents

1.  [Quick Diagnostics: The Health Check](#1-quick-diagnostics-the-health-check)
2.  [Environment Issues](#2-environment-issues)
3.  [Build Failures](#3-build-failures)
4.  [Runtime Crashes](#4-runtime-crashes)
5.  [Training Issues](#5-training-issues)
6.  [Simulation Issues](#6-simulation-issues)
7.  [Performance Bottlenecks](#7-performance-bottlenecks)
8.  [Data Generation Issues](#8-data-generation-issues)
9.  [Studio (Tauri) Issues](#9-studio-tauri-issues)
10. [Optimizer Integration Issues](#10-optimizer-integration-issues)
11. [Neural Network Issues](#11-neural-network-issues)
12. [Policy Issues](#12-policy-issues)
13. [Checkpoint and Model Loading Issues](#13-checkpoint-and-model-loading-issues)
14. [Common Error Messages Reference](#14-common-error-messages-reference)
15. [Emergency Recovery Procedures](#15-emergency-recovery-procedures)
16. [Asking for Help](#16-asking-for-help)

---

## 1. Quick Diagnostics: The Health Check

Before diving deep, run basic sanity checks.

```bash
# Check Python environment
python --version  # Should be 3.9+
python -c "import torch; print(torch.__version__)"  # Should be 2.2.2
python -c "import torch; print(torch.cuda.is_available())"  # Should be True for GPU

# Run test suite
python main.py test_suite

# Check code quality
uv run ruff check .
```

**Expected Output:**

```text
[✓] Python: 3.9.x or higher
[✓] PyTorch: 2.2.2
[✓] CUDA: Available (RTX 3090ti/4080)
[✓] All tests passed
[✓] Ruff: All checks passed!
```

### Quick Fix Commands

| Issue               | Quick Fix                                             |
| ------------------- | ----------------------------------------------------- |
| UV not found        | `curl -LsSf https://astral.sh/uv/install.sh \| sh`    |
| Python venv missing | `uv sync`                                             |
| Studio not starting | `just studio-install && just studio`                  |
| Tests failing       | `python main.py test_suite --verbose`                 |
| CUDA not detected   | `nvidia-smi` to verify driver, then reinstall PyTorch |

### System Requirements Checklist

| Requirement | Minimum | Recommended |
| ----------- | ------- | ----------- |
| Python      | 3.9     | 3.11        |
| RAM         | 16GB    | 32GB        |
| GPU VRAM    | 8GB     | 16GB+       |
| Disk Space  | 10GB    | 50GB        |
| CUDA        | 11.8    | 12.x        |

---

## 2. Environment Issues

### 2.1 UV / Python

#### Symptom: `ModuleNotFoundError: No module named 'torch'`

- **Cause**: Virtual environment not activated or packages not installed.
- **Fix**:
  ```bash
  source .venv/bin/activate
  uv sync
  ```

#### Symptom: `RuntimeError: PyTorch not compiled with CUDA enabled`

- **Cause**: CPU-only version of PyTorch installed.
- **Fix**:
  ```bash
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
  ```

#### Symptom: `ImportError: libcudnn.so.8: cannot open shared object file`

- **Cause**: CUDA libraries not in PATH.
- **Fix**:
  ```bash
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  # Add to ~/.bashrc for persistence
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
  ```

#### Symptom: `TypeError: 'NoneType' object is not subscriptable` during config loading

- **Cause**: Configuration file missing or malformed.
- **Fix**:
  - Check that `assets/configs/config.yaml` exists.
  - Validate YAML syntax: `python -c "import yaml; yaml.safe_load(open('assets/configs/config.yaml'))"`

#### Symptom: `uv: command not found`

- **Cause**: UV not installed or not in PATH.
- **Fix**:

  ```bash
  # Install UV
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Add to PATH (if not automatic)
  export PATH="$HOME/.cargo/bin:$PATH"
  ```

#### Symptom: `error: externally-managed-environment`

- **Cause**: System Python is protected (common on Ubuntu 24.04+).
- **Fix**: Always use UV or a virtual environment:
  ```bash
  uv sync  # Creates and manages .venv automatically
  ```

### 2.2 CUDA / GPU

#### Symptom: `CUDA out of memory`

- **Cause**: Batch size too large for available GPU memory.
- **Fix**:

  ```bash
  # Reduce batch size in training
  python main.py train_lightning train.batch_size=64

  # Or clear CUDA cache
  python -c "import torch; torch.cuda.empty_cache()"
  ```

#### Symptom: `RuntimeError: CUDA error: device-side assert triggered`

- **Cause**: NaN in computation or invalid tensor operation.
- **Fix**:
  ```bash
  # Enable anomaly detection
  CUDA_LAUNCH_BLOCKING=1 python main.py train_lightning +debug=true
  ```

#### Symptom: `CUDA driver version is insufficient for CUDA runtime version`

- **Cause**: NVIDIA driver too old for installed CUDA toolkit.
- **Fix**:

  ```bash
  # Check current driver
  nvidia-smi

  # Update driver (Ubuntu)
  sudo apt update
  sudo ubuntu-drivers autoinstall
  sudo reboot
  ```

#### Symptom: `RuntimeError: CUDA error: no kernel image is available for execution on the device`

- **Cause**: PyTorch compiled for different GPU architecture.
- **Fix**: Reinstall PyTorch with correct CUDA version for your GPU.

### 2.3 Conda Environment

#### Symptom: `CondaValueError: The target prefix is the base prefix`

- **Cause**: Trying to install in base environment.
- **Fix**:
  ```bash
  conda deactivate
  conda env create --file env/environment.yml -y --name wsr
  conda activate wsr
  ```

#### Symptom: Conda and UV conflict

- **Cause**: Both environments active simultaneously.
- **Fix**:
  ```bash
  conda deactivate
  source .venv/bin/activate
  ```

---

## 3. Build Failures

### 3.1 Dependency Issues

#### Symptom: `error: failed to parse Cargo.toml` (if using Rust components)

- **Cause**: Corrupted or missing dependencies.
- **Fix**:
  ```bash
  # Clean and rebuild
  cargo clean
  cargo build --release
  ```

#### Symptom: `pip install` fails with `PEP 517` error

- **Cause**: Virtual environment corrupted.
- **Fix**:
  ```bash
  deactivate
  rm -rf .venv
  uv sync
  source .venv/bin/activate
  ```

#### Symptom: `Could not build wheels for scipy`

- **Cause**: Missing system dependencies for scientific packages.
- **Fix** (Ubuntu/Debian):
  ```bash
  sudo apt-get install gfortran libopenblas-dev liblapack-dev
  ```

### 3.2 PyInstaller Build Failures

#### Symptom: `FileNotFoundError` during PyInstaller build

- **Cause**: Missing data files or incorrect paths in `build.spec`.
- **Fix**:
  - Verify all paths in `build.spec` are correct.
  - Ensure `--add-data` includes all necessary assets.

#### Symptom: Built executable crashes immediately

- **Cause**: Hidden imports not specified.
- **Fix**: Add missing imports to `build.spec`:
  ```python
  hiddenimports=['torch', ...]
  ```

---

## 4. Runtime Crashes

### 4.1 Python Exceptions

#### Symptom: `KeyError: 'model'` during training

- **Diagnosis**: Missing key in configuration or opts dictionary.
- **Fix**: Check that all required arguments are provided:
  ```bash
  python main.py train_lightning env.num_loc=50 model=am env.name=vrpp
  ```

#### Symptom: `AttributeError: 'str' object has no attribute 'generator'`

- **Cause**: The environment object (`env`) was converted to a string during HParam sanitization before `model.setup()` could use it.
- **Fix**: Ensure `env` is injected into `common_kwargs` _after_ the `deep_sanitize` call in `logic/src/pipeline/features/train.py`.

#### Symptom: `TypeError: calculate_loss() got an unexpected keyword argument 'env'`

- **Cause**: The `calculate_loss` method signature in `AdaptiveImitation` (or other RL modules) does not match the base class `RL4COLitModule`.
- **Fix**: Update the method signature to accept `env: any = None`.

#### Symptom: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

- **Cause**: Tensor dimension mismatch in neural network.
- **Fix**:
  - Check model architecture configuration.
  - Verify input dimensions match expected values.
  - Enable debug mode: `python main.py train_lightning +debug=true`

#### Symptom: `IndexError: index out of range in self`

- **Cause**: Accessing tensor element beyond its size.
- **Fix**:
  - Check graph_size matches data.
  - Verify batch indexing logic.

### 4.2 Segmentation Faults

**Symptom**: Process exits with `Exit Code 139` or `Segmentation fault`.

**Root Cause**: Memory access violation, typically in low-level operations.

**Action**: Check stack trace and recent changes to C extensions.

**Debugging Steps**:

```bash
# Enable core dumps
ulimit -c unlimited

# Run with GDB
gdb python -ex "run main.py train" -ex "bt"
```

### 4.3 Deadlocks and Hangs

**Symptom**: Process freezes without error message.

**Causes**:

1. Deadlock in multiprocessing
2. Infinite loop in algorithm
3. Waiting for unavailable resource

**Fix**:

```bash
# Find process ID
ps aux | grep python

# Check what it's doing
strace -p <PID>

# Force terminate
kill -9 <PID>
```

---

## 5. Training Issues

### 5.1 Model Not Learning (Loss Stays Flat)

**Diagnosis Checklist**:

1.  **Check Learning Rate**: Too high or too low?
    - _Fix_: Try adjusting: `optim.lr=1e-4`
2.  **Check Normalization**: Are inputs normalized?
    - _Fix_: Ensure batch normalization is enabled in model config.
3.  **Check Rewards**: Is the reward signal meaningful?
    - _Fix_: Print rewards during training to verify non-zero values.
4.  **Check Gradient Flow**: Are gradients vanishing?
    - _Fix_: Enable gradient logging: `train.log_step=10`

### 5.2 Training Diverges (NaN Loss)

**Symptom**: Loss suddenly becomes `NaN` or `inf`.

**Causes**:

1.  Learning rate too high.
2.  Division by zero in custom loss.
3.  Numerical instability in attention mechanism.

**Fixes**:

1.  Reduce learning rate by 10x.
2.  Add epsilon to denominators: `x / (y + 1e-8)`.
3.  Use gradient clipping: `rl.max_grad_norm=1.0`
4.  Check for NaN in inputs:
    ```python
    if torch.isnan(input).any():
        raise ValueError("NaN detected in input")
    ```

### 5.3 GPU Utilization Low (<50%)

**Symptom**: `nvidia-smi` shows low GPU usage during training.

**Causes**:

1.  CPU is the bottleneck (data loading).
2.  Batch size too small.
3.  Model too small for GPU.

**Fixes**:

```bash
# Increase batch size
python main.py train_lightning train.batch_size=512

# Use multiple dataloader workers
python main.py train_lightning train.num_workers=4

# Increase model size
python main.py train_lightning model.hidden_dim=256
```

### 5.4 Overfitting

**Symptom**: Training cost decreases but validation cost increases.

**Fixes**:

1.  Add dropout: `model.dropout=0.1`
2.  Use data augmentation.
3.  Reduce model capacity: `model.num_encoder_layers=2`
4.  Increase dataset size: `--n_epochs 20` for data generation

### 5.5 Training Too Slow

**Symptom**: Each epoch takes hours.

**Diagnostic**:

```bash
# Profile training
python -m cProfile -o profile.stats main.py train_lightning train.n_epochs=1
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(20)"
```

**Fixes**:

1. Enable mixed precision: `train.enable_scaler=true`
2. Reduce graph size for debugging
3. Use smaller validation set
4. Disable WandB logging for local runs

### 5.6 Baseline Not Improving

**Symptom**: Rollout baseline stays constant.

**Causes**:

1. Baseline update frequency too low
2. Greedy policy not improving
3. Baseline model diverged

**Fixes**:

```bash
# Increase baseline update frequency
python main.py train_lightning rl.baseline=rollout

# Try exponential baseline instead
python main.py train --baseline exponential
```

---

## 6. Simulation Issues

### 6.1 Unrealistic Results

**Symptom**: Policy achieves 100% collection with zero cost.

**Causes**:

1.  Incorrect reward calculation.
2.  Missing capacity constraints.
3.  Fill rate parameters too low.

**Diagnosis**:

```bash
# Check simulation logs
cat assets/output/<experiment>/log_mean.json

# Verify bin parameters
python -c "from logic.src.pipeline.simulator.loader import load_simulator_data; print(load_simulator_data('riomaior', 'plastic', 20)[0])"
```

### 6.2 Simulation Hangs

**Symptom**: Simulation stuck on specific day or policy.

**Causes**:

1.  Infinite loop in heuristic policy.
2.  Deadlock in parallel execution.

**Fixes**:

```bash
# Run with single core for debugging
python main.py test_sim --cpu_cores 1 --days 1

# Enable verbose logging
python main.py test_sim --verbose
```

### 6.3 Distance Matrix Errors

**Symptom**: `ValueError: Distance matrix not symmetric`

**Cause**: Corrupted distance matrix file.

**Fix**:

```bash
# Regenerate distance matrix
python -c "from logic.src.pipeline.simulator.network import compute_distance_matrix; compute_distance_matrix('riomaior', force_recompute=True)"
```

### 6.4 Policy Returns Empty Route

**Symptom**: Policy execution returns no nodes visited.

**Causes**:

1. All nodes masked (infeasible)
2. Policy threshold too high
3. Vehicle capacity exhausted immediately

**Diagnosis**:

```python
# Check mask state
print(f"Masked nodes: {mask.sum()}/{mask.shape[-1]}")
print(f"Vehicle capacity: {state.remaining_capacity}")
```

### 6.5 Inconsistent Results Across Runs

**Symptom**: Same configuration gives different results.

**Cause**: Random seeds not fixed.

**Fix**:

```bash
python main.py test_sim --seed 42 --np_seed 42 --torch_seed 42
```

---

## 7. Performance Bottlenecks

### 7.1 Slow Training (>1 hour per epoch)

**Action**: Profile the code.

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    # Your training loop here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Common Bottlenecks**:

- **Data Loading**: Use `--num_workers 4`
- **Graph Construction**: Cache edge indices
- **Attention Computation**: Use Flash Attention

### 7.2 Slow Simulation (<100 days/hour)

**Checklist**:

1.  Is parallel execution enabled? Use `--cpu_cores -1` for all cores.
2.  Are policies compiled? Check for JIT compilation opportunities.
3.  Is logging slowing things down? Reduce `--log_step` frequency.

### 7.3 Memory Leaks

**Symptom**: Memory usage grows continuously over time.

**Diagnosis**:

```bash
# Python memory profiling
pip install memory-profiler
python -m memory_profiler main.py train_lightning
```

**Common Causes**:

1.  Not clearing CUDA cache: Add `torch.cuda.empty_cache()` after each epoch.
2.  Storing all episode data: Use bounded replay buffers.
3.  Matplotlib figures not closed: `plt.close('all')`
4.  Tensors not detached: Use `.detach()` for logging values.

### 7.4 Disk I/O Bottleneck

**Symptom**: High disk usage, slow data loading.

**Fixes**:

1. Move data to SSD
2. Use memory-mapped files
3. Pre-load datasets into RAM:
   ```python
   dataset = dataset.cache()  # If using torch dataset
   ```

---

## 8. Data Generation Issues

### 8.1 Generated Data Invalid

**Symptom**: `AssertionError` during loading or training.

**Causes**:

1.  Seed mismatch.
2.  Data distribution parameter error.
3.  File corruption.

**Fixes**:

```bash
# Regenerate with explicit parameters
python main.py generate_data val --problem vrpp --graph_sizes 50 --seed 1234 --data_distribution gamma1

# Verify generated file
python -c "import pickle; data=pickle.load(open('data/vrpp/vrpp50_val_seed1234.pkl', 'rb')); print(len(data))"
```

### 8.2 Insufficient Disk Space

**Symptom**: `OSError: [Errno 28] No space left on device`

**Fix**:

```bash
# Check disk usage
df -h

# Clean old outputs
rm -rf assets/output/*

# Clean cached datasets
rm -rf data/*/virtual_*
```

### 8.3 Data Loading Too Slow

**Symptom**: DataLoader is the bottleneck.

**Fixes**:

```python
# Increase workers
DataLoader(dataset, num_workers=8, pin_memory=True, persistent_workers=True)
```

### 8.4 Pickle Compatibility Issues

**Symptom**: `UnpicklingError` or `ModuleNotFoundError` when loading old data.

**Cause**: Data pickled with different Python/package version.

**Fix**: Regenerate data with current environment.

---

## 9. Studio (Tauri) Issues

### 9.1 Studio Won't Launch

**Symptom**: Blank window or WebKitGTK errors when running `just studio`.

**Fixes**:

```bash
# Install JS dependencies (first checkout)
just studio-install

# Linux WebKitGTK rendering issues
WEBKIT_DISABLE_COMPOSITING_MODE=1 just studio

# Missing system libraries (Debian/Ubuntu)
sudo apt-get install libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev
```

### 9.2 Studio Can't Spawn Python Processes

**Symptom**: Launch buttons fail with "Failed to start process".

**Fixes**:

1. Set the correct **Project Root** and **Python executable** in the Studio Settings page.
2. Verify the CLI works standalone: `uv run python main.py test_sim --help`.

### 9.3 Frontend Type Errors After Changes

```bash
just studio-check   # tsc --noEmit
just studio-clippy  # Rust lint
```

---

## 10. Optimizer Integration Issues

### 10.1 Gurobi License Error

**Symptom**: `GurobiError: No Gurobi license found`

**Fix**:

```bash
# Verify license file
ls $GUROBI_HOME/gurobi.lic

# Check environment variable
echo $GRB_LICENSE_FILE

# Activate license
grbgetkey <your-license-key>
```

### 10.2 Gurobi Takes Too Long

**Symptom**: Optimization doesn't finish within reasonable time.

**Causes**:

1.  Problem size too large.
2.  No time limit set.
3.  MIP gap tolerance too tight.

**Fixes**:

```python
# Set time limit
opts['gurobi_time_limit'] = 300  # 5 minutes

# Set MIP gap
opts['gurobi_mip_gap'] = 0.01  # 1%

# Enable logging
opts['gurobi_verbose'] = True
```

### 10.3 Hexaly Installation Issues

**Symptom**: `ModuleNotFoundError: No module named 'localsolver'`

**Fix**:

```bash
# Install Hexaly Python API
pip install hexaly

# Verify license
hexaly info
```

### 10.4 OR-Tools Integration Issues

**Symptom**: `ImportError: cannot import name 'pywrapcp'`

**Fix**:

```bash
uv pip install ortools --force-reinstall
python -c "from ortools.constraint_solver import pywrapcp; print('OK')"
```

---

## 11. Neural Network Issues

### 11.1 Attention Weights All Zero

**Symptom**: Attention visualization shows uniform or zero weights.

**Causes**:

1. Masking applied incorrectly
2. Softmax overflow/underflow
3. Temperature scaling too extreme

**Fixes**:

```python
# Check mask values
print(f"Mask True count: {mask.sum()}")

# Add temperature scaling
scores = scores / temperature
```

### 11.2 Encoder Output All Same

**Symptom**: All node embeddings identical after encoding.

**Causes**:

1. Mean pooling collapsing information
2. Over-smoothing in deep GNN

**Fixes**:

1. Reduce encoder layers: `model.num_encoder_layers=2`
2. Add skip connections
3. Use different aggregation

### 11.3 Decoder Stuck in Loop

**Symptom**: Model repeatedly selects same node or depot.

**Causes**:

1. Greedy decoding getting stuck
2. Mask not updated properly
3. Context embedding not changing

**Fix**: Check state update logic:

```python
# Verify visited mask updates
assert state.visited[selected_node] == True
```

### 11.4 Gradient Explosion

**Symptom**: Gradients become extremely large, training unstable.

**Fixes**:

```bash
# Enable gradient clipping
python main.py train_lightning rl.max_grad_norm=1.0

# Use gradient scaling for mixed precision
python main.py train_lightning train.enable_scaler=true
```

---

## 12. Policy Issues

### 12.1 ALNS Not Improving

**Symptom**: ALNS solution quality stagnates early.

### 12.2 HGS crashes with `TypeError: 'int' object is not iterable`

- **Cause**: Vectorized HGS solver returning a single integer or flat list for a route, instead of a list of routes, which the parsing loop doesn't expect.
- **Fix**: Update `logic/src/models/policies/classical/hgs.py` to check `isinstance(routes[0], int)` and wrap single routes in a list.

**Causes**:

1. Temperature cooling too fast
2. Destroy fraction too small
3. Not enough iterations

**Fixes**:

```python
# Adjust ALNS parameters
alns_config = {
    'cooling_rate': 0.9999,  # Slower cooling
    'destroy_fraction': 0.4,  # More aggressive
    'iterations': 20000,      # More iterations
}
```

### 12.2 BPC Memory Exhaustion

**Symptom**: BPC runs out of memory on large instances.

**Fixes**:

1. Set node limit: `--bpc_node_limit 10000`
2. Set column limit: `--bpc_column_limit 50000`
3. Use heuristic instead for large instances

### 12.3 HGS Population Collapse

**Symptom**: All solutions in population become identical.

**Causes**:

1. Diversity penalty too low
2. Selection pressure too high

**Fixes**:

```python
# Increase diversity
hgs_config['min_population_diversity'] = 0.1
```

### 12.4 Neural Policy Worse Than Random

**Symptom**: Trained model performs worse than random baseline.

**Causes**:

1. Model not actually trained (checkpoint issue)
2. Wrong problem type specified
3. Data distribution mismatch

**Diagnosis**:

```bash
# Verify model loaded
python -c "import torch; m = torch.load('model.pt'); print(m.keys())"

# Check model performance in greedy mode
python main.py eval --decode_strategy greedy
```

---

## 13. Checkpoint and Model Loading Issues

### 13.1 Checkpoint Not Found

**Symptom**: `FileNotFoundError` when resuming training.

**Fix**:

```bash
# List available checkpoints
ls -la model_weights/vrpp_20/

# Use correct path
python main.py train --load_path model_weights/vrpp_20/am/epoch-99.pt
```

### 13.2 Model Architecture Mismatch

**Symptom**: `RuntimeError: Error(s) in loading state_dict`

**Cause**: Model architecture changed since checkpoint was saved.

**Fix**: Load with `strict=False` or retrain:

```python
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

### 13.3 Optimizer State Incompatible

**Symptom**: Training crashes after loading checkpoint.

**Cause**: Different optimizer or parameters.

**Fix**: Reset optimizer when resuming:

```bash
python main.py train --load_path model.pt --reset_optimizer
```

### 13.4 CUDA/CPU Device Mismatch

**Symptom**: `RuntimeError: Attempting to deserialize object on CUDA device`

**Fix**:

```python
# Load to CPU first
checkpoint = torch.load('model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
```

---

## 14. Common Error Messages Reference

| Error Message                                                 | Likely Cause                                | Quick Fix                                         |
| ------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------- |
| `ModuleNotFoundError: No module named 'logic'`                | Virtual environment issue                   | `source .venv/bin/activate; uv sync`              |
| `CUDA out of memory`                                          | Batch size too large                        | Reduce `--batch_size`                             |
| `AssertionError: Invalid graph size`                          | Config mismatch                             | Check `--graph_size` matches data                 |
| `FileNotFoundError: model not found`                          | Wrong model path                            | Verify path with `ls model_weights/`       |
| `ValueError: Cannot perform reduction on dimension 0`         | Empty batch                                 | Check dataloader and batch size                   |
| `RuntimeError: Expected all tensors to be on the same device` | Device mismatch                             | Ensure all tensors on same device (CPU/CUDA)      |
| `OSError: [Errno 28] No space left on device`                 | Disk full                                   | Free up disk space, clean temp files              |
| `TimeoutError`                                                | Network or computation timeout              | Increase timeout, check connectivity              |
| `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`     | CUDA/cuDNN version mismatch                 | Reinstall PyTorch for your CUDA version           |
| `AttributeError: module 'torch' has no attribute 'compiler'`  | PyTorch version too old                     | Upgrade PyTorch: `uv pip install torch --upgrade` |
| `GurobiError: Model too large for size-limited license`       | Academic license limits                     | Use smaller instances or get full license         |
| `RecursionError: maximum recursion depth exceeded`            | Infinite recursion in policy                | Check termination conditions                      |
| `PicklingError: Can't pickle local object`                    | Lambda or local function in multiprocessing | Use module-level functions                        |
| `BrokenPipeError`                                             | Child process died                          | Check worker error logs                           |

---

## 15. Emergency Recovery Procedures

### 15.1 Corrupted Virtual Environment

```bash
# Nuclear option: rebuild everything
deactivate 2>/dev/null
rm -rf .venv
uv sync
source .venv/bin/activate
```

### 15.2 Stuck Training Process

```bash
# Find and kill training processes
pkill -f "python main.py train"

# Clear GPU memory
nvidia-smi --gpu-reset

# Or from Python
python -c "import torch; torch.cuda.empty_cache()"
```

### 15.3 Corrupted Checkpoint

```python
# Attempt partial recovery
import torch

try:
    checkpoint = torch.load('corrupt_checkpoint.pt')
except:
    # Try loading with pickle directly
    import pickle
    with open('corrupt_checkpoint.pt', 'rb') as f:
        checkpoint = pickle.load(f)

# Save recovered model
torch.save(checkpoint['model_state_dict'], 'recovered_model.pt')
```

### 15.4 Database/Log Corruption

```bash
# Remove corrupted logs and restart
rm -rf assets/output/corrupted_experiment/
rm -rf wandb/corrupted_run/

# Restart with fresh logging
python main.py train --run_name fresh_start
```

### 15.5 Git Recovery

```bash
# Discard local changes and reset
git stash
git checkout main
git pull origin main

# Or hard reset (destructive)
git reset --hard origin/main
```

---

## 16. Asking for Help

When opening an issue, provide the **"Crash Tuple"**:

1.  **The Command**: Exactly what you typed.
2.  **The Stack Trace**: The full output.
3.  **The Context**: Commit hash (`git rev-parse HEAD`) and OS.
4.  **The Reproduction Steps**: Minimal steps to reproduce the issue.
5.  **Expected vs Actual Behavior**: What you expected vs what happened.

### Issue Template

```markdown
## Environment

- OS: Ubuntu 24.04
- Python: 3.9.x
- PyTorch: 2.2.2
- CUDA: 11.8
- Commit: abc1234

## Steps to Reproduce

1. Run `uv sync`
2. Run `python main.py train --model am --graph_size 50`
3. Observe error after 5 epochs

## Expected Behavior

Training should complete all 10 epochs successfully.

## Actual Behavior

Training crashes with the following error:
`error message here`

## Full Stack Trace

`paste here`

## Additional Context

- This worked on commit xyz789
- Only happens with graph_size 50, works fine with 20
```

### Diagnostic Information Script

Run this script to gather system information for bug reports:

```bash
#!/bin/bash
echo "=== WSmart-Route Diagnostic Report ==="
echo "Date: $(date)"
echo ""
echo "=== System Info ==="
uname -a
echo ""
echo "=== Python ==="
python --version
which python
echo ""
echo "=== PyTorch ==="
python -c "import torch; print(f'Version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "No NVIDIA GPU detected"
echo ""
echo "=== Disk Space ==="
df -h .
echo ""
echo "=== Git Status ==="
git rev-parse HEAD
git status --short
```

---

**Remember**: The best debugging is prevention. Write tests, validate inputs, and use type hints consistently. When in doubt, enable verbose logging and check the logs first.
