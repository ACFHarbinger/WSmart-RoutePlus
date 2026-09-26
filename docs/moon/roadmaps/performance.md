> Split out of [`docs/moon/ROADMAP.md`](../ROADMAP.md) on 2026-08-27 so each theme can be updated independently. See that file for the Anchor Index and Cross-Cutting Themes table.

---

## F — Performance

### §F.1 — Batched Neural Inference Optimization

**Pain**: The neural decoders (AM, TAM, DDAM, MoE) process instances sequentially during evaluation and simulation. When evaluating against a test set of 10,000 instances, inference time dominates. No `torch.compile()` or `torch.inference_mode()` wrappers are in place.

**Options**

- **A** — Wrap all `model.forward()` evaluation calls in `torch.inference_mode()` (context manager). Zero-code-change speedup of ~10-15% by disabling gradient tracking and version counter overhead. `[Completed 2026-08-27]`
- **B** — Apply `torch.compile(model, mode='reduce-overhead')` to the decoder at evaluation time (PyTorch 2.x); measure speedup on the target GPU.
- **C** — Implement a `BatchedInferenceEngine` that collects N problem instances and runs a single batched forward pass; current code may process one instance at a time during simulation.
- **D** — Export models to ONNX / TensorRT for production inference; 2-5× speedup on NVIDIA GPUs with quantization.

**Recommendation**: **Option A** `[Done]` (adopted across `logic/src/pipeline/features/eval/evaluators/`), **Option B** as the next step (PyTorch 2.2 compile is mature for attention models), **Option C** for the simulation loop specifically.

**Effort × Impact**: Very Low–Medium effort / High impact

---

### §F.2 — GPU Memory Management

**Pain**: The framework targets GPUs with 12-24 GB VRAM, but there is no systematic GPU memory profiling. Memory leaks during long training runs (e.g., growing activation buffers from `PolicyVizMixin` or incorrect `detach()` calls) are currently invisible.

**Options**

- **A** — Add `torch.cuda.memory_summary()` logging and peak memory metric logging (`memory/peak_allocated_mb`, `memory/peak_reserved_mb`) at the end of each epoch via `GPUMemoryMonitor`. `[Completed 2026-08-27]`
- **B** — Use the existing `logic/src/tracking/profiling/memory.py` profiler to generate per-epoch memory traces and write them to `assets/profiling/`.
- **C** — Add `torch.cuda.reset_peak_memory_stats()` at the start of each training epoch to get accurate per-epoch peak measurements. `[Completed 2026-08-27]`
- **D** — Use `torch.utils.checkpoint` (gradient checkpointing) on the encoder layers to trade compute for memory on large instances (100+ nodes).

**Recommendation**: **Options A + C** `[Done]` (implemented via `GPUMemoryMonitor` callback and auto-registered in `WSTrainer`); **Option B** for detailed profiling when a leak is suspected; **Option D** for scaling to larger instances.

**Effort × Impact**: Very Low effort (Options A/C) / High impact

---

### §F.3 — Test Suite Speed

**Pain**: With 218 test files, the full test suite may be slow. No test parallelization is configured (no `pytest-xdist` in the CI pipeline). Fast-marked tests (`@pytest.mark.fast`) are not run in isolation by default.

**Options**

- **A** — Add `pytest-xdist` (`-n auto`) to the CI test job; ensure test files are isolation-clean (no shared mutable global state). `[Quick Win]`
- **B** — Split CI into a `fast` job (runs `@pytest.mark.fast` tests on every push) and a `full` job (all tests, runs on PR merge or nightly).
- **C** — Profile the test suite with `pytest --durations=20` to identify the slowest 20 tests; optimize or mark them `@pytest.mark.slow`.
- **D** — Use `pytest-split` to distribute tests across multiple parallel CI runners for very large suites.

**Recommendation**: **Option C** first (identify the bottlenecks), then **Option B** (fast/full split), then **Option A** (parallelization) once isolation is confirmed.

**Effort × Impact**: Low effort / High impact

---

### §F.4 — Data Loading Optimization

**Pain**: Training data is loaded from `.pkl` files (pickled tensors). For large datasets (graph sizes 100-317), loading dominates the time-to-first-batch. There is no dataset caching, pinned memory, or prefetch worker configuration.

**Options**

- **A** — Switch from `.pkl` to `.pt` (torch.save) for dataset files; PyTorch's tensor serialization is faster and avoids Python's pickle deserialization overhead. `[Quick Win]`
- **B** — Enable `DataLoader(num_workers=4, pin_memory=True, persistent_workers=True)` in the training pipeline; this overlaps CPU data loading with GPU computation.
- **C** — Use `torch.utils.data.IterableDataset` with a streaming generator to avoid loading entire datasets into RAM for very large graph sizes.
- **D** — Pre-compute and cache distance matrices for training instances to avoid recomputation each epoch.

**Recommendation**: **Option B** is the highest-leverage single change (overlapped loading eliminates GPU idle time); **Option A** reduces load latency itself. Both are non-breaking changes.

**Effort × Impact**: Low effort / High impact

---

### §F.5 — Simulation Throughput: Shared Memory & Vectorization

**Pain**: The simulation engine uses `multiprocessing` with a `Manager()` lock and counter for shared metrics synchronization (`_lock`, `_counter`, `_shared_metrics` in `simulator.py`). Manager proxies have significant IPC overhead compared to `multiprocessing.shared_memory`.

**Options**

- **A** — Replace `Manager().dict()` metrics with `multiprocessing.shared_memory.SharedMemory` backed `numpy` arrays; eliminate the Manager proxy round-trip. `[Research]`
- **B** — Use `multiprocessing.Pool.starmap()` with a return-value accumulation pattern instead of shared memory; simpler but requires collecting all results at the end of each day.
- **C** — Vectorize single-day simulation using batched tensor operations on GPU (run N days in parallel as a batch); eliminates multiprocessing entirely for GPU-local simulations.
- **D** — Profile the simulation with `cProfile` to determine whether IPC or computation is the bottleneck before optimizing.

**Recommendation**: **Option D** first (profile before optimizing), then **Option B** as a simpler refactor if IPC is the bottleneck, **Option A** for maximum throughput, **Option C** for GPU-resident simulation.

**Effort × Impact**: Low effort (Option D) / High impact

---

### §F.6 — McCabe Complexity Reduction

**Pain**: CI enforces a McCabe complexity ceiling of 15 (`--max-complexity 15`). Functions in the BPC solver, ALNS destroy/repair operators, and simulation orchestration are likely near or above this threshold, causing CI noise.

**Options**

- **A** — Run `uv run ruff check . --select C90` to identify functions above threshold; refactor the top-10 most complex functions by extracting helper methods. `[Quick Win]`
- **B** — Lower the threshold progressively (15 → 12 → 10) over three sprints to drive continuous simplification.
- **C** — Exempt well-understood algorithmic functions (BPC pricing, ALNS operators) with `# noqa: C901` inline suppressions, with a comment explaining why the complexity is justified.

**Recommendation**: **Option A** for the worst offenders (extract helpers), **Option C** for mathematically-justified complexity that cannot be reduced without obscuring the algorithm.

**Effort × Impact**: Low effort / Medium impact `[Quick Win]`

---

### §F.7 — CUDA-Aware Tensor Operations Audit

**Pain**: The codebase uses `get_device()` for device management, but there may be silent CPU fallbacks when `tensor.to(device)` is called on already-CPU tensors or when tensor operations create new CPU tensors (e.g., via `.numpy()`, `item()`, or list comprehensions inside `forward()`).

**Options**

- **A** — Add a custom `ruff` rule (or `grep` CI check) that flags `.numpy()`, `.item()`, and `list()` calls inside `*.py` files under `logic/src/models/`; these operations force CPU synchronization and break CUDA graphs.
- **B** — Enable `PYTORCH_NO_CUDA_MEMORY_CACHING=1` in the test environment and run the test suite to surface memory-related device mismatches.
- **C** — Use `torch.autograd.set_detect_anomaly(True)` during a profiling run to catch NaN/Inf values that indicate silent device or type mismatches.

**Recommendation**: **Option A** (static analysis check) prevents future regressions; **Option C** is the fastest diagnostic for an existing issue.

**Effort × Impact**: Low effort / High impact

---

### Effort × Impact Matrix — Performance

| Item                                       | Effort    | Impact    | Priority         |
| ------------------------------------------ | --------- | --------- | ---------------- |
| §F.1 Option A (inference_mode wrapper)     | Very Low  | High      | P0 `[Quick Win]` |
| §F.2 Option A+C (GPU memory logging)       | Very Low  | High      | P0 `[Quick Win]` |
| §F.6 Option A (complexity refactor top-10) | Low       | Medium    | P0 `[Quick Win]` |
| §F.3 Option C (profile test durations)     | Low       | High      | P1               |
| §F.4 Option B (DataLoader pinned memory)   | Low       | High      | P1               |
| §F.7 Option A (CPU sync audit)             | Low       | High      | P1               |
| §F.3 Option B (fast/full CI split)         | Low       | High      | P1               |
| §F.1 Option B (torch.compile)              | Medium    | High      | P2               |
| §F.4 Option A (pkl → pt format)            | Low       | Medium    | P2               |
| §F.5 Option D (simulation profiling)       | Low       | High      | P2               |
| §F.5 Option B (Pool.starmap refactor)      | Medium    | High      | P2               |
| §F.4 Option D (cache distance matrices)    | Medium    | High      | P2               |
| §F.1 Option D (TensorRT export)            | High      | Very High | P3               |
| §F.5 Option A (SharedMemory refactor)      | High      | High      | P3 `[Research]`  |
| §F.5 Option C (GPU-vectorized simulation)  | Very High | Very High | P3 `[Research]`  |

---

