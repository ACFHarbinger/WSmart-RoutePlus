# WSmart-Route Architecture

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Gurobi](https://img.shields.io/badge/Gurobi-11.0-ED1C24?logo=gurobi&logoColor=white)](https://www.gurobi.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MyPy](https://img.shields.io/badge/MyPy-checked-2f4f4f.svg)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/pytest-testing-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-50%25-green.svg)](https://coverage.readthedocs.io/)
[![CI](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml/badge.svg)](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml)

> **Version**: 5.0 | **Last Updated**: April 2026 | **Source diagrams**: `docs/moon/packages.mmd`, `docs/moon/classes.mmd`

---

## 1. Module Dependency Overview

Top-level package layout and key inter-module dependencies across the logic layer.

```mermaid
graph TD
    CLI --> CONFIGS
    CLI --> PIPELINE
    CONFIGS --> ENVS
    CONFIGS --> MODELS
    CONFIGS --> POLICIES
    PIPELINE --> RLTRAIN
    PIPELINE --> SIMULATIONS
    PIPELINE --> FEATURES
    RLTRAIN --> ENVS
    RLTRAIN --> MODELS
    RLTRAIN --> POLICIES
    RLTRAIN --> TRACKING
    SIMULATIONS --> POLICIES
    SIMULATIONS --> BINS
    FEATURES --> RLTRAIN
    FEATURES --> DATA
    DATA --> ENVS
    DATA --> POLICIES
    MODELS --> ENCODER
    MODELS --> DECODER
    MODELS --> NORMALIZATION
    ENVS --> ENV
    ENVS --> OBJECTIVE
    POLICIES --> ALNS
    POLICIES --> HGS
    POLICIES --> BPC
    POLICIES --> NEURAL
    UTILS --> CONFIGS
    UTILS --> ENVS
    UTILS --> POLICIES
```

---

## 2. Environment (Problem) Hierarchy

Problem environments and their generators; `IEnv` is the shared contract for all routing problems.

```mermaid
classDiagram
    class IEnv {
        <<interface>>
        +reset(td) TensorDict
        +step(td) TensorDict
        +get_reward(td, actions) Tensor
        +check_solution_validity(td, actions)
        +dataset(phase) Dataset
    }

    class VrppEnv {
        +name = "vrpp"
        +generator: VRPPGenerator
        +get_reward(td, actions) Tensor
    }
    class WcvrpEnv {
        +name = "wcvrp"
        +generator: WCVRPGenerator
    }
    class ScwcvrpEnv {
        +name = "scwcvrp"
        +stochastic: bool
        +generator: SCWCVRPGenerator
    }
    class CvrpEnv {
        +capacity: float
        +generator: CVRPGenerator
    }
    class TspEnv {
        +generator: TSPGenerator
    }
    class AtspEnv {
        +generator: ATSPGenerator
    }
    class PdpEnv {
        +generator: PDPGenerator
    }

    class Generator {
        <<abstract>>
        +n_loc: int
        +generate(batch_size) TensorDict
    }
    class VRPPGenerator {
        +capacity: float
        +waste_distribution: str
        +area: str
        +max_length: float
    }
    class WCVRPGenerator {
        +capacity: float
        +alpha: float
        +beta: float
    }
    class SCWCVRPGenerator {
        +gamma_alpha: float
        +gamma_beta: float
        +n_scenarios: int
    }

    IEnv <|-- VrppEnv
    IEnv <|-- WcvrpEnv
    IEnv <|-- ScwcvrpEnv
    IEnv <|-- CvrpEnv
    IEnv <|-- TspEnv
    IEnv <|-- AtspEnv
    IEnv <|-- PdpEnv

    Generator <|-- VRPPGenerator
    Generator <|-- WCVRPGenerator
    Generator <|-- SCWCVRPGenerator

    VrppEnv *-- VRPPGenerator
    WcvrpEnv *-- WCVRPGenerator
    ScwcvrpEnv *-- SCWCVRPGenerator
```

---

## 3. Neural Model Hierarchy

Encoder/decoder component graph and major model implementations built on PyTorch `nn.Module`.

```mermaid
classDiagram
    class IModel {
        <<interface>>
        +forward(td, env, phase) TensorDict
    }

    class AttentionModel {
        +encoder: TransformerEncoderBase
        +decoder: GlimpseDecoder
        +context_embedder: ContextEmbedder
        +embed_dim: int
        +n_heads: int
        +tanh_clipping: float
        +forward(td) TensorDict
    }
    class PointerNetwork {
        +encoder: PointerEncoder
        +decoder: PointerDecoder
    }
    class MatNet {
        +decoder: MatNetDecoder
        +embed_dim: int
    }
    class MDAM {
        +decoder: MDAMDecoder
        +n_paths: int
    }
    class DeepACO {
        +heatmap_model: nn.Module
        +decoder: ACODecoder
        +n_ants: int
    }
    class DACT {
        +encoder: DACTEncoder
        +decoder: DACTDecoder
    }
    class PolyNet {
        +decoder: PolyNetDecoder
        +order: int
    }
    class CriticNetwork {
        +encoder: TransformerEncoderBase
        +value_head: nn.Linear
        +forward(td) Tensor
    }
    class MandatoryManager {
        +gate_head: GateHead
        +mandatory_head: MandatorySelectionHead
        +forward(td) gate_prob
    }

    class TransformerEncoderBase {
        <<abstract>>
        +embed_dim: int
        +n_heads: int
        +n_layers: int
        +feed_forward_hidden: int
        +forward(td) Tensor
    }
    class GraphConvolutionEncoder {
        +gated: bool
        +n_layers: int
    }
    class GatedGraphAttConvEncoder {
        +gate_activation: str
    }

    class GlimpseDecoder {
        +embed_dim: int
        +n_heads: int
        +tanh_clipping: float
        +forward(td, embeddings, env) TensorDict
    }
    class PointerDecoder {
        +lstm: nn.LSTM
        +glimpse_attn: nn.Module
        +pointer_attn: nn.Module
    }
    class ACODecoder {
        +alpha: float
        +beta: float
        +n_ants: int
        +n_iterations: int
        +rho: float
    }

    IModel <|-- AttentionModel
    IModel <|-- PointerNetwork
    IModel <|-- MatNet
    IModel <|-- MDAM
    IModel <|-- DeepACO
    IModel <|-- DACT
    IModel <|-- PolyNet
    IModel <|-- CriticNetwork
    IModel <|-- MandatoryManager

    TransformerEncoderBase <|-- GraphConvolutionEncoder
    TransformerEncoderBase <|-- GatedGraphAttConvEncoder

    GlimpseDecoder <|-- PointerDecoder

    AttentionModel *-- TransformerEncoderBase : encoder
    AttentionModel *-- GlimpseDecoder : decoder
    PointerNetwork *-- PointerDecoder : decoder
    DeepACO *-- ACODecoder : decoder
    CriticNetwork *-- TransformerEncoderBase : encoder
```

---

## 4. Policy & Solver Hierarchy

`IPolicy` is the shared solving contract; policies range from exact solvers to RL-guided hybrids.

```mermaid
classDiagram
    class IPolicy {
        <<interface>>
        +solve(dist, wastes, capacity) routes
        +configure(cfg)
    }
    class BaseRoutingPolicy {
        +mandatory_selection: List
        +route_improvement: List
    }
    class BaseMultiPeriodPolicy {
        +n_days: int
        +horizon: int
    }
    class BaseJointPolicy {
        +constructor: IRouteConstructor
        +improver: IRouteImprovement
    }

    class BPCPolicy {
        +solver: BPCSolver
    }
    class ALNSPolicy {
        +solver: ALNSSolver
    }
    class ALNSSolver {
        +destroy_ops: List
        +repair_ops: List
        +destroy_weights: ndarray
        +repair_weights: ndarray
        +acceptance: IAcceptanceCriterion
        +build_initial_solution()
        +select_operator() op
        +solve() routes
    }
    class HGSPolicy {
        +solver: HGSSolver
    }
    class HGSSolver {
        +population: List~Individual~
        +crossover_strategy: str
        +local_search: LocalSearch
        +solve() routes
    }
    class ACOPolicy {
        +n_ants: int
        +n_iterations: int
        +rho: float
        +alpha: float
        +beta: float
    }
    class NAPolicy {
        +model: AttentionModel
        +decode_strategy: str
    }
    class DRALNSPolicy {
        +rl_agent: DRALNSPPOAgent
        +alns: ALNSSolver
    }
    class RLALNSPolicy {
        +rl_model: IModel
        +alns: ALNSSolver
    }

    class IAcceptanceCriterion {
        <<interface>>
        +accept(candidate, current, best) bool
    }
    class OnlyImproving
    class BoltzmannMetropolisCriterion {
        +temperature: float
        +cooling_rate: float
    }
    class LateAcceptance {
        +history_size: int
    }
    class RecordToRecordTravel {
        +deviation: float
    }

    IPolicy <|-- BaseRoutingPolicy
    IPolicy <|-- BaseMultiPeriodPolicy
    IPolicy <|-- BaseJointPolicy

    BaseRoutingPolicy <|-- BPCPolicy
    BaseRoutingPolicy <|-- ALNSPolicy
    BaseRoutingPolicy <|-- HGSPolicy
    BaseRoutingPolicy <|-- ACOPolicy
    BaseRoutingPolicy <|-- NAPolicy
    BaseRoutingPolicy <|-- DRALNSPolicy
    BaseRoutingPolicy <|-- RLALNSPolicy

    ALNSPolicy *-- ALNSSolver
    HGSPolicy *-- HGSSolver
    DRALNSPolicy *-- ALNSSolver
    RLALNSPolicy *-- ALNSSolver

    ALNSSolver o-- IAcceptanceCriterion
    IAcceptanceCriterion <|-- OnlyImproving
    IAcceptanceCriterion <|-- BoltzmannMetropolisCriterion
    IAcceptanceCriterion <|-- LateAcceptance
    IAcceptanceCriterion <|-- RecordToRecordTravel
```

---

## 5. RL Training Pipeline

Lightning-based RL module hierarchy with interchangeable baseline strategies.

```mermaid
classDiagram
    class LightningModule {
        <<PyTorch Lightning>>
        +training_step(batch, idx)
        +configure_optimizers()
        +on_train_epoch_end()
    }
    class RL4COLitModule {
        +env: IEnv
        +policy: IModel
        +baseline: Baseline
        +optimizer: Optimizer
        +calculate_loss(td, out) Tensor
        +shared_step(batch, phase)
    }
    class REINFORCE {
        +baseline: Baseline
        +calculate_loss(td, out) Tensor
    }
    class PPO {
        +critic: nn.Module
        +eps_clip: float
        +ppo_epochs: int
        +entropy_weight: float
        +value_loss_coef: float
        +calculate_loss(td, out) Tensor
    }
    class A2C {
        +critic: nn.Module
        +actor_lr: float
        +critic_lr: float
        +entropy_coef: float
        +normalize_advantage: bool
        +calculate_loss(td, out) Tensor
    }
    class POMO {
        +num_augment: int
        +num_starts: int
        +calculate_loss(td, out) Tensor
    }
    class DRALNSLitModule {
        +rl_env: DRALNSEnv
        +agent: DRALNSPPOAgent
        +training_step(batch, idx)
    }

    class Baseline {
        <<abstract>>
        +eval(td, cost) Tensor
        +epoch_callback(model, env, batch)
        +rollout(model, env, batch)
    }
    class RolloutBaseline {
        +baseline_model: IModel
        +eval(td, cost) Tensor
    }
    class CriticBaseline {
        +critic: CriticNetwork
        +eval(td, cost) Tensor
    }
    class ExponentialBaseline {
        +beta: float
        +exp_beta: float
    }
    class POMOBaseline
    class WarmupBaseline {
        +baseline: Baseline
        +n_epochs: int
    }

    LightningModule <|-- RL4COLitModule
    RL4COLitModule <|-- REINFORCE
    RL4COLitModule <|-- PPO
    RL4COLitModule <|-- A2C
    RL4COLitModule <|-- POMO
    RL4COLitModule <|-- DRALNSLitModule

    Baseline <|-- RolloutBaseline
    Baseline <|-- CriticBaseline
    Baseline <|-- ExponentialBaseline
    Baseline <|-- POMOBaseline
    Baseline <|-- WarmupBaseline

    REINFORCE o-- Baseline
    RolloutBaseline *-- IModel : baseline_model
    CriticBaseline *-- CriticNetwork
    WarmupBaseline *-- Baseline
```

---

## 6. Simulator Architecture

State-machine-driven multi-day simulation; `SimulationContext` owns all components and drives state transitions.

```mermaid
classDiagram
    class SimulationContext {
        +cfg: SimConfig
        +policy: IPolicy
        +model_tup: tuple
        +bins: Bins
        +network: Network
        +checkpoint: SimulationCheckpoint
        +tracker: SimulationRunTracker
        +daily_log: Dict~int, DayMetrics~
        +run()
        +transition_to(state: SimulationState)
        +get_current_state_tuple() tuple
    }
    class SimulationState {
        <<abstract>>
        +handle(ctx: SimulationContext)
    }
    class InitializingState {
        +handle(ctx) load data, init bins & network
    }
    class RunningState {
        +handle(ctx) execute day loop
    }
    class FinishingState {
        +handle(ctx) aggregate & save results
    }

    class SimulationDayContext {
        +day: int
        +coords: ndarray
        +fill: ndarray
        +distance_matrix: ndarray
        +tour: List~int~
        +profit: float
        +cost: float
        +overflows: int
        +get(key) Any
    }

    class Bins {
        +levels: ndarray
        +capacities: ndarray
        +rates: ndarray
        +fill_stochastic()
        +collect(routes) float
        +count_overflows() int
    }
    class Network {
        +distance_matrix: ndarray
        +coords: ndarray
        +compute_distances(method: str)
    }
    class SimulationCheckpoint {
        +checkpoint_dir: Path
        +save_state(day, ctx)
        +load_state() ctx
        +find_last_checkpoint_day() int
    }
    class SimulationRunTracker {
        +log_day(day, metrics: DayMetrics)
        +log_final(summary: dict)
        +log_failure(reason: str)
    }
    class DayMetrics {
        +kg: float
        +km: float
        +cost: float
        +profit: float
        +overflows: int
        +tour: List~int~
    }

    SimulationContext *-- SimulationState : current state
    SimulationContext *-- Bins
    SimulationContext *-- Network
    SimulationContext *-- SimulationCheckpoint
    SimulationContext *-- SimulationRunTracker

    SimulationState <|-- InitializingState
    SimulationState <|-- RunningState
    SimulationState <|-- FinishingState

    RunningState *-- SimulationDayContext : per-day ctx
    SimulationDayContext *-- DayMetrics
```

---

## 7. Configuration Hierarchy

Hydra-composed config tree; each solver family has a typed `*Config` → `*Params` pair.

```mermaid
classDiagram
    class Config {
        +model: ModelConfig
        +env: EnvConfig
        +train: TrainConfig
        +eval: EvalConfig
        +data: DataConfig
        +sim: SimConfig
        +tracking: TrackingConfig
    }
    class EnvConfig {
        +name: str
        +num_loc: int
        +graph: GraphConfig
        +reward: ObjectiveConfig
    }
    class ModelConfig {
        +name: str
        +embed_dim: int
        +encoder: EncoderConfig
        +decoder: DecoderConfig
        +normalization: NormConfig
    }
    class TrainConfig {
        +n_epochs: int
        +batch_size: int
        +lr: float
        +rl_algorithm: str
        +baseline: str
        +decoding: DecodingConfig
    }
    class SimConfig {
        +policies: List~str~
        +n_days: int
        +area: str
        +mandatory_selection: List
        +route_improvement: List
    }
    class SolverConfig {
        <<abstract>>
        +seed: Optional~int~
        +time_limit: float
        +mandatory_selection: List
        +route_improvement: List
    }
    class ALNSConfig {
        +max_iterations: int
        +n_removal: int
        +destroy_ops: List~str~
        +repair_ops: List~str~
        +vrpp: bool
    }
    class HGSConfig {
        +population_size: int
        +n_iterations: int
        +crossover: str
    }
    class BPCConfig {
        +mip_gap: float
        +branch_strategy: str
        +farkas_pricing: bool
    }
    class ACOConfig {
        +n_ants: int
        +n_iterations: int
        +alpha: float
        +beta: float
        +rho: float
    }

    Config *-- EnvConfig
    Config *-- ModelConfig
    Config *-- TrainConfig
    Config *-- SimConfig

    SolverConfig <|-- ALNSConfig
    SolverConfig <|-- HGSConfig
    SolverConfig <|-- BPCConfig
    SolverConfig <|-- ACOConfig

    SimConfig o-- SolverConfig : per-policy config
```

---

## 8. Command Execution Flows

### 8.1 Train (`train_lightning`, `hpo`, `meta_train`)

Lightning training lifecycle from CLI invocation through engine to final checkpoint.

```mermaid
sequenceDiagram
    participant CLI as CLI (main.py)
    participant Hydra as Hydra Dispatch
    participant Engine as Train Engine<br/>(engine.py)
    participant Factory as Model Factory
    participant L as PyTorch Lightning
    participant WST as WSTrainer

    CLI->>Hydra: python main.py train_lightning ...
    Hydra->>Engine: run_training(cfg)
    Engine->>Engine: seed_everything(cfg.seed)
    Engine->>Factory: create_model(cfg)
    Factory-->>Engine: model (LitModule)
    Engine->>Engine: instantiate callbacks<br/>(SpeedMonitor, ProgressBar, etc.)
    Engine->>WST: init WSTrainer(max_epochs, strategy, etc.)
    Engine->>L: trainer.fit(model)
    L-->>Engine: Training complete
    Engine->>Engine: save_weights(cfg.train.final_model_path)
    Engine-->>Hydra: Return val/reward metric
```

### 8.2 Evaluate (`eval`)

Deterministic model assessment over dataset instances with optional multi-process scatter.

```mermaid
sequenceDiagram
    participant CLI as CLI (main.py)
    participant Hydra as Hydra Dispatch
    participant Engine as Eval Engine<br/>(engine.py)
    participant Problem as Problem (e.g. VRPP)
    participant Model as Loaded Model
    participant Eval as evaluate_policy

    CLI->>Hydra: python main.py eval <dataset> ...
    Hydra->>Engine: run_evaluate_model(cfg)
    Engine->>Engine: load_model(cfg.eval.policy.load_path)
    Engine->>Problem: make_dataset(dataset_path, **ds_kwargs)
    Problem-->>Engine: dataset
    alt Single Process
        Engine->>Eval: evaluate_policy(model, dataloader)
    else Multi-Process
        Engine->>Engine: mp.Pool.map(eval_dataset_mp)
        Engine->>Eval: evaluate_policy (per process batch)
    end
    Eval-->>Engine: costs, sequences, duration
    Engine->>Engine: Aggregate Metrics (avg cost, km, kg, overflows)
    Engine->>Engine: save_dataset(results, out_file)
    Engine-->>Hydra: return 0.0
```

### 8.3 Simulator (`test_sim`)

State-machine-driven multi-day simulation with sequential or parallel execution paths.

```mermaid
sequenceDiagram
    participant CLI as CLI (main.py)
    participant Hydra as Hydra Dispatch
    participant Engine as Test Engine<br/>(engine.py)
    participant Orch as Orchestrator
    participant Context as SimulationContext
    participant Sim as Simulator States

    CLI->>Hydra: python main.py test_sim ...
    Hydra->>Engine: run_simulator_test(cfg)
    Engine->>Engine: _validate_sim_config(cfg)
    Engine->>Engine: _resolve_data_size(cfg) & config reps
    Engine->>Orch: simulator_testing(...)
    alt Sequential
        Orch->>Context: sequential_simulations(...)
        Context->>Context: run()
        Context->>Sim: Initializing -> Running -> Finishing
    else Parallel
        Orch->>Context: run_parallel_simulation(...)
        Context->>Sim: Map run_single_simulation across Pool
    end
    Sim-->>Orch: DayResults (routes, metrics)
    Orch->>Orch: Process & Aggregate Final Output
    Orch-->>Engine: Complete
```

### 8.4 Data Generation (`gen_data`)

VRP instance builder pipeline producing training tensors or multi-day simulation datasets.

```mermaid
sequenceDiagram
    participant CLI as CLI (main.py)
    participant Hydra as Hydra Dispatch
    participant Generator as generators.py
    participant Builder as VRPInstanceBuilder
    participant Repo as FileSystemRepository

    CLI->>Hydra: python main.py gen_data ...
    Hydra->>Generator: generate_datasets(cfg)
    Generator->>Generator: validate_data_config(cfg)
    Generator->>Repo: set_repository(ROOT_DIR)
    Generator->>Generator: Collect problem configs & dists
    loop For each distribution / instance matrix
        Generator->>Builder: init VRPInstanceBuilder()
        Generator->>Builder: set attributes (size, area, noise...)
        alt dataset_type == test_simulator
            Generator->>Builder: set_num_days(n_days)
            Generator->>Builder: build()
            Generator->>Generator: save_simulation_dataset(.npz)
        else dataset_type == train (or train_time)
            Generator->>Builder: set_num_days(...)
            Generator->>Builder: build_td()
            Generator->>Generator: save_td_dataset(.td)
        end
    end
    Generator-->>Hydra: Complete
```
