# Model Implementation Analysis Report

**Project**: WSmart+ Route
**Date**: August 27, 2026
**Purpose**: Comparison of model papers (`bibliography/models/`) vs implementations (`logic/src/models/`)
**Status**: Complete first pass (Grok, issue #63). Scores are faithfulness to the named paper, not quality of the code.

---

## Paper ↔ implementation map

Confirmed against the filesystem, not assumed from names. POMO has no `logic/src/models/core/pomo/` directory.

| Paper (`bibliography/models/`) | Implementation | Kind |
|---|---|---|
| `Attention_Model.pdf` | `logic/src/models/core/attention_model/` + `GlimpseDecoder` | constructive policy |
| `Pointer_Networks.pdf` | `logic/src/models/core/pointer_network/` | constructive policy |
| `POMO.pdf` | `logic/src/pipeline/rl/core/pomo.py` (algorithm) and `pomo_size` on AM | RL trainer, not a separate architecture |
| `Sym-NCO.pdf` | `logic/src/models/core/attention_model/symnco_policy.py` | AM subclass + projection head |
| `DACT.pdf` | `logic/src/models/core/dact/` | improvement (2-opt pairs) |
| `DR-ALNS.pdf` | `logic/src/models/core/dr_alns/` | PPO-controlled ALNS |
| `DeepACO.pdf` | `logic/src/models/core/deepaco/` | NAR heatmap + ACO |
| `GFACS.pdf` | `logic/src/models/core/gfacs/` | DeepACO + GFlowNet TB |
| `GLOP.pdf` | `logic/src/models/core/glop/` | hierarchical partition + local |
| `MATNet.pdf` (**wrong PDF**: PV forecasting) | `logic/src/models/core/matnet/` (Kwon mixed-score) | matrix row/col constructive |
| `MDAM.pdf` | `logic/src/models/core/mdam/` | multi-decoder constructive |
| `N2S.pdf` | `logic/src/models/core/n2s/` | improvement (k-NN attention) |
| `NARGNN.pdf` (Li/Chen/Koltun GCN+tree search) | `logic/src/models/core/nargnn/` (TSP heatmap, no tree search) | mismatch |
| `NeuOpt.pdf` | `logic/src/models/core/neuopt/` | improvement (pairwise) |
| `PolyNet.pdf` | `logic/src/models/core/polynet/` | K-conditioned AM |

Present in `core/` with **no** matching file in `bibliography/models/`: `hybrid_attention_model/`, `moe/`, `temporal_attention_model/`. Logged, not scored, until a paper is added or they are identified as in-house.

Shared building blocks most `core/` models compose: `logic/src/models/subnets/` (encoders/decoders/embeddings) and `logic/src/models/common/` (autoregressive / non-autoregressive / transductive / improvement). `logic/src/models/common/critic_network/` contains a documented deprecated `LegacyCriticNetwork` (#58); not re-flagged here.

---

## Executive Summary

15 of 15 bibliography papers scored. Pattern:

- Constructive AM-family (AM, Pointer, MDAM, PolyNet) and ACO-family (DeepACO, GFACS, GLOP, DR-ALNS) are real implementations of the named papers, typically 4/5, with default-value or solver-choice drift rather than missing algorithms.
- Two bibliography PDFs do not match the code they sit next to: `MATNet.pdf` is a PV-forecasting paper; `NARGNN.pdf` is Li/Chen/Koltun GCN+tree-search while `core/nargnn/` is a TSP heatmap constructor.
- POMO and Sym-NCO are **training methods** on AM, not separate `core/` networks. Sym-NCO's problem-symmetricity loss is now wired (it was a commented no-op: `loss_ps` stayed 0).
- The three improvement models (DACT, NeuOpt, N2S) share one pairwise `(i,j)` decoder template. DACT still has CPE; the dual-aspect collaborative attention that gives the paper its name is collapsed to a single stream. NeuOpt's encoder ignores the current tour. N2S keeps k-NN attention but is wired to `tsp_kopt`, not pickup-and-delivery.

---

## Faithfulness Score Summary

| Model | Score | Rationale |
| :---- | :---- | :-------- |
| **AM** | 4/5 | Encoder/decoder/clip/mask match Kool et al. 2019. Class defaults for depth and FF width now match the paper (3 / 512). YAML still uses instance norm and GELU. |
| **Pointer** | 4/5 | LSTM encoder + pointer decoder with masking is Vinyals 2015. `tanh_clipping=10` is the AM-era clip, not in the original Ptr-Net paper. YAML `hidden_dim=128` vs policy default 512. |
| **POMO** | 4/5 | Implemented as a REINFORCE trainer with dihedral-8 augment and multi-start shared baseline (Kwon 2020), plus `pomo_size` on AM. No standalone POMO network — that is how the paper is meant to be used. `mandatory_starts_only` is a domain extension. |
| **Sym-NCO** | 4/5 | Projection head + all three paper losses. `problem_symmetricity_loss` was a commented no-op; now called. Inherits AM YAML norm/GELU drift. |
| **DACT** | 3/5 | CPE and pairwise 2-opt decoder exist. Dual-aspect collaborative attention is a single stream (coords + positional add, then self-attention). |
| **NeuOpt** | 3/5 | Pairwise decoder matches the improvement template; encoder never reads the current tour; no explicit k-opt action parameterisation. Relies on `tsp_kopt` env. |
| **N2S** | 2/5 | k-NN masked attention is the paper's efficiency trick. Wired to `tsp_kopt` with a generic pair decoder, not PDP ruin/recreate of pickup-delivery pairs. |
| **DeepACO** | 4/5 | GNN heatmap + ACO ants with α/β/ρ and optional local search. Default `n_iterations=1` is a thin ACS loop. |
| **GFACS** | 4/5 | DeepACO plus learnable `logZ` and Trajectory Balance in `gfacs/model.py`. |
| **DR-ALNS** | 4/5 | PPO, four heads, two×64 MLP, and Table 1's seven state features in order. Operator zoo is a VRPP subset. |
| **MDAM** | 4/5 | Shared encoder, 5 decoder paths, pairwise KL to discourage collapse. |
| **MATNet** | 4/5 | Row/column mixed-score encoder (Kwon et al. 2021), 5 layers, instance norm. |
| **GLOP** | 4/5 | NAR partition then local subproblem. Default local solver is `"greedy"`, not LKH. |
| **NARGNN** | 2/5 | `NARGNN.pdf` is Li, Chen & Koltun (GCN + guided tree search for vertex-subset NP-hard problems). The code is a TSP edge-heatmap NAR constructor (Joshi-style), with no tree search. |
| **PolyNet** | 4/5 | AM encoder conditioned on K strategy vectors via `PolyNetDecoder`. |

---

## 1. Attention Model (Kool et al. 2019)

**Paper**: Kool, van Hoof & Welling, "Attention, Learn to Solve Routing Problems!", ICLR 2019 (`Attention_Model.pdf`)
**Implementation**: `logic/src/models/core/attention_model/{model,policy,decoding}.py`, decoder `logic/src/models/subnets/decoders/glimpse/decoder.py`, config `logic/configs/models/am.yaml`
**Faithfulness**: ★★★★☆ (4/5)

### Key components from the paper

- Encoder: N Transformer layers (MHA + FF), typically **3** layers, embed 128, 8 heads, FF hidden 512
- Batch normalization in the encoder
- Decoder: autoregressive; context = graph embedding + first node + last node; multi-head **glimpse**, then single-head compatibility with **tanh clip C=10**
- Mask visited / infeasible nodes **before** softmax
- Train with REINFORCE + greedy-rollout baseline (this lives in the RL pipeline, not the model class)

### What matches

- `GlimpseDecoder` is the paper decoder: `project_node_embeddings` / `project_fixed_context` / `project_step_context`, `tanh_clipping=10.0` (`TANH_CLIPPING` in `logic/src/constants/models.py` is commented as the Kool 2019 standard)
- `mask_inner` and `mask_logits` default True — matches the required invalid-move masking in `AGENTS.md` §6.1
- `AttentionModelPolicy` default `embed_dim=128`, `n_heads=8`, `n_encode_layers=3`
- `am.yaml` encoder: gat, embed 128, hidden 512, 3 layers, 8 heads

### Differences

1. **Normalization (documented adaptation, not a silent bug)**
   - Paper: batch norm
   - `AttentionModelPolicy` default: `"batch"`
   - `am.yaml`: `norm_type: "instance"` with the comment "instance works best for VRP"
   - Assessment: deliberate. The YAML, not the class default, is what `python main.py train` actually composes unless overridden.

2. **Activation**
   - Paper FFN: ReLU
   - `am.yaml`: GELU
   - Assessment: modern Transformer default; undocumented relative to the paper. Not a correctness bug.

3. **Class-default drift**
   - `AttentionModel.__init__` default `n_encode_layers` is now 3 (was 2; paper and `am.yaml` already said 3)
   - `AttentionModelPolicy` default `hidden_dim` is now 512 (was 128; paper and `am.yaml` already said 512). Tests that want a thinner net already pass the count.

4. **Extensions, labelled as such**
   - `pomo_size`, `spatial_bias`, `connection_type` (residual/dense/hyper), `temporal_horizon`, problem-specific context embedders (`VRPPContextEmbedder`, `WCVRPContextEmbedder`)
   - These are framework features, not paper claims

**Overall**: architecture is the paper's AM. Class defaults for depth and FF width now match Kool 2019. Remaining 4/5 is YAML instance-norm / GELU vs the paper's batch-norm / ReLU.

---

## 2. Pointer Networks (Vinyals et al. 2015)

**Paper**: Vinyals, Fortunato & Jaitly, "Pointer Networks", NeurIPS 2015 (`Pointer_Networks.pdf`)
**Implementation**: `logic/src/models/core/pointer_network/{model,policy}.py`, `PointerEncoder` / `PointerDecoder` in `logic/src/models/subnets/`
**Faithfulness**: ★★★★☆ (4/5)

### What matches

- LSTM encoder over the input sequence, decoder that **points** at input positions via attention (not a softmax over a fixed output vocab)
- Variable-length inputs; used here as the sequential baseline the AM paper itself compared against
- `n_glimpses=1`, mask on glimpses and on logits

### Differences

1. **Tanh clip C=10** is the AM paper's trick, applied here by default (`tanh_clipping: float = 10.0`). Original Ptr-Net uses unclipped pointing attention.
2. `ptr.yaml` sets `hidden_dim: 128` and `n_layers: 2`; `PointerNetworkPolicy` default `hidden_dim=512`. Same class-vs-YAML trap as AM, inverted.
3. `input_dim = 2` hardcoded (coordinates). Fine for Euclidean TSP/VRP; not the original's generic sequence setting.

**Overall**: it is a Pointer Network. The AM-era clip and the Hydra/class hidden-size split keep it off 5/5.

---

## 3. POMO (Kwon et al. 2020)

**Paper**: Kwon et al., "POMO: Policy Optimization with Multiple Optima for Neural Combinatorial Optimization", NeurIPS 2020 (`POMO.pdf`)
**Implementation**: `logic/src/pipeline/rl/core/pomo.py` (subclass of `REINFORCE`); `pomo_size` on `AttentionModel` / `GlimpseDecoder`; forced start nodes in `AttentionModelPolicy.forward`
**Faithfulness**: ★★★★☆ (4/5)

### Why this is not under `models/core/`

POMO is a **training algorithm** for an existing constructive policy (almost always AM): N starting nodes × shared baseline × optional dihedral augmentation. Putting it next to DACT/NeuOpt in a "model" table is a bibliography-layout fact, not an architecture fact.

### What matches

- `num_augment=8`, `augment_fn="dihedral8"`, `first_aug_identity=True` — the paper's x8 instance augmentation
- Multi-start decoding; shared baseline (`baseline="none"` on the parent REINFORCE, handled inside `calculate_loss`)
- `num_starts=None` → number of nodes, which is the paper default for TSP

### Differences

- `mandatory_starts_only`: domain extension for VRPP mandatory bins. Documented in the constructor. Does not change the paper algorithm when the TensorDict has no `mandatory` field.
- No separate POMO network weights — reuse AM. That is correct.

**Overall**: 4/5 as a training method. Would be 2/5 if scored as a missing `core/pomo/` model; that scoring would be wrong.

---

## 4. Sym-NCO (Kim et al. 2022)

**Paper**: Kim, Park & Park, "Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization", NeurIPS 2022 (`Sym-NCO.pdf`)
**Implementation**: `logic/src/models/core/attention_model/symnco_policy.py` (projection head); `logic/src/pipeline/rl/core/symnco.py` (losses); `logic/configs/models/symnco.yaml`
**Faithfulness**: ★★★★☆ (4/5)

### What matches

- Policy is AM plus a 3-layer ReLU MLP projection head (`embed_dim → embed_dim`), which is the RL4CO / paper recipe for the invariance term
- Trainer subclasses POMO: dihedral augmentation, multi-start, then three named losses
- `alpha=0.2` (invariance) and `beta=1.0` (solution symmetricity) match the paper's typical weights
- `invariance_loss` and `solution_symmetricity_loss` in `logic/src/utils/tasks/losses.py` are the real functions

### Differences

1. **Problem-symmetricity loss was dead; now wired.** `shared_step` commented the term and never imported `problem_symmetricity_loss`. It now calls it on dim 1 (augmentation axis), matching `solution_symmetricity_loss` on dim -1 (starts). Regression: `test_shared_step_includes_problem_symmetricity_loss`.
2. **`train/loss_inv` logged the leftover zero tensor.** It now logs the computed invariance term.
3. Same AM YAML drift as §1 (instance norm / GELU in `symnco.yaml`).

**Overall**: all three paper losses are live. Remaining 4/5 is inherited AM default/YAML drift, not a missing algorithm.

---

## 5. DACT (Ma et al. 2021)

**Paper**: Ma et al., "Learning to Iteratively Solve Routing Problems with Dual-Aspect Collaborative Transformer", NeurIPS 2021 (`DACT.pdf`)
**Implementation**: `logic/src/models/core/dact/{encoder,decoder,policy,model}.py`; CPE in `logic/src/models/subnets/embeddings/positional/cyclic_positional_embedding.py`
**Faithfulness**: ★★★☆☆ (3/5)

### What matches

- Improvement policy, not constructive: decoder emits a pair `(i, j)` for a 2-opt-style move
- Cyclic positional encoding exists (`pos_type="CPE"` default) and is the paper's tour-order encoding
- 3 layers, 8 heads, embed 128, FF width 4×, ReLU — paper-like
- Layer-norm (paper used LN on the improvement transformer)

### Differences

1. **Dual-aspect collaboration is a single stream.** The paper keeps a *node* aspect and a *position* aspect as two embeddings that attend to each other (DAC-Att). The encoder projects coordinates, adds CPE via `pos_embedding(h, pos_normalized)`, then runs ordinary self-attention. There is no second stream and no cross-aspect attention.
2. Current-tour order is injected (`td["solution"]` sorted to a position index). That is more than NeuOpt does, less than the paper.

**Overall**: it is an improvement Transformer with CPE, not Dual-Aspect Collaborative Transformer as named.

---

## 6. NeuOpt (Ma et al. 2023)

**Paper**: Ma et al., "NeuOpt: Neural k-Opt Optimization for Combinatorial Routing" (`NeuOpt.pdf`)
**Implementation**: `logic/src/models/core/neuopt/{encoder,decoder,policy,model}.py`
**Faithfulness**: ★★★☆☆ (3/5)

### What matches

- Iterative improvement with a Transformer encoder and a pairwise move decoder
- `env_name="tsp_kopt"` — k-opt semantics are delegated to the environment
- Defaults 3 / 8 / 128

### Differences

1. **Encoder never reads the current tour.** `NeuOptEncoder.forward` concatenates depot+locs and self-attends. DACT at least injects solution order. NeuOpt's paper conditions on the incumbent k-opt state.
2. **Decoder is the same pairwise Q·K template as DACT/N2S**, not an explicit k-node k-opt parameterisation (k, then k indices).
3. The three improvement decoders are close enough to be one class with a name swap.

**Overall**: an improvement policy sitting on `tsp_kopt`. The "k-opt" in the paper title is not visible in the network.

---

## 7. N2S (Li et al.)

**Paper**: Ma et al., "Efficient Neural Neighborhood Search for Pickup and Delivery Problems" (`N2S.pdf`)
**Implementation**: `logic/src/models/core/n2s/{encoder,decoder,policy,model}.py`
**Faithfulness**: ★★☆☆☆ (2/5)

### What matches

- k-nearest-neighbour attention mask (`k_neighbors=20`) is the paper's sparse-neighborhood encoder trick
- Pairwise decoder for a local move

### Differences

1. **Wrong problem.** Policy hardcodes `env_name="tsp_kopt"`. N2S is a pickup-and-delivery method (paired pickup/delivery, ruin a pair, reinsert).
2. **Wrong operator.** Generic `(i, j)` attention, not the paper's remove-and-reinsert of a PD pair.
3. Encoder is a *single* MHA+FF block (not a stack), coords only, no PD pairing features.

**Overall**: the neighborhood mask is the only paper-specific piece. This is the weakest of the 15.

---

## 8. DeepACO (Ye et al. 2023)

**Paper**: Ye et al., "DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization", ICLR 2023 (`DeepACO.pdf`)
**Implementation**: `logic/src/models/core/deepaco/`; encoder `subnets/encoders/deepaco/`; decoder `subnets/decoders/deepaco/`
**Faithfulness**: ★★★★☆ (4/5)

### What matches

- Two-stage: GNN predicts an edge heatmap, ACO constructs from pheromone × heuristic
- `alpha`, `beta`, `rho`, `n_ants=20`, optional local search
- Non-autoregressive policy base

### Differences

- Default `n_iterations=1` is a single ACO pass; the paper runs a multi-iteration ACS
- Heatmap GNN is a framework encoder, not a line-by-line copy of the paper's specific GNN

**Overall**: it is DeepACO. The thin default iteration count is the only material drift.

---

## 9. GFACS

**Paper**: GFlowNet Ant Colony System (`GFACS.pdf`)
**Implementation**: `logic/src/models/core/gfacs/{policy,model}.py`, `subnets/encoders/gfacs/`
**Faithfulness**: ★★★★☆ (4/5)

### What matches

- Subclasses DeepACO; adds a learnable scalar `logZ`
- Trajectory Balance in `GFACS.calculate_loss`: `(log_likelihood + logZ − (log_pb + β·advantage))²`
- Optional local-search TB term when `train_with_local_search` is on
- `return_all=True` so every ant contributes to the TB residual

### Differences

- Same ACO-iteration default as DeepACO
- Backward policy is uniform (`calculate_log_pb_uniform`), which is a standard GFlowNet simplification the paper may or may not use depending on the section — treated as a documented engineering choice, not a silent omission

---

## 10. DR-ALNS (Reijnen et al. AAAI 2024)

**Paper**: Deep Reinforcement Learning for Adaptive Large Neighborhood Search (`DR-ALNS.pdf`)
**Implementation**: `logic/src/models/core/dr_alns/{ppo_agent,dr_alns_solver,ppo_trainer}.py`
**Faithfulness**: ★★★★☆ (4/5)

### What matches

- PPO actor-critic, **four** discrete heads: destroy, repair, severity, temperature — that is the paper's control interface
- Shared MLP (`state_dim=7`, `hidden_dim=64`) then linear heads
- Solver loop is ALNS: destroy → repair → accept, with the agent choosing the knobs
- Real operators: `random_removal`, `worst_removal`, `cluster_removal`; `greedy_insertion`, `regret_2_insertion`
- VRPP extras (`mandatory_nodes`, waste/capacity/R/C) are domain wrapping, labelled as such

### Differences

- Operator catalogue is a subset of a full ALNS zoo (three destroy, two repair). The paper's OPSWTW instantiation uses problem-specific destroy/repair; the code uses VRPP operators.

**State vector vs Table 1 (line-checked).** Paper Table 1 is seven problem-agnostic features. `DRALNSState.to_tensor` emits exactly those, in order: Best improved, Current accepted, Current improved, Is current best, Cost difference best, Stagnation count, Search budget. MLP is two hidden layers of size 64, matching the paper's PPO network. Four action heads match A1–A4 (destroy, repair, severity 1–10, acceptance temperature).

---

## 11. MDAM (Xin et al. 2021)

**Paper**: Xin et al., Multi-Decoder Attention Model (`MDAM.pdf`)
**Implementation**: `logic/src/models/core/mdam/`; `subnets/encoders/mdam/`; `subnets/decoders/mdam/`
**Faithfulness**: ★★★★☆ (4/5)

### What matches

- Shared GAT encoder, `num_paths=5` independent decoders
- Pairwise KL on the first-step distributions to keep paths from collapsing (`_compute_initial_kl_divergence`)
- 3 layers, 8 heads, embed 128, batch-norm default (closer to AM-the-paper than `am.yaml`)

### Differences

- KL is computed on the *initial* step only, not every decode step. That is a cheaper proxy for the paper's diversity term.
- Best-of-paths aggregation is the training wrapper's job; not re-checked here.

---

## 12. MATNet (Kwon et al. 2021)

**Intended paper**: Kwon et al., Matrix Encoding Networks for Combinatorial Optimization (NeurIPS 2021)
**File in `bibliography/models/MATNet.pdf`**: **wrong PDF** — Tortora et al., "MATNet: Multi-Level Fusion Transformer-Based Model for Day-Ahead PV Generation Forecasting" (IEEE Trans. Smart Grid). Namesake collision; that paper is not implemented here.
**Implementation**: `logic/src/models/core/matnet/`; `MixedScoreMHA` in `subnets/modules/matnet_attention.py`
**Faithfulness**: ★★★★☆ (4/5) to Kwon et al. (the architecture the code comments cite). Unscored against the PDF that is actually in the folder.

### Mixed-score equation (line-checked against the Kwon architecture)

`MixedScoreMHA.forward`:

\[
\mathrm{compat} = \frac{1}{\sqrt{d}} \big( Q_{\mathrm{row}} K_{\mathrm{col}}^\top + (Q_{\mathrm{col}} K_{\mathrm{row}}^\top)^\top + W_{\mathrm{mat}} \odot M \big)
\]

then row-softmax over columns and column-softmax over rows, values from the opposite stream. That *is* mixed-score attention (dot-product both ways plus a learned scale on the raw cost matrix). Dual FF + instance-norm residuals on each stream match the encoder-layer template. Defaults: 5 layers, FF 512, instance norm.

### Differences

- Written for matrix problems (ATSP/FFSP). Euclidean VRPP is a domain stretch.
- The bibliography file does not contain this paper. Anyone reading `MATNet.pdf` from `bibliography/models/` will audit the wrong work.

---

## 13. GLOP (Ye et al.)

**Paper**: Global and Local Optimization for routing (`GLOP.pdf`)
**Implementation**: `logic/src/models/core/glop/`; `subnets/modules/glop_factory.py`
**Faithfulness**: ★★★★☆ (4/5)

### What matches

- Two-stage: NAR global partition, then a local subproblem solver per cluster
- `n_samples=10` partition variants; env-specific adapter via `get_adapter(env_name)`

### Differences

- Default `subprob_solver="greedy"`. The paper's published numbers use a strong local solver (LKH / equivalent). Greedy is a speed default, not the paper's experimental setup.
- `embed_dim=64` is smaller than the AM-family 128.

---

## 14. NARGNN

**PDF in folder**: Li, Chen & Koltun, "Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search" (arXiv:1810.10659)
**Implementation**: `logic/src/models/core/nargnn/` — NAR TSP *edge heatmap* + greedy/sampling decoder
**Faithfulness**: ★★☆☆☆ (2/5) to the PDF that is actually in `bibliography/models/`

### What the PDF describes

- Vertex-wise GCN that scores whether a *vertex* is in the optimal set
- Diverse solution heads + **guided tree search** to explore the combinatorial space
- Evaluated on SAT / MVC / MAXCUT-style problems, including graphs with 10⁵ nodes

### What the code does

- Predicts an `[N, N]` *edge* heatmap (Joshi-style NAR TSP constructor)
- No tree search, no vertex-inclusion head, no SAT/MVC tasks
- 15 graph layers + 5 heatmap layers, SiLU, mean aggregation

The filename `NARGNN` matches a different literature thread (non-autoregressive GNN heatmaps). The PDF does not. This is a bibliography mapping error, not a quietly drifted Joshi clone.

---

## 15. PolyNet (Hottung et al.)

**Paper**: PolyNet — multiple solution strategies in one weight set (`PolyNet.pdf`)
**Implementation**: `logic/src/models/core/polynet/`; `subnets/decoders/polynet/`
**Faithfulness**: ★★★★☆ (4/5)

### What matches

- Shared GAT encoder; decoder is conditioned on a strategy index `k ∈ {1,…,K}` via learned binary/strategy vectors
- `num_encoder_layers=6`, embed 128, FF 512, instance norm, tanh clip 10
- Population-style train/val/test all default to `"sampling"` (you want diversity)

### Differences

- `encoder_type` string accepts `"AM"` / `"POMO"` as labels; that is a factory knob, not a second architecture
- K must be passed in; there is no YAML default in `logic/configs/models/` for polynet (no `polynet.yaml` next to `am.yaml`)

---

## Cross-cutting findings

1. **Class vs YAML defaults** (AM §1, Pointer §2) is the constructive-family failure mode. Factory paths that compose Hydra YAMLs are closer to the papers than raw `Cls(...)` construction.
2. **Improvement-family copy-paste.** DACT, NeuOpt, and N2S decoders are the same pairwise Q·K block. Differentiating paper claims (DAC-Att, k-opt, PD ruin/recreate) did not survive the shared `ImprovementPolicy` template.
3. **Sym-NCO `loss_ps`** was a concrete defect (helper existed, call did not). Wired in the follow-up commit; the invariance log now tracks the computed term.
4. **Bibliography PDF mismatches.** `MATNet.pdf` is a PV-forecasting transformer, not Kwon's matrix encoder. `NARGNN.pdf` is Li/Chen/Koltun GCN+tree-search, not the heatmap constructor in `core/nargnn/`. DR-ALNS Table 1 matches the 7-d state 1:1. Mixed-score attention in code matches Kwon's formula even though the PDF in the folder does not.

## In-house `core/` models (no PDF in `bibliography/models/`)

Not scored 1–5. Recorded so they are not mistaken for missing bibliography entries.

| Dir | What it is |
|---|---|
| `temporal_attention_model/` | AM + GRU/LSTM fill-level predictor fused before encoding. In-house WCVRP extension (`tam.yaml`). |
| `moe/` | AM with sparse MoE layers (`num_experts=4`, top-2, noisy gating). Class default encoder depth still 2. |
| `hybrid_attention_model/` | Two-stage: pick a vectorized classical constructor (HGS/ALNS/ACO) then apply vector local-search operators. Neural+OR hybrid, not a named paper in this folder. |
