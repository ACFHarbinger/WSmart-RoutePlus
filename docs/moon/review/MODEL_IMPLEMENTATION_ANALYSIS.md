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
| `MATNet.pdf` | `logic/src/models/core/matnet/` | matrix row/col constructive |
| `MDAM.pdf` | `logic/src/models/core/mdam/` | multi-decoder constructive |
| `N2S.pdf` | `logic/src/models/core/n2s/` | improvement (k-NN attention) |
| `NARGNN.pdf` | `logic/src/models/core/nargnn/` | NAR GNN heatmap |
| `NeuOpt.pdf` | `logic/src/models/core/neuopt/` | improvement (pairwise) |
| `PolyNet.pdf` | `logic/src/models/core/polynet/` | K-conditioned AM |

Present in `core/` with **no** matching file in `bibliography/models/`: `hybrid_attention_model/`, `moe/`, `temporal_attention_model/`. Logged, not scored, until a paper is added or they are identified as in-house.

Shared building blocks most `core/` models compose: `logic/src/models/subnets/` (encoders/decoders/embeddings) and `logic/src/models/common/` (autoregressive / non-autoregressive / transductive / improvement). `logic/src/models/common/critic_network/` contains a documented deprecated `LegacyCriticNetwork` (#58); not re-flagged here.

---

## Executive Summary

15 of 15 bibliography papers scored. Pattern:

- Constructive AM-family (AM, Pointer, MDAM, PolyNet, MATNet) and NAR/ACO-family (NARGNN, DeepACO, GFACS, GLOP, DR-ALNS) are real implementations of the named papers, typically 4/5, with default-value or solver-choice drift rather than missing algorithms.
- POMO and Sym-NCO are **training methods** on AM, not separate `core/` networks. Sym-NCO's problem-symmetricity loss is dead code: `shared_step` comments the term, never imports `problem_symmetricity_loss`, and adds a tensor that stays 0.
- The three improvement models (DACT, NeuOpt, N2S) share one pairwise `(i,j)` decoder template. DACT still has CPE; the dual-aspect collaborative attention that gives the paper its name is collapsed to a single stream. NeuOpt's encoder ignores the current tour. N2S keeps k-NN attention but is wired to `tsp_kopt`, not pickup-and-delivery.

---

## Faithfulness Score Summary

| Model | Score | Rationale |
| :---- | :---- | :-------- |
| **AM** | 4/5 | Encoder/decoder/clip/mask match Kool et al. 2019. YAML defaults diverge (instance norm, GELU, extras); class defaults diverge from both YAML and the paper (`n_encode_layers=2`). |
| **Pointer** | 4/5 | LSTM encoder + pointer decoder with masking is Vinyals 2015. `tanh_clipping=10` is the AM-era clip, not in the original Ptr-Net paper. YAML `hidden_dim=128` vs policy default 512. |
| **POMO** | 4/5 | Implemented as a REINFORCE trainer with dihedral-8 augment and multi-start shared baseline (Kwon 2020), plus `pomo_size` on AM. No standalone POMO network — that is how the paper is meant to be used. `mandatory_starts_only` is a domain extension. |
| **Sym-NCO** | 3/5 | Projection head + invariance + solution-symmetricity are present. Problem-symmetricity loss is commented in `shared_step` and never called (`loss_ps` stays 0). |
| **DACT** | 3/5 | CPE and pairwise 2-opt decoder exist. Dual-aspect collaborative attention is a single stream (coords + positional add, then self-attention). |
| **NeuOpt** | 3/5 | Pairwise decoder matches the improvement template; encoder never reads the current tour; no explicit k-opt action parameterisation. Relies on `tsp_kopt` env. |
| **N2S** | 2/5 | k-NN masked attention is the paper's efficiency trick. Wired to `tsp_kopt` with a generic pair decoder, not PDP ruin/recreate of pickup-delivery pairs. |
| **DeepACO** | 4/5 | GNN heatmap + ACO ants with α/β/ρ and optional local search. Default `n_iterations=1` is a thin ACS loop. |
| **GFACS** | 4/5 | DeepACO plus learnable `logZ` and Trajectory Balance in `gfacs/model.py`. |
| **DR-ALNS** | 4/5 | PPO agent with four heads (destroy/repair/severity/temperature) driving real ALNS operators. Compact 7-d search state. |
| **MDAM** | 4/5 | Shared encoder, 5 decoder paths, pairwise KL to discourage collapse. |
| **MATNet** | 4/5 | Row/column mixed-score encoder (Kwon et al. 2021), 5 layers, instance norm. |
| **GLOP** | 4/5 | NAR partition then local subproblem. Default local solver is `"greedy"`, not LKH. |
| **NARGNN** | 4/5 | NAR edge heatmap from a deep GNN (15 graph layers + 5 heatmap layers). |
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

3. **Class-default drift (the real mismatch)**
   - `AttentionModel.__init__` default `n_encode_layers=2` (`model.py`), paper and YAML say 3
   - `AttentionModelPolicy` default `hidden_dim=128`, paper and YAML say 512
   - Anyone who instantiates `AttentionModel(...)` / `AttentionModelPolicy(...)` without the Hydra YAML gets a thinner net than Kool 2019. Factory paths that go through `am.yaml` are fine.

4. **Extensions, labelled as such**
   - `pomo_size`, `spatial_bias`, `connection_type` (residual/dense/hyper), `temporal_horizon`, problem-specific context embedders (`VRPPContextEmbedder`, `WCVRPContextEmbedder`)
   - These are framework features, not paper claims

**Overall**: architecture is the paper's AM. Score is 4/5 because of the class-default / YAML / paper three-way split on depth, FF width, and norm. Not a 3 — the composed training default is still an AM.

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
**Faithfulness**: ★★★☆☆ (3/5)

### What matches

- Policy is AM plus a 3-layer ReLU MLP projection head (`embed_dim → embed_dim`), which is the RL4CO / paper recipe for the invariance term
- Trainer subclasses POMO: dihedral augmentation, multi-start, then three named losses
- `alpha=0.2` (invariance) and `beta=1.0` (solution symmetricity) match the paper's typical weights
- `invariance_loss` and `solution_symmetricity_loss` in `logic/src/utils/tasks/losses.py` are the real functions

### Differences

1. **Problem-symmetricity loss is dead.** `SymNCO.shared_step` comments "1. Problem symmetricity loss" and then never computes it. `problem_symmetricity_loss` is implemented and unit-tested, but `symnco.py` does not import it. `loss_ps` is initialised to `0.0` and added into the total. The paper's problem-symmetricity term (consistency across geometric augmentations of the *instance*) is therefore always zero.
2. **Logging bug attached to the same block.** `self.log("train/loss_inv", loss_inv)` logs the leftover zero tensor, not `loss_inv_val`.
3. Same AM default-drift inheritance as §1 (instance norm / GELU in `symnco.yaml`).

**Overall**: the projection head and two of three paper losses are live. Dropping the named third loss is why this is 3/5, not 4. Flagged for #61 rather than silently patched here.

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

**Paper**: Li, Yan & Wu, Neural Neighborhood Search for pickup-and-delivery (`N2S.pdf`)
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

- Operator catalogue is a subset of a full ALNS zoo (three destroy, two repair)
- Search-state feature vector is 7-d; not re-derived against the paper's exact feature list in this pass

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

**Paper**: Kwon et al., "Matrix Encoding Networks for Neural Combinatorial Optimization" (`MATNet.pdf`)
**Implementation**: `logic/src/models/core/matnet/`; `subnets/encoders/matnet/` (row/col layers); `subnets/decoders/matnet/`
**Faithfulness**: ★★★★☆ (4/5)

### What matches

- Dual embeddings: row and column, mixed-score attention over the cost/distance matrix
- `MatNetEncoder` stacks `MatNetEncoderLayer` on `(row_emb, col_emb, matrix)`
- Default `num_layers=5`, instance norm, tanh clip 10 — ATSP-scale MATNet
- Init embedding is matrix-statistic based (`MatNetInitEmbedding`)

### Differences

- Written for matrix problems (ATSP/FFSP). Using it on Euclidean VRPP would be a domain stretch; the architecture itself is the paper's.

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

**Paper**: Non-autoregressive GNN heatmap constructor (`NARGNN.pdf`; Joshi et al. 2019 lineage)
**Implementation**: `logic/src/models/core/nargnn/`; `subnets/encoders/nargnn/`
**Faithfulness**: ★★★★☆ (4/5)

### What matches

- Predicts an `[N, N]` edge heatmap, then a NAR decoder (greedy / sampling / beam) extracts a tour
- Deep GNN: 15 graph-encoder layers + 5 heatmap-generator layers, SiLU, mean aggregation — in the Joshi-style "deep GCN heatmap" family

### Differences

- Defaults (`embed_dim=64`, `env_name="tsp"`) are TSP-sized, not VRPP-sized
- Exact layer count / residual recipe vs the PDF was not line-checked; the *kind* of model matches

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
3. **Sym-NCO `loss_ps`** is a concrete defect, not a scoring quibble: the helper exists, the call does not. Logged on the bus for #61.
4. **Unscored `core/` dirs** (`hybrid_attention_model/`, `moe/`, `temporal_attention_model/`) still have no `bibliography/models/` PDF.

No further bibliography/models papers remain. #63 first pass is complete; a later pass can line-check MATNet mixed-score equations and the DR-ALNS 7-d state against the PDFs.
