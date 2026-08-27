# Model Implementation Analysis Report

**Project**: WSmart+ Route
**Date**: August 27, 2026
**Purpose**: Comparison of model papers (`bibliography/models/`) vs implementations (`logic/src/models/`)
**Status**: In progress (Grok, issue #63). Scores filled as each model is checked; TBD rows are placeholders, not verdicts.

---

## Paper ↔ implementation map

Confirmed against the filesystem, not assumed from names. POMO has no `logic/src/models/core/pomo/` directory.

| Paper (`bibliography/models/`) | Implementation | Kind |
|---|---|---|
| `Attention_Model.pdf` | `logic/src/models/core/attention_model/` + `GlimpseDecoder` | constructive policy |
| `Pointer_Networks.pdf` | `logic/src/models/core/pointer_network/` | constructive policy |
| `POMO.pdf` | `logic/src/pipeline/rl/core/pomo.py` (algorithm) and `pomo_size` on AM | RL trainer, not a separate architecture |
| `Sym-NCO.pdf` | `logic/src/models/core/attention_model/symnco_policy.py` | AM subclass + projection head |
| `DACT.pdf` | `logic/src/models/core/dact/` | TBD |
| `DR-ALNS.pdf` | `logic/src/models/core/dr_alns/` | TBD |
| `DeepACO.pdf` | `logic/src/models/core/deepaco/` | TBD |
| `GFACS.pdf` | `logic/src/models/core/gfacs/` | TBD |
| `GLOP.pdf` | `logic/src/models/core/glop/` | TBD |
| `MATNet.pdf` | `logic/src/models/core/matnet/` | TBD |
| `MDAM.pdf` | `logic/src/models/core/mdam/` | TBD |
| `N2S.pdf` | `logic/src/models/core/n2s/` | TBD |
| `NARGNN.pdf` | `logic/src/models/core/nargnn/` | TBD |
| `NeuOpt.pdf` | `logic/src/models/core/neuopt/` | TBD |
| `PolyNet.pdf` | `logic/src/models/core/polynet/` | TBD |

Present in `core/` with **no** matching file in `bibliography/models/`: `hybrid_attention_model/`, `moe/`, `temporal_attention_model/`. Logged, not scored, until a paper is added or they are identified as in-house.

Shared building blocks most `core/` models compose: `logic/src/models/subnets/` (encoders/decoders/embeddings) and `logic/src/models/common/` (autoregressive / non-autoregressive / transductive / improvement). `logic/src/models/common/critic_network/` contains a documented deprecated `LegacyCriticNetwork` (#58); not re-flagged here.

---

## Executive Summary

Partial — 3 of 15 bibliography papers checked so far.

- The flagship constructor is a real Attention Model (Kool et al. 2019), not a name-only wrapper: GAT encoder, glimpse decoder, tanh-clip 10, mask-before-softmax.
- Several "models" in the bibliography are **training methods** on top of AM (POMO, Sym-NCO) rather than separate `core/` architectures. Scoring them as if they were competing encoders would be the wrong comparison.
- Default-value drift is the main failure mode: `AttentionModel.__init__` says `n_encode_layers=2` and `AttentionModelPolicy` says `hidden_dim=128` / `normalization="batch"`, while `logic/configs/models/am.yaml` (and the paper) want 3 layers, FF hidden 512, and the YAML further switches norm to instance and activation to GELU. Anyone constructing the class without the YAML gets a different model than the documented default.

---

## Faithfulness Score Summary

| Model | Score | Rationale |
| :---- | :---- | :-------- |
| **AM** | 4/5 | Encoder/decoder/clip/mask match Kool et al. 2019. YAML defaults diverge (instance norm, GELU, extras); class defaults diverge from both YAML and the paper (`n_encode_layers=2`). |
| **Pointer** | 4/5 | LSTM encoder + pointer decoder with masking is Vinyals 2015. `tanh_clipping=10` is the AM-era clip, not in the original Ptr-Net paper. YAML `hidden_dim=128` vs policy default 512. |
| **POMO** | 4/5 | Implemented as a REINFORCE trainer with dihedral-8 augment and multi-start shared baseline (Kwon 2020), plus `pomo_size` on AM. No standalone POMO network — that is how the paper is meant to be used. `mandatory_starts_only` is a domain extension. |
| **Sym-NCO** | TBD | Located; not yet scored. |
| **DACT** | TBD | |
| **DR-ALNS** | TBD | |
| **DeepACO** | TBD | |
| **GFACS** | TBD | |
| **GLOP** | TBD | |
| **MATNet** | TBD | |
| **MDAM** | TBD | |
| **N2S** | TBD | |
| **NARGNN** | TBD | |
| **NeuOpt** | TBD | |
| **PolyNet** | TBD | |

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

## Stopped here

Next: Sym-NCO (Kim et al. 2022) in `symnco_policy.py`, then DACT. Coordinate with #61 (Codex, general bug pass on `logic/src/models/`) on the bus if a faithfulness finding is also a code defect — the AM `n_encode_layers=2` class default is the first candidate.
