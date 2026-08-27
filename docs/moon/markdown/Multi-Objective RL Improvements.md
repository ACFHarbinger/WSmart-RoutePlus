# Multi-Objective RL Improvements for WSmart-Route

Based on the analysis of _GDPO: Group reward-Decoupled Normalization Policy Optimization_, this report outlines specific architectural improvements for the WSmart-Route codebase to enhance multi-objective capabilities, particularly for the Periodic Capacitated Vehicle Routing Problem (PCVRP).

## 1. Preference-Conditioned MOCO (PMOCO)

**Concept**: Instead of training a single policy for a fixed set of weights $\mathbf{w}$, train a **Hypernetwork** or **Conditioned Decoder** that takes $\mathbf{w}$ as input and generates policy parameters $\theta_\mathbf{w}$ (or modulates attention weights). This allows a single trained model to generate the entire Pareto frontier at inference time.

**Implementation Plan**:

- **New Module**: `logic/src/models/pmoco.py`
  - Implement a lightweight MLP hypernetwork.
  - Input: Preference vector $\mathbf{w}$.
  - Output: Weights/Biases for the final Attention layers or a modulation vector for the query/key projections.
- **Integration**:
  - Extend `RLConfig` to support `pmoco: bool = True`.
  - Modify `AttentionModel` to accept a `preference_embedding` argument.
- **Synergy with GDPO**: Use GDPO for the training loss stability, but weight the normalized advantages by the input preference $\mathbf{w}$ (which is varied during training).

## 2. Projected Conflicting Gradients (PCGrad)

**Concept**: When gradients for different objectives conflict (negative cosine similarity), project one gradient onto the normal plane of the other to prevent destructive interference.

**Implementation Plan**:

- **New Optimizer Wrapper**: `logic/src/optim/pcgrad.py`
  - Wraps the standard PyTorch optimizer.
  - Requires computing gradients for each objective independently (computationally expensive) or for groups of objectives.
- **Use Case**: Best for scenarios where objectives are strictly competitive, e.g., minimizing _Distance_ vs maximizing _Driver Consistency_ (where consistent routes are often geometrically inefficient).

## 3. Advanced Conditional Rewards

**Concept**: Gate the optimization of secondary objectives (Efficiency, Consistency) on the satisfaction of primary constraints (Feasibility).

**Current Status**: Basic support implemented via `gdpo_conditional_key`.

**Future Enhancement**:

- **Feasibility-Aware Reward Function**:
  $$R_{total} = R_{feasibility} + \mathbb{I}(feasible) \cdot (\sum w_k \hat{A}_k)$$
- **Logic**: If a solution is infeasible, the gradient should ONLY drive it towards feasibility. The secondary objectives should not dilute this signal.
- **Implementation**:
  - Modify `GDPO.calculate_loss` to completely mask out the advantage of secondary objectives if `conditional_key` is 0, rather than just zeroing the advantage (which might still affect the mean/std calc).

## 4. Hierarchical GDPO for PCVRP

**Concept**: Apply GDPO specifically to the **Upper-Level Scheduler** in the `HRLModule`.

**Rationale**: The scheduler makes pattern assignment decisions that affect multiple heterogeneous downstream metrics:

- **Cost**: Continuous, high variance (aggregated from daily routes).
- **Consistency**: Discrete (count of diver switches).
- **Balance**: Statistical (variance of fleet usage).

**Implementation Plan**:

- **Scheduler**: Use `mandatoryManager` or a simpler tabular/bandit manager for pattern assignment.
- **Reward Signal**: The "env" for the scheduler is the simulator running the Lower-Level Router.
- **GDPO Application**: The scheduler's epoch involves running $G$ simulations. Normalize Cost, Consistency, and Balance _independently_ across these $G$ simulations before updating the scheduler policy.

## 5. Foundation Model Routing (Multi-Task Learning)

**Concept**: Treat different VRP variants (CVRP, VRPP, WCVRP) as different "objectives" or tasks in a Multi-Task Learning (MTL) framework.

**Implementation Plan**:

- **Attribute Embeddings**: Add a `ConstraintEncoder` to the model that encodes the problem type (time windows, capacity, periodicity).
- **Zero-Shot Generalization**: Train on a mix of VRPP and WCVRP instances. Use GDPO to normalize the rewards from these different tasks so that the model doesn't overfit to the task with larger reward magnitudes.
