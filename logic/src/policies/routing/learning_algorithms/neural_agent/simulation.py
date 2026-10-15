"""
Simulator day logic for Neural Agent.

Attributes:
    SimulationMixin: Mixin for single-day simulation logic in NeuralAgent.

Example:
    >>> from logic.src.policies.routing.learning_algorithms import NeuralAgent
    >>> agent = NeuralAgent(model)
    >>> routes, metrics = agent.run_day(env)
    >>> print(f"Best routes: {routes}")
    >>> print(f"Metrics: {metrics}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

if TYPE_CHECKING:
    from logic.src.configs.policies.na import NeuralAgentConfig

from logic.src.tracking.hooks.attention_hooks import add_attention_hooks
from logic.src.utils.decoding.beam_search import _beam_search
from logic.src.utils.decoding.decoding_utils import backtrack
from logic.src.utils.routing.tours import (
    get_route_cost,
)


class SimulationMixin:
    """
    Mixin for single-day simulation logic in NeuralAgent.

    Attributes:
        None
    """

    def compute_simulator_day(  # noqa: C901
        self,
        input: Dict[str, Any],
        graph: Tuple[Any, Any],
        distC: torch.Tensor,
        profit_vars: Optional[Any] = None,
        waste_history: Optional[torch.Tensor] = None,
        cost_weights: Optional[Dict[str, float]] = None,
        mandatory: Optional[torch.Tensor] = None,
        params: Optional[NeuralAgentConfig] = None,
        **kwargs: Any,
    ):
        """Execute neural routing policy for a single simulation day.

        Args:
            input: Input for the neural policy.
            graph: Graph for the neural policy.
            distC: Distance cost for the neural policy.
            profit_vars: Profit variables for the neural policy.
            waste_history: Waste history for the neural policy.
            cost_weights: Cost weights for the neural policy.
            mandatory: Mandatory nodes for the neural policy.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple of (routes, cost, profit).
        """
        edges, dist_matrix = graph
        hook_data = add_attention_hooks(self.model.encoder)  # type: ignore[attr-defined]

        # Check mandatory: if provided and empty (no bins to visit), skip routing
        if mandatory is not None:
            has_mandatory = self._validate_mandatory(mandatory)
            if not has_mandatory:
                for handle in hook_data["handles"]:
                    handle.remove()
                return ([0, 0], 0, {"attention_weights": torch.tensor([]), "graph_masks": [], "mandatory_empty": True})

        mask = None
        input_for_model = input.copy()  # Shallow copy
        if "edges" not in input_for_model and edges is not None:
            input_for_model["edges"] = edges
        if "dist" not in input_for_model and dist_matrix is not None:
            input_for_model["dist"] = dist_matrix

        if mandatory is not None:
            input_for_model["mandatory"] = mandatory.unsqueeze(0) if mandatory.dim() == 1 else mandatory

        self._prepare_temporal_features(input_for_model, waste_history)

        # Expand dist_matrix for route improvement if needed (like in original code)
        if dist_matrix is not None and dist_matrix.dim() == 2:
            dist_matrix = dist_matrix.unsqueeze(0)
            # We don't have embeddings size yet to expand against batch.
            # But model forward will handle batching.

        strategy = params.decoding_strategy if params else "greedy"
        try:
            from logic.src.pipeline.simulations.actions.base import _record_live_params

            _record_live_params(
                "consumer_decoding",
                {
                    "strategy": strategy,
                    "beam_width": int(params.beam_width) if params else 1,
                    "reward_weight": float(params.reward_weight) if params else 0.0,
                    "length_penalty_alpha": float(params.length_penalty_alpha) if params else 0.0,
                    "applied": True,
                },
            )
        except Exception:
            pass
        if strategy == "beam_search":
            beam_width = params.beam_width if params else 1

            # 1. Initialize state with all relevant constraints and variables
            state = self.problem.make_state(
                input_for_model,
                cost_weights=cost_weights,
                profit_vars=profit_vars,
                mandatory=mandatory,
                dist_matrix=kwargs.get("dm_tensor"),
                dm=kwargs.get("dm_tensor"),
                **kwargs,
            )

            # 2. Encode graph features and precompute fixed embeddings for the decoder
            # Initial projection to latent space
            init_embeddings, _ = self.model._get_initial_embeddings(input_for_model)  # type: ignore[attr-defined]
            # Graph Attention Encoding: this is where nodes "see" each other
            embeddings = self.model.encoder(init_embeddings)  # type: ignore[attr-defined]
            # Decoder pre-computation: keys and values for the glimpse mechanism
            fixed = self.model.decoder._precompute(embeddings)  # type: ignore[attr-defined]

            # 3. Define expansion logic that bridges beam search and the model
            batch_size = state.ids.size(0)
            num_nodes = fixed.node_embeddings.size(1)
            step_count = [1]

            def propose_expansions(beam: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                # Get raw log-probs from decoder: [batch * beam_width, num_nodes]
                log_p = self.model.propose_expansions(beam, fixed, normalize=True)  # type: ignore[attr-defined]

                # Apply current environment mask
                current_mask = beam.state.get_mask()
                if current_mask is not None:
                    if current_mask.dim() == 3:
                        current_mask = current_mask.squeeze(1)
                    log_p = log_p.masked_fill(current_mask, float("-inf"))

                # 0. Get current beam width for indexing
                current_beam_width = log_p.size(0) // batch_size

                # Reward-aware scoring (incorporating Profit and Distance signals)
                if params and getattr(params, "reward_weight", 0.0) > 0:
                    td = beam.state.td
                    batch_indices = torch.arange(batch_size, device=log_p.device).repeat_interleave(current_beam_width)

                    # 1. Profit signal from the current state (waste levels)
                    waste = td["waste"]  # [B * beam, N] or [B * beam, N-1]
                    if waste.shape[-1] < log_p.shape[-1]:
                        # Depot at index 0 has zero profit
                        waste = torch.cat([torch.zeros_like(waste[:, :1]), waste], dim=-1)

                    # 2. Distance signal (look up in pre-expanded dist_matrix)
                    current_node = td["current_node"].squeeze(-1)  # [B * beam]
                    # dist_matrix is [B, N, N]
                    dists = dist_matrix[batch_indices, current_node, :]  # [B * beam, N]

                    # 3. Apply weights
                    # Use length weight from cost_weights if available
                    w_len = 0.1
                    if cost_weights and "length" in cost_weights:
                        cw_val = cost_weights["length"]
                        w_len = cw_val[batch_indices].unsqueeze(-1) if isinstance(cw_val, torch.Tensor) else cw_val

                    # Incremental reward proxy: Profit - Cost
                    incremental_reward = waste - dists * w_len
                    # Inject into log_p (additive in log-space acts as a soft priority)
                    log_p = log_p + params.reward_weight * incremental_reward

                # Un-normalize the beam score to get cumulative raw log_p
                lp_alpha = params.length_penalty_alpha if params else 0.0
                lp_prev = ((step_count[0] - 1) ** lp_alpha) if step_count[0] > 1 else 1.0
                raw_beam_score = beam.score * lp_prev

                # Combine current log-probs with accumulated raw beam scores
                combined_score = log_p + raw_beam_score.unsqueeze(-1)

                # Apply length penalty to the combined score for sorting and storing
                lp_curr = step_count[0] ** lp_alpha
                norm_score = combined_score / lp_curr

                # Reshape to [batch_size, current_beam_width * num_nodes]
                norm_score_batch = norm_score.view(batch_size, -1)
                valid_mask = ~torch.isinf(norm_score_batch)

                parents_list = []
                actions_list = []
                scores_list = []

                for b in range(batch_size):
                    valid_b = torch.nonzero(valid_mask[b]).squeeze(-1)

                    # Prune to top beam_width to save memory and avoid OOM
                    if valid_b.size(0) > beam_width:
                        sort_b = norm_score_batch[b, valid_b]
                        _, top_k_idx = torch.topk(sort_b, beam_width)
                        valid_b = valid_b[top_k_idx]

                    scores_b = norm_score_batch[b, valid_b]

                    # Map back to (parent_in_beam, action_index)
                    parent_b = (b * current_beam_width) + (valid_b // num_nodes)
                    action_b = valid_b % num_nodes

                    parents_list.append(parent_b)
                    actions_list.append(action_b)
                    scores_list.append(scores_b)

                step_count[0] += 1
                return torch.cat(parents_list), torch.cat(actions_list), torch.cat(scores_list)

            # 4. Execute Beam Search
            beams, _ = _beam_search(state, beam_width, propose_expansions)

            # 5. Backtrack to reconstruct the best action sequence
            actions_list = [b.action for b in beams[1:]]
            parents_list = [b.parent for b in beams[1:]]

            if not actions_list:
                # Safety fallback: if no nodes were visited, return dummy
                pi = torch.zeros((state.ids.size(0), 1), dtype=torch.long, device=state.ids.device)
            else:
                all_pi = backtrack(parents_list, actions_list)
                # Reshape to [batch, beam_width, seq_len] and take the best (index 0)
                pi = all_pi.view(state.ids.size(0), beam_width, -1)[:, 0, :]
        else:
            # Standard forward pass (Greedy/Sampling)
            # We explicitly pass the strategy to the model to avoid it trying to do
            # its own (potentially unsupported) beam search from internal config.
            model_strategy = strategy if strategy in ("greedy", "sampling") else "greedy"
            env = self._training_env(profit_vars, device=input_for_model["locs"].device)
            if env is not None:
                # Decode in the environment the model was trained in. The legacy path below uses
                # the old problem state, whose capacity, depot and stop rules differ from training,
                # so a model that skips empty bins in training visited them in the simulator.
                td = self._training_td(input_for_model, profit_vars, mandatory)
                out = self.model(env.reset(td), env, strategy=model_strategy)  # type: ignore[attr-defined]
            else:
                out = self.model(  # type: ignore[attr-defined]
                    input_for_model,
                    return_pi=True,
                    mask=mask,
                    strategy=model_strategy,
                    profit_vars=profit_vars,
                    cost_weights=cost_weights,
                )
            if isinstance(out, dict):
                pi = out.get("actions")
                if pi is None:
                    pi = out.get("pi")
            else:
                # Handle tuple output from model
                _, pi, _, _ = out

        if pi is None:
            raise ValueError("Model output does not contain 'actions' or 'pi'")

        route = torch.cat((torch.tensor([0]).to(pi.device), pi.squeeze(0)))
        cost = get_route_cost(distC * 100, route)
        for handle in hook_data["handles"]:
            handle.remove()

        attention_weights = torch.tensor([])
        if hook_data["weights"]:
            attention_weights = torch.stack(hook_data["weights"])

        route_list = route if isinstance(route, list) else route.cpu().tolist()
        n_visited = sum(1 for n in route_list if n != 0)
        if hasattr(self, "_viz_record"):
            self._viz_record(cost=float(cost), n_visited=n_visited, route_len=len(route_list))
        return (route_list, cost, {"attention_weights": attention_weights, "graph_masks": hook_data["masks"]})

    def _training_env(self, profit_vars: Optional[Dict[str, Any]], device: Any = "cpu") -> Optional[Any]:
        """Return the RL environment of the model's problem, priced with the simulator's economics.

        Args:
            profit_vars: Simulator profit variables (revenue_kg, bin_capacity, cost_km, vehicle_capacity).
            device: Device of the model input; the environment runs on it.

        Returns:
            The environment, or None when the model's problem has no RL environment.
        """
        from logic.src.envs import get_env

        env_name = getattr(self.model, "env_name", None) or getattr(self.problem, "NAME", None)
        if env_name not in ("ptp", "cptp"):
            return None
        pv = profit_vars or {}
        cache_key = (env_name, str(device), tuple(sorted((k, float(v)) for k, v in pv.items() if isinstance(v, (int, float)))))
        if getattr(self, "_env_cache_key", None) != cache_key:
            self._env = get_env(
                env_name,
                generator_params={"num_loc": 1},
                check_env_specs=False,
                waste_weight=float(pv.get("revenue_kg", 1.0)) * float(pv.get("bin_capacity", 1.0)),
                cost_weight=float(pv.get("cost_km", 1.0)),
                device=device,
            )
            if hasattr(self._env, "mask_empty_bins"):
                self._env.mask_empty_bins = True
            self._env_cache_key = cache_key
        return self._env

    @staticmethod
    def _training_td(
        input_for_model: Dict[str, Any], profit_vars: Optional[Dict[str, Any]], mandatory: Optional[torch.Tensor]
    ) -> Any:
        """Build the TensorDict the training environment resets from.

        Args:
            input_for_model: Simulator model input (locs, depot, waste in bin fills).
            profit_vars: Simulator profit variables; ``vehicle_capacity`` is in bin fills.
            mandatory: Optional mandatory-bin mask.

        Returns:
            TensorDict with one instance.
        """
        from tensordict import TensorDict

        locs = input_for_model["locs"]
        locs = locs.unsqueeze(0) if locs.dim() == 2 else locs
        depot = input_for_model["depot"]
        depot = depot.unsqueeze(0) if depot.dim() == 1 else depot
        waste = input_for_model["waste"]
        waste = waste.unsqueeze(0) if waste.dim() == 1 else waste
        td = TensorDict(
            {"locs": locs.float(), "depot": depot.float(), "waste": waste.float()},
            batch_size=[locs.size(0)],
            device=locs.device,
        )
        capacity = (profit_vars or {}).get("vehicle_capacity")
        if capacity is not None:
            td["capacity"] = torch.full((locs.size(0),), float(capacity), device=locs.device)
        if mandatory is not None:
            m = mandatory if isinstance(mandatory, torch.Tensor) else torch.as_tensor(mandatory)
            td["mandatory"] = (m.unsqueeze(0) if m.dim() == 1 else m).bool().to(locs.device)
        return td

    def _validate_mandatory(self, mandatory: torch.Tensor) -> torch.Tensor:
        """Validates if any bins are marked as mandatory.

        Args:
            mandatory: Tensor of mandatory bins.

        Returns:
            bool: True if any bins are marked as mandatory, False otherwise.
        """
        if not isinstance(mandatory, torch.Tensor):
            mandatory = torch.tensor(mandatory, dtype=torch.bool)
        if mandatory.dim() == 1:
            mandatory = mandatory.unsqueeze(0)
        return mandatory[:, 1:].any() if mandatory.size(1) > 1 else mandatory.any()

    def _prepare_temporal_features(self, input_for_model: Dict[str, Any], waste_history: torch.Tensor):
        """Populates temporal features (fill1, fill2, ...) based on waste history.

        Args:
            input_for_model: Input for the neural policy.
            waste_history: Waste history for the neural policy.

        """
        horizon = getattr(self.model, "temporal_horizon", 0)  # type: ignore[attr-defined]
        if horizon <= 0 or waste_history is None:
            return

        loc_tensor = input_for_model.get("locs") if "locs" in input_for_model.keys() else input_for_model.get("loc")
        if loc_tensor is None:
            raise KeyError("Input must contain 'loc' or 'locs'")

        h_feat = waste_history.unsqueeze(0) if waste_history.dim() == 2 else waste_history
        n_bins = loc_tensor.size(1) if loc_tensor.dim() == 3 else loc_tensor.size(0)

        if h_feat.size(1) != n_bins and h_feat.size(2) == n_bins:
            h_feat = h_feat.permute(0, 2, 1)

        if h_feat.max() > 2.0:
            h_feat = h_feat / 100.0

        for h in range(1, horizon + 1):
            if h_feat.size(2) >= horizon + 1:
                input_for_model[f"fill{h}"] = h_feat[:, :, -horizon - 1 : -1][:, :, h - 1]
            else:
                input_for_model[f"fill{h}"] = torch.zeros_like(h_feat[:, :, 0])
