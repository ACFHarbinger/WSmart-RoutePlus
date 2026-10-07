"""
Base problem definition and legacy constants.

Attributes:
    BaseProblem: Class definition of BaseProblem.

Example:
    None
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.envs import get_env
from logic.src.interfaces import ITraversable
from logic.src.utils.data.td_state_wrapper import TensorDictStateWrapper


class BaseProblem:
    """
    Legacy base class for routing problems.

    Attributes:
        NAME: Name of the problem
    """

    NAME: str = "base"

    @staticmethod
    def validate_tours(pi: torch.Tensor) -> bool:
        """Validates tours (no duplicates except depot).

        Args:
            pi: Tensor of tours to validate, shape (batch, sequence_length)

        Returns:
            bool: True if tours are valid, False otherwise
        """
        if pi.size(-1) <= 1:
            return True
        sorted_pi: torch.Tensor = pi.data.sort(1)[0]
        if not ((sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all():
            raise ValueError("Tour validation failed: duplicates detected (excluding depot).")
        return True

    @staticmethod
    def get_waste_with_depot(
        dataset: Dict[str, Any],
        pi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract waste tensor guaranteed to include depot at index 0.

        If dataset['waste'] has width N (lacks depot column), prepends a zero
        depot waste column. If it already has width N+1 (e.g. from env.reset),
        returns it directly without shifting customer indices.

        Args:
            dataset: Problem dataset containing 'waste', optionally 'locs',
                'depot', 'visited', or 'num_loc'.
            pi: Optional tour tensor [batch, nodes].

        Returns:
            torch.Tensor: Waste tensor with depot at index 0.
        """
        waste = dataset["waste"]
        locs = dataset.get("locs") if "locs" in dataset else dataset.get("loc")
        depot = dataset.get("depot")

        # Structural metadata takes precedence over coordinate equality: a
        # customer may legitimately share the depot location or have zero waste.
        num_customers = None
        if "visited" in dataset:
            num_customers = dataset["visited"].shape[-1] - 1
        elif "num_loc" in dataset:
            num_customers = int(dataset["num_loc"])
        elif locs is not None and depot is not None and locs.shape[-2] > 1:
            num_customers = locs.shape[-2] - int(torch.allclose(locs[..., 0, :], depot, atol=1e-4))

        if num_customers is not None:
            if waste.shape[-1] == num_customers + 1:
                return waste
            if waste.shape[-1] != num_customers:
                raise ValueError("waste width must equal num_loc or num_loc + 1")

        # Without schema metadata retain the legacy customer-only contract.
        # Neither a zero first demand nor a partial tour proves a depot column.
        return torch.cat((waste.new_zeros(*waste.shape[:-1], 1), waste), dim=-1)

    @staticmethod
    def get_tour_length(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates tour length.

        Args:
            dataset (Dict[str, Any]): Dataset containing node locations and optionally edges.
                Expected keys: 'locs' (Tensor[batch, num_nodes, 2]) or 'edges' (Tensor[batch, num_nodes, num_nodes]).
                'depot' (Tensor[batch, 2]) is also supported.
            pi (torch.Tensor): Tensor of tours to validate, shape (batch, sequence_length).
                Tours should include depot at start/end if applicable.
            dist_matrix (Optional[torch.Tensor]): Optional precomputed distance matrix,
                shape (batch, num_nodes, num_nodes) or (num_nodes, num_nodes).

        Returns:
            torch.Tensor: Tensor of tour lengths, shape (batch,)
        """
        if pi.size(-1) <= 1:
            return torch.zeros(pi.size(0), device=pi.device)

        use_dist_matrix = dist_matrix is not None and isinstance(dist_matrix, torch.Tensor)
        if use_dist_matrix:
            # Simple distance matrix lookup
            if dist_matrix.dim() == 2:  # type: ignore[union-attr]
                dist_matrix = dist_matrix.unsqueeze(0)  # type: ignore[union-attr]
            src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
            dst_mask: torch.Tensor = dst_vertices != 0
            pair_mask: torch.Tensor = (src_vertices != 0) & (dst_mask)
            dists: torch.Tensor = dist_matrix[0, src_vertices, dst_vertices] * pair_mask.float()  # type: ignore[index]
            last_dst: torch.Tensor = torch.max(
                dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device),
                dim=1,
            ).indices
            length: torch.Tensor = (
                dist_matrix[  # type: ignore[index]
                    0, dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0
                ]
                + dists.sum(dim=1)
                + dist_matrix[0, 0, pi[:, 0]]  # type: ignore[index]
            )
        else:
            loc_val = dataset.get("locs") if "locs" in dataset else dataset.get("loc")
            waste_val = dataset.get("waste")
            if "visited" in dataset:
                has_depot_in_loc = loc_val is not None and loc_val.size(1) == dataset["visited"].size(-1)
            elif "num_loc" in dataset:
                has_depot_in_loc = loc_val is not None and loc_val.size(1) == int(dataset["num_loc"]) + 1
            else:
                has_depot_in_loc = (
                    loc_val is not None
                    and "depot" in dataset
                    and loc_val.size(1) > 1
                    and torch.allclose(loc_val[:, 0, :], dataset["depot"], atol=1e-4)
                ) or (loc_val is not None and waste_val is not None and loc_val.size(1) == 1 + waste_val.size(1))
            if has_depot_in_loc:
                # already concatenated
                loc_with_depot: Any = loc_val
            elif loc_val is not None:
                loc_with_depot = torch.cat((dataset["depot"][:, None, :], loc_val), 1)
            else:
                # Fallback for empty/missing loc
                return torch.zeros(pi.size(0), device=pi.device)

            d: torch.Tensor = loc_with_depot.gather(1, pi.unsqueeze(-1).expand(*pi.size(), loc_with_depot.size(-1)))
            length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)
                + (d[:, 0] - dataset["depot"]).norm(p=2, dim=-1)
                + (d[:, -1] - dataset["depot"]).norm(p=2, dim=-1)
            )
        return length

    @classmethod
    def beam_search(
        cls,
        input: Union[ITraversable, TensorDict],
        beam_size: int,
        cost_weights: Union[torch.Tensor, List[float], float, Dict[str, float]],
        model: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> TensorDict:
        """Beam search bridge.

        Args:
            input (Union[ITraversable, TensorDict]): Input data, can be ITraversable or TensorDict.
            beam_size (int): Number of beams for beam search.
            cost_weights (Union[torch.Tensor, List[float], float, Dict[str, float]]): Weights for cost calculation.
            model (Optional[nn.Module]): Model to use for beam search.
            kwargs: Additional keyword arguments.

        Returns:
            TensorDict: TensorDict containing the best tours found by beam search.
        """
        from logic.src.utils.decoding import beam_search as beam_search_func

        assert model is not None
        fixed = model.precompute_fixed(input, edges=input.get("edges"))

        def propose_expansions(beam):
            """Propose expansions for the current beam search state."""
            if model is None:
                raise ValueError("Model is required for proposing expansions.")
            return model.propose_expansions(beam, fixed, normalize=True)

        # Note: make_state is problem-specific, must be implemented by subclasses
        state = cls.make_state(input, cost_weights=cost_weights, **kwargs)
        return beam_search_func(state, beam_size, propose_expansions)

    @classmethod
    def make_state(
        cls, input_data: Any, edges: Any = None, cost_weights: Any = None, dist_matrix: Any = None, **kwargs: Any
    ) -> Any:
        """
        Bridge to RL4CO environments.
        Initializes a TensorDict from the input and returns a state wrapper.
        Args:
            input_data (Any): Input data.
            edges (Any): Edges data.
            cost_weights (Any): Weights for cost calculation.
            dist_matrix (Any): Distance matrix.
            kwargs (Any): Additional keyword arguments.

        Returns:
            Any: State wrapper.
        """
        env_name = cls.NAME
        if isinstance(input_data, ITraversable):
            bs, device = cls._get_batch_info(input_data)
            env = get_env(env_name, batch_size=torch.Size([bs]), device=device)
            td = cls._prepare_td_from_traversable(input_data, bs, device)
        elif isinstance(input_data, TensorDict):
            td = input_data
            bs = td.batch_size[0] if len(td.batch_size) > 0 else 1
            env = get_env(env_name, batch_size=torch.Size([bs]), device=td.device)
        else:
            td = TensorDict({}, batch_size=[1])
            env = get_env(env_name, batch_size=torch.Size([1]))

        cls._ensure_required_keys(td, env_name, edges, dist_matrix, **kwargs)

        td_reset = TensorDict(
            source={k: v for k, v in td.items()},
            batch_size=td.batch_size,
            device=td.device,
        )
        td = env.reset(td_reset)
        return TensorDictStateWrapper(td, env_name, env=env)

    @classmethod
    def make_dataset(
        cls,
        filename: Optional[str] = None,
        num_samples: Optional[int] = None,
        offset: int = 0,
        **kwargs: Any,
    ) -> "torch.utils.data.Dataset":
        """Build an evaluation dataset of problem instances.

        Supported sources:

        - ``.npz`` simulator datasets written by ``gen_data`` (keys ``locs``,
          ``depot``, ``waste`` with shape ``[samples, days, nodes]`` and
          ``max_waste``): every (sample, day) pair becomes one instance and
          fill levels are normalised by ``max_waste``.
        - ``.td`` / ``.pt`` TensorDict datasets written by the training pipeline.
        - ``.pkl`` pickles holding either a list of instance dicts / tuples
          ``(depot, locs, waste)`` or a dict of arrays.
        - no filename: ``num_samples`` random instances from the environment
          generator (``size`` selects the number of nodes).

        Args:
            filename: Dataset path (relative paths are resolved against the project root).
            num_samples: Number of instances to keep (``None`` keeps everything after ``offset``).
            offset: Number of leading instances to skip.
            kwargs: Extra options; ``size`` is used when generating instances.

        Returns:
            A map-style dataset whose items are plain ``{key: tensor}`` dicts.
        """
        import os
        import pickle

        import numpy as np

        from logic.src.constants import ROOT_DIR

        def _to_tensor(v: Any) -> torch.Tensor:
            return v.float() if torch.is_tensor(v) else torch.as_tensor(np.asarray(v), dtype=torch.float32)

        if filename is None:
            n = int(num_samples or 1)
            env = get_env(cls.NAME, num_loc=int(kwargs.get("size") or 20), batch_size=torch.Size([n]))
            td = env.generator(n) if hasattr(env, "generator") else env.reset()
            data = {k: _to_tensor(v) for k, v in td.items() if torch.is_tensor(v)}
            return _InstanceDataset(data)

        path = filename if os.path.isabs(filename) or os.path.exists(filename) else os.path.join(ROOT_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {filename}")
        ext = os.path.splitext(path)[1].lower()

        data: Dict[str, torch.Tensor]
        if ext == ".npz":
            raw = dict(np.load(path, allow_pickle=True))
            locs = _to_tensor(raw["locs"])  # [S, N, 2]
            depot = _to_tensor(raw["depot"])  # [S, 2]
            waste = _to_tensor(raw["waste"])  # [S, D, N] or [S, N]
            if waste.dim() == 2:
                waste = waste.unsqueeze(1)
            n_samples, n_days = waste.shape[0], waste.shape[1]
            max_waste = _to_tensor(raw["max_waste"]) if "max_waste" in raw else torch.ones(n_samples)
            waste = waste / max_waste.view(-1, 1, 1).clamp_min(1e-9)
            data = {
                "locs": locs.unsqueeze(1).expand(-1, n_days, -1, -1).reshape(n_samples * n_days, *locs.shape[1:]),
                "depot": depot.unsqueeze(1).expand(-1, n_days, -1).reshape(n_samples * n_days, -1),
                "waste": waste.reshape(n_samples * n_days, -1),
            }
        elif ext in (".td", ".pt"):
            td = torch.load(path, weights_only=False)
            data = {k: _to_tensor(v) for k, v in td.items() if torch.is_tensor(v)}
        elif ext == ".pkl":
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            if isinstance(obj, dict):
                data = {k: _to_tensor(v) for k, v in obj.items()}
            else:
                rows = []
                for inst in obj:
                    if isinstance(inst, dict):
                        rows.append({cls._map_key(k): _to_tensor(v) for k, v in inst.items()})
                    else:  # legacy tuple layout: (depot, locs, waste, ...)
                        depot_i, locs_i, waste_i = inst[0], inst[1], inst[2]
                        rows.append(
                            {"depot": _to_tensor(depot_i), "locs": _to_tensor(locs_i), "waste": _to_tensor(waste_i)}
                        )
                data = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
        else:
            raise ValueError(f"Unsupported dataset format '{ext}' (expected .npz, .td, .pt or .pkl)")

        total = next(iter(data.values())).shape[0]
        end = total if num_samples is None else min(total, offset + int(num_samples))
        data = {k: v[offset:end] for k, v in data.items()}
        return _InstanceDataset(data)

    @staticmethod
    def _get_batch_info(input_data: Any) -> tuple[int, torch.device]:
        """Extract batch size and device from input data.

        Args:
            input_data: Input data.

        Returns:
            tuple[int, torch.device]: Batch size and device.
        """
        bs = 1
        device = torch.device("cpu")
        for k in ["loc", "locs", "waste"]:
            if k in input_data and torch.is_tensor(input_data[k]):
                bs = input_data[k].size(0)
                device = input_data[k].device
                break
        return bs, device

    @classmethod
    def _prepare_td_from_traversable(cls, input_data: Any, bs: int, device: torch.device) -> TensorDict:
        """Create a TensorDict from an ITraversable object.

        Args:
            input_data: Input data.
            bs: Batch size.
            device: Device.

        Returns:
            TensorDict: TensorDict containing the batched input data.
        """
        td_data = {}
        for k, v in input_data.items():
            if torch.is_tensor(v):
                target_key = cls._map_key(k)
                td_data[target_key] = cls._batch_tensor(v, bs)
            else:
                td_data[k] = v
        return TensorDict(td_data, batch_size=[bs], device=device)

    @staticmethod
    def _map_key(k: str) -> str:
        """Map legacy keys to environment keys.

        Args:
            k (str): Key to map.

        Returns:
            str: Mapped key.
        """
        if k == "loc":
            return "locs"
        if k in ["wastes"]:
            return "waste"
        return k

    @staticmethod
    def _batch_tensor(v: torch.Tensor, bs: int) -> torch.Tensor:
        """Ensure tensor has correct batch dimension.

        Args:
            v (torch.Tensor): Tensor to batch.
            bs (int): Batch size.

        Returns:
            torch.Tensor: Tensor with correct batch dimension.
        """
        if v.dim() >= 1 and v.size(0) == bs:
            return v
        if v.dim() >= 2:
            return v.unsqueeze(0).expand(bs, *([-1] * v.dim()))
        return v.expand(bs, *([-1] * v.dim())) if v.dim() > 0 else v.unsqueeze(0).expand(bs)

    @staticmethod
    def _ensure_required_keys(td: TensorDict, env_name: str, edges: Any, dist_matrix: Any, **kwargs: Any) -> None:
        """Ensure all required keys are present in the TensorDict.

        Args:
            td (TensorDict): TensorDict to check.
            env_name (str): Environment name.
            edges (Any): Edges data.
            dist_matrix (Any): Distance matrix.
            kwargs (Any): Additional keyword arguments.
        """
        bs = td.batch_size[0]
        if "dist" not in td.keys() and dist_matrix is not None:
            td["dist"] = dist_matrix
        if "edges" not in td.keys() and edges is not None:
            td["edges"] = edges
        if "locs" not in td.keys() and "loc" in td.keys():
            td["locs"] = td["loc"]
        if "capacity" not in td.keys():
            cap = kwargs.get("vehicle_capacity") or kwargs.get("profit_vars", {}).get("vehicle_capacity")
            if cap:
                td["capacity"] = torch.full((bs,), cap, device=td.device)


class _InstanceDataset(torch.utils.data.Dataset):
    """Map-style dataset over a dict of equally sized tensors (items are plain dicts)."""

    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
        self._n = next(iter(data.values())).shape[0] if data else 0

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {k: v[index] for k, v in self.data.items()}
