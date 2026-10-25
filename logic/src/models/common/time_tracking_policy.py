"""Performance profiling wrapper for neural policies.

Provides timing instrumentation that automatically handles GPU synchronization
to ensure accurate duration measurements for CUDA-based model execution.

Attributes:
    TimeTrackingPolicy: wrapper module for tracking inference duration.

Example:
    >>> from logic.src.models.common import TimeTrackingPolicy
    >>> timed_policy = TimeTrackingPolicy(policy)
    >>> out = timed_policy(td, env)
    >>> print(f"Time: {out['inference_time']}")
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase


class TimeTrackingPolicy(nn.Module):
    """Wrapper for neural policies to track inference duration.

    Instruments the underlying policy to measure execution time, automatically
    handling GPU synchronization to ensure accurate profiling of CUDA kernels.
    The resulting duration is injected into the output dictionary.

    Attributes:
        policy: The underlying neural policy being profiled.
    """

    def __init__(self, policy: nn.Module):
        """Initializes the timing wrapper.

        Args:
            policy: Neural policy module to wrap for performance tracking.
        """
        super().__init__()
        self.policy = policy

    def forward(self, td: TensorDict, env: Optional[RL4COEnvBase] = None, **kwargs: Any) -> Dict[str, Any]:
        """Executes the wrapped policy and measures elapsed time.

        Synchronizes the CUDA device if necessary before and after execution to
        guarantee that the recorded time includes asynchronous GPU kernel completion.

        Args:
            td: Input state data as a TensorDict.
            env: Optional problem environment defining physical constraints.
            kwargs: Additional model parameters passed to the inner policy.

        Returns:
            Dict[str, Any]: Combined output from the policy including a new
                'inference_time' tensor scalar.
        """
        # Synchronize for accurate GPU timing
        if td.device is not None and td.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        # Run policy
        out = self.policy(td, env, **kwargs)

        if td.device is not None and td.device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        # Calculate duration
        duration = end - start

        # Inject into output
        # Use same device as reward for mathematical compatibility
        device = out["reward"].device if "reward" in out else td.device

        out["inference_time"] = torch.tensor(duration, dtype=torch.float32, device=device)

        return out

    def __getattr__(self, name: str) -> Any:
        """Forwards attribute access to the underlying policy.

        Ensures that parameters, buffers, and submodules of the wrapped policy
        remain directly accessible via the wrapper instance.

        Args:
            name: Name of the attribute to retrieve.

        Returns:
            Any: The attribute from this wrapper or the wrapped policy.

        Raises:
            AttributeError: If the attribute is not found in either the wrapper
                or the policy.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.policy, name)
