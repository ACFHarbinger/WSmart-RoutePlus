"""Abstract base of the cutting-plane separation engines.

Attributes:
    CuttingPlaneEngine: Separation engine interface and shared orthogonality filter.

Example:
    >>> class MyEngine(CuttingPlaneEngine): ...
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import VRPPMasterProblem


class CuttingPlaneEngine(ABC):
    """Abstract base class for cutting plane separation engines.

    Each engine implements a specific family of valid inequalities and
    provides separation algorithms to identify violated cuts.

    Attributes:
        None (abstract).
    """

    @abstractmethod
    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """Identify violated cuts and add them to the master problem.

        Args:
            master: Master problem instance.
            max_cuts: Maximum number of cuts to add.
            kwargs: Engine-specific parameters (e.g., node_depth, thresholds).

        Returns:
            int: Number of cuts added.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the engine name for logging.

        Returns:
            str: Name of the engine.
        """
        pass

    @property
    def engines(self) -> List["CuttingPlaneEngine"]:
        """Returns the list of sub-engines if this is a composite engine.

        Returns:
            List[CuttingPlaneEngine]: A list of child engines, or an empty list if not composite.
        """
        return []

    def _is_orthogonal(self, candidate_vec: np.ndarray, active_vecs: List[np.ndarray], threshold: float = 0.8) -> bool:
        """Calculates maximum cosine similarity between candidate and active vectors.

        Rejects if similarity > threshold (i.e. if cut is too parallel to existing cuts).

        Args:
            candidate_vec: Coefficient vector of the new cut.
            active_vecs: List of coefficient vectors of existing cuts.
            threshold: Maximum allowed cosine similarity.

        Returns:
            bool: True if orthogonal enough, False otherwise.
        """
        if not active_vecs:
            return True

        norm_c = np.linalg.norm(candidate_vec)
        if norm_c < 1e-6:
            return False

        for a_vec in active_vecs:
            norm_a = np.linalg.norm(a_vec)
            if norm_a < 1e-6:
                continue
            cos_sim = np.abs(np.dot(candidate_vec, a_vec)) / (norm_c * norm_a)
            if cos_sim > threshold:
                return False
        return True
