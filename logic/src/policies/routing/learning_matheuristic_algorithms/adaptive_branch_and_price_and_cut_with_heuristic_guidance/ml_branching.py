"""
ML-Guided Node Selection and Variable Fixing.

Attributes:
    MLBranchingStrategy: A class that implements the ML-guided node selection and variable fixing strategy.

    Role in the Pipeline
    --------------------
    Plugs into the BPC Branch-and-Bound tree as the variable-selection oracle.
    At each fractional LP relaxation, the oracle selects the variable to branch
    on.  Two modes are supported:

    **GNN surrogate (when** ``model`` **is set)**
        A graph neural network trained via imitation learning on strong-
        branching decisions encodes the B&B state as a bipartite graph and
        predicts a ranking score for each fractional variable.  Inference is
        O(|variables|) per node, compared to O(|variables| · LP_solves) for
        exact strong branching.

    **Reliability branching (fallback)**
        Selects the variable with the highest pseudo-cost score:

            Score(xᵢ) = min(Ψ↓ᵢ, Ψ↑ᵢ) + c · max(Ψ↓ᵢ, Ψ↑ᵢ)

        Variables not yet evaluated (no pseudo-cost history) receive score
        +∞ to prioritise exploration of unexplored candidates first.

    Pseudo-cost Update
    ------------------
    After each exact evaluation (strong branching or child LP solve):

        Ψ_new = α · Ψ_old + (1 − α) · Δ_observed

    where Δ is the observed objective degradation per unit of fractionality,
    and α is the EMA coefficient (``pseudocost_ema_alpha``).

    GNN Feature Representation
    --------------------------
    ``compute_gnn_features`` constructs a bipartite graph representation of
    the current fractional LP state:

    Node features (one per fractional variable):
        [current_fill, mean_fill_rate, variance_of_scenario_prizes,
         expected_days_to_overflow]

    Edge features (one per fractional arc in the LP):
        [λ_ij, |reduced_cost_ij|]

    This representation is compatible with standard B&B GNN architectures
    (e.g. Gasse et al. 2019, Nair et al. 2020).

Example:
    >>> import numpy as np
    >>> from my_gnn_model_pkg import load_gnn_branching_model
    >>> model = load_gnn_branching_model("path/to/trained.pth")
    >>> branching_oracle = MLBranchingStrategy(model=model)
    ...
"""

from typing import Any, Dict, Optional, Tuple


class MLBranchingStrategy:
    """
    Imitation-learning surrogate for strong branching, falling back to
    reliability branching when no trained model is available.

    Role in the Pipeline
    --------------------
    Plugs into the BPC Branch-and-Bound tree as the variable-selection oracle.
    At each fractional LP relaxation, the oracle selects the variable to branch
    on.  Two modes are supported:

    **GNN surrogate (when** ``model`` **is set)**
        A graph neural network trained via imitation learning on strong-
        branching decisions encodes the B&B state as a bipartite graph and
        predicts a ranking score for each fractional variable.  Inference is
        O(|variables|) per node, compared to O(|variables| · LP_solves) for
        exact strong branching.

    **Reliability branching (fallback)**
        Selects the variable with the highest pseudo-cost score:

            Score(xᵢ) = min(Ψ↓ᵢ, Ψ↑ᵢ) + c · max(Ψ↓ᵢ, Ψ↑ᵢ)

        Variables not yet evaluated (no pseudo-cost history) receive score
        +∞ to prioritise exploration of unexplored candidates first.

    Pseudo-cost Update
    ------------------
    After each exact evaluation (strong branching or child LP solve):

        Ψ_new = α · Ψ_old + (1 − α) · Δ_observed

    where Δ is the observed objective degradation per unit of fractionality,
    and α is the EMA coefficient (``pseudocost_ema_alpha``).

    GNN Feature Representation
    --------------------------
    ``compute_gnn_features`` constructs a bipartite graph representation of
    the current fractional LP state:

    Node features (one per fractional variable):
        [current_fill, mean_fill_rate, variance_of_scenario_prizes,
         expected_days_to_overflow]

    Edge features (one per fractional arc in the LP):
        [λ_ij, |reduced_cost_ij|]

    This representation is compatible with standard B&B GNN architectures
    (e.g. Gasse et al. 2019, Nair et al. 2020).

    Attributes:
    ----------
    model : Any or None
        Trained GNN branching model.  ``None`` activates the reliability
        branching fallback for the entire solve.
    reliability_c : float
        Blending coefficient c in the reliability branching score.
    pseudocost_ema_alpha : float
        EMA coefficient α for pseudo-cost running average updates.
    historical_pseudocosts : Dict[int, Tuple[float, float]]
        Running pseudo-cost estimates {var_id: (Ψ↓, Ψ↑)}.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        reliability_c: float = 1.0,
        pseudocost_ema_alpha: float = 0.5,
    ):
        """
        Args:
            model: Trained GNN branching surrogate.  If ``None``, reliability
                branching is used exclusively.
            reliability_c: Blending coefficient c in Score(xᵢ) =
                min(Ψ↓, Ψ↑) + c · max(Ψ↓, Ψ↑).  c = 1 weights both
                directions equally; c > 1 penalises high asymmetry.
            pseudocost_ema_alpha: EMA coefficient α ∈ (0, 1] for pseudo-cost
                updates.  α = 1 ignores history (always takes the new
                observation); α → 0 freezes estimates.
        """
        self.model = model
        self.reliability_c = reliability_c
        self.pseudocost_ema_alpha = pseudocost_ema_alpha
        self.historical_pseudocosts: Dict[int, Tuple[float, float]] = {}

    def _reliability_score(self, var_id: int) -> float:
        """
        Compute the reliability branching score for a variable.

            Score(xᵢ) = min(Ψ↓ᵢ, Ψ↑ᵢ) + c · max(Ψ↓ᵢ, Ψ↑ᵢ)

        Variables with no pseudo-cost history receive score +∞ to ensure
        they are evaluated (via exact strong branching or child LP solve)
        before variables with established history.

        Args:
            var_id: Variable identifier.

        Returns:
            Reliability branching score.  +∞ for unexplored variables.
        """
        if var_id not in self.historical_pseudocosts:
            return float("inf")

        psi_down, psi_up = self.historical_pseudocosts[var_id]
        return min(psi_down, psi_up) + self.reliability_c * max(psi_down, psi_up)
