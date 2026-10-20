"""
Learned Packing Algorithm Module.

Implements a packing algorithm that uses a pre-trained machine learning model
(e.g., Random Forest, XGBoost, or a Neural Network) to imitate an expensive
exact solver (like the MIP Knapsack). It extracts features for each bin and
predicts the probability that the bin should be selected.

Attributes:
    LearnedSelection: The Learned Packing algorithm.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_learned import LearnedSelection
    >>> strategy = LearnedSelection()
    >>> bins = strategy.pack(context)
"""

import os
from typing import Any, List, Optional, Tuple

import joblib
import numpy as np
import torch

from logic.src.enums import PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry
from logic.src.utils.registry import ALL


@ALL.register(
    PolicyTag.PACKING,
    PolicyTag.REINFORCEMENT_LEARNING,
)
@PackingRegistry.register("learned")
class LearnedSelection(IPackingAlgorithm):
    """Selection strategy based on a pre-trained imitation model.

    Attributes:
        model_path (Optional[str]): Path to the model file.
        _learned_threshold (float): Prediction probability threshold.
        _model (Any): The loaded model.
        _model_type (Optional[str]): Type of model ('sklearn' or 'torch').
    """

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5):
        """
        Initializes the learned packing algorithm.

        Args:
            model_path: Optional path to the learned model.
            threshold: Selection threshold.
        """
        self.model_path = model_path
        self._learned_threshold = threshold
        self._model: Any = None
        self._model_type: Optional[str] = None

    def _load_model(self, path: str) -> None:
        """Lazy load the model from the specified path.

        Args:
            path: Path to the model file.

        """
        if not os.path.exists(path):
            raise RuntimeError(f"LearnedSelection: Model file not found at {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".pkl":
            self._model = joblib.load(path)
            self._model_type = "sklearn"
        elif ext == ".pt":
            self._model = torch.load(path, map_location="cpu", weights_only=False)
            self._model.eval()
            self._model_type = "torch"
        else:
            raise RuntimeError(f"LearnedSelection: Unsupported model extension {ext}")

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Extract features and predict selection using the learned model.

        Extracts features for each bin (fill ratio, accumulation rate, distance,
        revenue, etc.) and uses the pre-trained model to predict selection
        probability.

        Args:
            context: PackingContext with all bin properties.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.

        Raises:
            RuntimeError: If no model path is provided or model cannot be loaded.
        """
        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "LearnedSelection"})

        # Resolve model path from context if not provided at init
        path = self.model_path if self.model_path is not None else getattr(context, "learned_model_path", None)
        if path is None:
            raise RuntimeError("LearnedSelection: No model_path provided.")

        if self._model is None:
            self._load_model(path)

        # 1. Feature Extraction (n_bins, 6)
        # Features: [fill_ratio, accumulation_rate, std_dev, dist_to_depot, revenue, days_to_overflow]
        fill_ratio = context.current_fill / context.max_fill
        acc_rate = context.accumulation_rates if context.accumulation_rates is not None else np.zeros(n_bins)
        std_dev = context.std_deviations if context.std_deviations is not None else np.zeros(n_bins)

        dist_to_depot = np.zeros(n_bins)
        if context.distance_matrix is not None:
            dist_to_depot = context.distance_matrix[0, 1:]

        bin_cap = context.bin_volume * context.bin_density
        revenue = fill_ratio * bin_cap * context.revenue_kg

        # Remaining capacity / acc_rate
        days_to_overflow = np.where(acc_rate > 0, (context.max_fill - context.current_fill) / acc_rate, 99.0)

        features = np.column_stack([fill_ratio, acc_rate, std_dev, dist_to_depot, revenue, days_to_overflow])

        # 2. Prediction
        threshold = (
            self._learned_threshold
            if self._learned_threshold is not None
            else getattr(context, "learned_threshold", 0.5)
        )

        if self._model_type == "sklearn":
            if hasattr(self._model, "predict_proba"):
                probs = self._model.predict_proba(features)[:, 1]
            else:
                probs = self._model.predict(features)
        elif self._model_type == "torch":
            with torch.no_grad():
                input_tensor = torch.from_numpy(features).float()
                # Assuming the model returns probabilities or logits
                output = self._model(input_tensor)
                if output.shape[-1] > 1:
                    probs = torch.softmax(output, dim=-1)[:, 1].numpy()
                else:
                    probs = torch.sigmoid(output).squeeze().numpy()
        else:
            raise RuntimeError("LearnedSelection: Model not loaded correctly.")

        mandatory_indices = np.nonzero(probs >= threshold)[0]

        return sorted((mandatory_indices + 1).tolist()), SearchContext.initialize(
            selection_metrics={"strategy": "LearnedSelection"}
        )
