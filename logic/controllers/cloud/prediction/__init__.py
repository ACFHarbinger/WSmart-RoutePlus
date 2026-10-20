"""Runtime and memory prediction for cloud cells (CR-11).

Attributes:
    Prediction: One cell's predicted wall-time quantiles and memory.
    RuntimePredictor: Quantile wall-time model, machine factors and memory estimates.
    default_predictor: Fitted on the bundled training table (cached).
    fitted_predictor: The bundled table plus a control plane's completed tasks.
    cell_features: Feature dict of a cell.
    record_predictions: Store a job's predictions as a ``prediction`` event.
    prediction_accuracy: Predicted against actual wall time of a finished job.
"""

from logic.controllers.cloud.prediction.features import cell_features
from logic.controllers.cloud.prediction.history import (
    cloud_records,
    fitted_predictor,
    prediction_accuracy,
    record_predictions,
)
from logic.controllers.cloud.prediction.model import Prediction, RuntimePredictor, default_predictor, load_table

__all__ = [
    "Prediction",
    "RuntimePredictor",
    "cell_features",
    "cloud_records",
    "default_predictor",
    "fitted_predictor",
    "load_table",
    "prediction_accuracy",
    "record_predictions",
]
