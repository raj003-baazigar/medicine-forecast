import os
import tensorflow as tf
from tensorflow import keras
from keras.saving import register_keras_serializable

# --- Custom metric for classifier ---
@register_keras_serializable()
def f1_score_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    precision = tp / (tf.reduce_sum(tf.cast(y_pred, tf.float32)) + 1e-8)
    recall = tp / (tf.reduce_sum(tf.cast(y_true, tf.float32)) + 1e-8)
    return 2 * ((precision * recall) / (precision + recall + 1e-8))


class DemandForecaster2026:
    def __init__(self, base_path: str, model_type: str = "regressor", target_element: str = "demand", lookback: int = 12):
        """
        Demand Forecasting Class.

        Args:
            base_path (str): Base directory where models are stored.
            model_type (str): "regressor" or "classifier".
            target_element (str): Column to forecast (default: demand).
            lookback (int): Lookback window for time-series.
        """
        self.base_path = base_path
        self.target_element = target_element
        self.lookback = lookback
        self.model_type = model_type.lower()

        if self.model_type == "regressor":
            model_file = "optimized_regressor_model.keras"
        elif self.model_type == "classifier":
            model_file = "optimized_classifier_model.keras"
        else:
            raise ValueError("❌ model_type must be 'regressor' or 'classifier'")

        keras_model_path = os.path.join(self.base_path, model_file)
        if not os.path.exists(keras_model_path):
            raise FileNotFoundError(f"❌ Model file not found at {keras_model_path}")

        # Load model (with custom metric for classifier)
        custom_objects = {"f1_score_metric": f1_score_metric} if self.model_type == "classifier" else {}
        self.model = keras.models.load_model(keras_model_path, custom_objects=custom_objects)

    def forecast(self, input_data):
        """Run model prediction"""
        return self.model.predict(input_data)
