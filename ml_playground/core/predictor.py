"""Turns a single user-entered row into a prediction using the fitted pipeline."""
import pandas as pd


def predict_single(pipeline, feature_columns: list, raw_values: dict, target_encoder=None):
    """
    raw_values: dict of {column_name: value} for one row, in the same
    schema the pipeline was trained on.
    """
    row = pd.DataFrame([{col: raw_values.get(col) for col in feature_columns}])
    prediction = pipeline.predict(row)[0]

    result = {"raw_prediction": prediction}
    if target_encoder is not None:
        result["label"] = target_encoder.inverse_transform([prediction])[0]
    else:
        result["label"] = prediction

    if hasattr(pipeline.named_steps.get("model", pipeline), "predict_proba"):
        try:
            proba = pipeline.predict_proba(row)[0]
            classes = pipeline.named_steps["model"].classes_
            if target_encoder is not None:
                classes = target_encoder.inverse_transform(classes)
            result["probabilities"] = dict(zip(classes, proba))
        except Exception:
            pass
    return result
