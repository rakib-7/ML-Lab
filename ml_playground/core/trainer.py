"""Single entry point for training any of the four algorithms.

Wraps preprocessing + model into one sklearn Pipeline, so the fitted
object is self-contained and can be reused directly for prediction
without re-deriving scalers/encoders.
"""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .models import REGISTRY, UNSUPERVISED
from .preprocessing import build_feature_preprocessor, encode_target_if_needed
from utils import metrics as metric_utils


def train(algorithm_name: str, params: dict, X, y=None, test_size: float = 0.2):
    """
    Trains the chosen algorithm on X (and y, if supervised).

    Returns a dict with:
      pipeline        - fitted sklearn Pipeline (preprocessor + model)
      task            - "classification" or "clustering"
      metrics         - dict of evaluation metrics
      target_encoder  - LabelEncoder if the target was categorical, else None
      col_types       - {"numeric": [...], "categorical": [...]}
      extra           - task-specific extras used for plotting (test sets, labels, etc.)
    """
    model_module = REGISTRY[algorithm_name]
    task = model_module.TASK
    preprocessor, col_types = build_feature_preprocessor(X)
    model = model_module.build_model(params)
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])

    if algorithm_name in UNSUPERVISED:
        pipeline.fit(X)
        labels = pipeline.named_steps["model"].labels_
        X_transformed = pipeline.named_steps["prep"].transform(X)
        computed = metric_utils.clustering_metrics(X_transformed, labels)
        return {
            "pipeline": pipeline,
            "task": task,
            "metrics": computed,
            "target_encoder": None,
            "col_types": col_types,
            "extra": {"labels": labels, "X_transformed": X_transformed, "X_raw": X},
        }

    # Supervised path (KNN / SVM / Decision Tree)
    y_encoded, target_encoder = encode_target_if_needed(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42,
        stratify=y_encoded if len(set(y_encoded)) > 1 else None,
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    computed = metric_utils.classification_metrics(y_test, y_pred)

    return {
        "pipeline": pipeline,
        "task": task,
        "metrics": computed,
        "target_encoder": target_encoder,
        "col_types": col_types,
        "extra": {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
            "class_labels": target_encoder.classes_.tolist() if target_encoder else sorted(set(y_encoded)),
        },
    }
