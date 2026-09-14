"""
One-vs-rest Perceptron classifier, implemented from scratch (Hagan-style
hardlim learning rule) rather than calling sklearn's built-in Perceptron.

This generalizes the logic from the uploaded fruit-classification app into
a proper sklearn-compatible estimator (fit/predict/predict_proba), so it
slots into the same ColumnTransformer + Pipeline used by every other
algorithm in this project, and works on ANY uploaded dataset -- not just
the fruit example.

Differences from the original app (not a copy):
  - Works for any number of classes/features, not hardcoded to 3 features
    and 4 fruits.
  - Trains one binary hardlim perceptron per class (one-vs-rest) inside a
    single estimator object instead of a dict of ad-hoc numpy arrays.
  - Exposes decision_function/predict_proba (softmax over net inputs) so
    it participates in the shared probability-bar-chart UI.
  - Keeps full per-class, per-epoch weight/bias history for the "how did
    this converge" visualization, but stores it as attributes on the
    fitted estimator instead of loose variables scattered through a script.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

TASK = "classification"

PARAM_SPEC = {
    "learning_rate": {
        "type": "float", "label": "Learning rate",
        "min": 0.01, "max": 1.0, "default": 1.0, "step": 0.01,
        "help": "How large a step to take when the perceptron makes a mistake. 1.0 is the classic Hagan-style rule.",
    },
    "max_epochs": {
        "type": "int", "label": "Max epochs",
        "min": 1, "max": 100, "default": 20, "step": 1,
        "help": "How many full passes over the training data to allow before giving up (even if not perfectly separated yet).",
    },
}


def build_model(params: dict):
    return Perceptron(
        learning_rate=params.get("learning_rate", 1.0),
        max_epochs=params.get("max_epochs", 20),
    )


class Perceptron(BaseEstimator, ClassifierMixin):
    """One-vs-rest hardlim perceptron, trained one sample at a time."""

    def __init__(self, learning_rate: float = 1.0, max_epochs: int = 20):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_features_in_ = n_features

        self.weights_ = {}
        self.biases_ = {}
        self.history_ = {}  # per class: errors_per_epoch, w_history, b_history

        for cls in self.classes_:
            target = np.where(y == cls, 1.0, -1.0)
            w = np.zeros(n_features)
            b = 0.0
            errors_per_epoch = []
            w_history = [w.copy()]
            b_history = [b]

            for _epoch in range(self.max_epochs):
                total_errors = 0
                for i in range(n_samples):
                    net = w @ X[i] + b
                    a = 1.0 if net >= 0 else -1.0
                    e = target[i] - a
                    if e != 0.0:
                        w = w + self.learning_rate * e * X[i]
                        b = b + self.learning_rate * e
                        total_errors += 1
                errors_per_epoch.append(total_errors)
                w_history.append(w.copy())
                b_history.append(b)
                if total_errors == 0:
                    break

            self.weights_[cls] = w
            self.biases_[cls] = b
            self.history_[cls] = {
                "errors_per_epoch": errors_per_epoch,
                "w_history": np.array(w_history),
                "b_history": np.array(b_history),
            }
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([X @ self.weights_[c] + self.biases_[c] for c in self.classes_])

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Not a true probability (perceptrons don't produce one) -- a softmax
        over net inputs, used only so the UI can show relative confidence."""
        scores = self.decision_function(X)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
