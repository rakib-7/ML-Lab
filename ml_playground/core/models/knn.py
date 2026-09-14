"""K-Nearest Neighbors classifier: definition + UI-facing hyperparameter spec."""
from sklearn.neighbors import KNeighborsClassifier

TASK = "classification"

PARAM_SPEC = {
    "n_neighbors": {
        "type": "int", "label": "Number of neighbors (k)",
        "min": 1, "max": 50, "default": 5, "step": 1,
        "help": "How many nearby points vote on the prediction. Small k = sensitive to noise, large k = smoother but can blur boundaries.",
    },
    "weights": {
        "type": "select", "label": "Weight function",
        "options": ["uniform", "distance"], "default": "uniform",
        "help": "'uniform' = all neighbors count equally. 'distance' = closer neighbors count more.",
    },
    "metric": {
        "type": "select", "label": "Distance metric",
        "options": ["minkowski", "euclidean", "manhattan"], "default": "minkowski",
        "help": "How distance between points is measured.",
    },
}


def build_model(params: dict):
    return KNeighborsClassifier(
        n_neighbors=params.get("n_neighbors", 5),
        weights=params.get("weights", "uniform"),
        metric=params.get("metric", "minkowski"),
    )
