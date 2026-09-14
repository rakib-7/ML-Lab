"""K-Means clustering: definition + UI-facing hyperparameter spec."""
from sklearn.cluster import KMeans

TASK = "clustering"

PARAM_SPEC = {
    "n_clusters": {
        "type": "int", "label": "Number of clusters (k)",
        "min": 2, "max": 15, "default": 3, "step": 1,
        "help": "How many groups to split the data into. Use the elbow plot to help choose this.",
    },
    "n_init": {
        "type": "int", "label": "Number of initializations",
        "min": 1, "max": 20, "default": 10, "step": 1,
        "help": "How many times to run with different starting centroids, keeping the best result. Higher = more stable but slower.",
    },
}


def build_model(params: dict):
    return KMeans(
        n_clusters=params.get("n_clusters", 3),
        n_init=params.get("n_init", 10),
        random_state=42,
    )
