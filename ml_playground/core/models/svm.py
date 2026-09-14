"""Support Vector Machine classifier: definition + UI-facing hyperparameter spec."""
from sklearn.svm import SVC

TASK = "classification"

PARAM_SPEC = {
    "kernel": {
        "type": "select", "label": "Kernel",
        "options": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf",
        "help": "The function used to separate classes. 'rbf' handles non-linear boundaries well; 'linear' is simplest and fastest.",
    },
    "C": {
        "type": "float", "label": "C (regularization)",
        "min": 0.01, "max": 100.0, "default": 1.0, "step": 0.01,
        "help": "Lower C = wider margin, more tolerant of misclassified points (less overfitting). Higher C = fits training data harder.",
    },
    "gamma": {
        "type": "select", "label": "Gamma",
        "options": ["scale", "auto"], "default": "scale",
        "help": "Controls how far the influence of a single training point reaches (only matters for rbf/poly/sigmoid kernels).",
    },
    "degree": {
        "type": "int", "label": "Polynomial degree",
        "min": 2, "max": 6, "default": 3, "step": 1,
        "help": "Only used when kernel = 'poly'.",
    },
}


def build_model(params: dict):
    return SVC(
        kernel=params.get("kernel", "rbf"),
        C=params.get("C", 1.0),
        gamma=params.get("gamma", "scale"),
        degree=params.get("degree", 3),
        probability=True,
    )
