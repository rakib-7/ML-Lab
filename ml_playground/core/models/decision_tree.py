"""Decision Tree classifier: definition + UI-facing hyperparameter spec."""
from sklearn.tree import DecisionTreeClassifier

TASK = "classification"

PARAM_SPEC = {
    "max_depth": {
        "type": "int", "label": "Max depth",
        "min": 1, "max": 30, "default": 5, "step": 1,
        "help": "How many levels deep the tree can grow. Shallow = simpler, less overfitting. Deep = can capture more detail but may overfit.",
    },
    "criterion": {
        "type": "select", "label": "Split quality criterion",
        "options": ["gini", "entropy", "log_loss"], "default": "gini",
        "help": "The metric used to decide the best way to split data at each node.",
    },
    "min_samples_split": {
        "type": "int", "label": "Min samples to split a node",
        "min": 2, "max": 50, "default": 2, "step": 1,
        "help": "A node needs at least this many samples before it's allowed to split further.",
    },
}


def build_model(params: dict):
    return DecisionTreeClassifier(
        max_depth=params.get("max_depth", 5),
        criterion=params.get("criterion", "gini"),
        min_samples_split=params.get("min_samples_split", 2),
        random_state=42,
    )
