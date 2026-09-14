"""Evaluation metrics, kept separate from training logic so they're easy to extend/test."""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score,
)


def classification_metrics(y_true, y_pred) -> dict:
    avg = "binary" if len(set(y_true)) == 2 else "weighted"
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def clustering_metrics(X_transformed, labels) -> dict:
    result = {"n_clusters_found": len(set(labels))}
    if len(set(labels)) > 1 and len(set(labels)) < len(labels):
        result["silhouette_score"] = round(silhouette_score(X_transformed, labels), 4)
    else:
        result["silhouette_score"] = None
    return result


def elbow_data(X_transformed, k_range=range(2, 11)):
    """Inertia for a range of k values, used to plot the elbow curve."""
    from sklearn.cluster import KMeans
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_transformed)
        inertias.append(km.inertia_)
    return list(k_range), inertias
