"""
For each algorithm, computes the intermediate numbers a human would need
to see *why* the model predicted what it did -- not just the final label.
Kept separate from predictor.py (which only returns the label) so the
"just give me an answer" path stays fast and simple.
"""
import numpy as np
import pandas as pd

from utils.naming import clean_feature_names


def _to_dense(arr):
    return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)


def explain_knn(trained, feature_cols, raw_values):
    pipeline = trained["pipeline"]
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    X_train, y_train = trained["extra"]["X_train"], trained["extra"]["y_train"]

    query_df = pd.DataFrame([raw_values])[feature_cols]
    query_t = _to_dense(prep.transform(query_df))[0]
    X_train_t = _to_dense(prep.transform(X_train))

    distances, indices = model.kneighbors(query_t.reshape(1, -1), n_neighbors=model.n_neighbors)
    neighbor_idx = indices[0]
    neighbor_labels_encoded = y_train[neighbor_idx]

    enc = trained["target_encoder"]
    train_labels_all = enc.inverse_transform(y_train) if enc else y_train
    neighbor_labels = enc.inverse_transform(neighbor_labels_encoded) if enc else neighbor_labels_encoded

    vote_counts = pd.Series(neighbor_labels).value_counts()
    return {
        "distances": distances[0], "neighbor_labels": neighbor_labels, "vote_counts": vote_counts,
        "X_train_t": X_train_t, "train_labels_all": train_labels_all,
        "neighbor_idx": neighbor_idx, "query_t": query_t,
    }


def explain_svm(trained, feature_cols, raw_values):
    pipeline = trained["pipeline"]
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    X_train, y_train = trained["extra"]["X_train"], trained["extra"]["y_train"]

    query_df = pd.DataFrame([raw_values])[feature_cols]
    query_t = _to_dense(prep.transform(query_df))[0]
    X_train_t = _to_dense(prep.transform(X_train))

    decision_vals = model.decision_function(query_t.reshape(1, -1))[0]
    classes = model.classes_
    enc = trained["target_encoder"]
    class_labels = enc.inverse_transform(classes) if enc else classes
    train_labels_all = enc.inverse_transform(y_train) if enc else y_train

    scores = dict(zip(class_labels, np.atleast_1d(decision_vals) if len(classes) > 2 else [decision_vals, -decision_vals]))
    return {
        "scores": scores, "support_idx": model.support_,
        "X_train_t": X_train_t, "train_labels_all": train_labels_all, "query_t": query_t,
    }


def explain_decision_tree(trained, feature_cols, raw_values):
    pipeline = trained["pipeline"]
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    query_df = pd.DataFrame([raw_values])[feature_cols]
    query_t = _to_dense(prep.transform(query_df))
    feat_names = prep.get_feature_names_out()

    node_indicator = model.decision_path(query_t)
    leaf_id = model.apply(query_t)[0]
    visited_nodes = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    tree = model.tree_
    feat_names_clean = clean_feature_names(feat_names)
    steps = []
    for node_id in visited_nodes:
        if node_id == leaf_id:
            continue
        feat_idx = tree.feature[node_id]
        feature = feat_names_clean[feat_idx]
        threshold = tree.threshold[node_id]
        value = query_t[0, feat_idx]
        if value <= threshold:
            steps.append(f"{feature} = {value:.2f}  ≤  {threshold:.2f}  →  go left")
        else:
            steps.append(f"{feature} = {value:.2f}  >  {threshold:.2f}  →  go right")
    return {"steps": steps, "leaf_id": leaf_id, "visited_nodes": visited_nodes.tolist()}


def explain_perceptron(trained, feature_cols, raw_values):
    pipeline = trained["pipeline"]
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    query_df = pd.DataFrame([raw_values])[feature_cols]
    query_t = _to_dense(prep.transform(query_df))[0]

    enc = trained["target_encoder"]
    breakdown = {}
    for cls in model.classes_:
        w, b = model.weights_[cls], model.biases_[cls]
        net = float(w @ query_t + b)
        label = enc.inverse_transform([cls])[0] if enc else cls
        breakdown[label] = net
    return {"breakdown": breakdown}


def explain_kmeans(trained, raw_values):
    pipeline = trained["pipeline"]
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    query_df = pd.DataFrame([raw_values])
    query_t = _to_dense(prep.transform(query_df))[0]
    centroids = model.cluster_centers_
    distances = np.linalg.norm(centroids - query_t, axis=1)
    return {
        "distances": {f"Cluster {i}": float(d) for i, d in enumerate(distances)},
        "query_t": query_t, "centroids": centroids,
        "predicted_cluster": int(np.argmin(distances)),
    }
