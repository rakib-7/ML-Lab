"""Matplotlib figure builders. Kept pure (return a Figure) so app.py just calls st.pyplot(fig)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree


def plot_confusion_matrix(cm, class_labels):
    fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=130)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_elbow(k_values, inertias):
    fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=130)
    ax.plot(k_values, inertias, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    fig.tight_layout()
    return fig


def plot_clusters_2d(X_transformed, labels):
    """Projects features to 2D via PCA (if needed) and colors points by cluster."""
    if X_transformed.shape[1] > 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(X_transformed)
        xlabel, ylabel = "PCA component 1", "PCA component 2"
    else:
        coords = X_transformed
        xlabel, ylabel = "Feature 1", "Feature 2"

    fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=130)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Cluster Assignments")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    return fig


def plot_decision_tree(tree_model, feature_names, class_names, max_fig_width=34.0, max_fig_height=10.0,
                        highlight_nodes=None):
    """Renders the REAL sklearn tree (not an abstraction). Width scales with
    leaf count (~1.3in/leaf is what it takes for boxes not to overlap,
    measured empirically) up to a generous cap; height scales with depth
    only and stays capped, since a tall figure is what forces vertical
    scrolling -- a wide one just reads left-to-right like a diagram. Past
    the width cap, font shrinks to compensate rather than letting boxes
    overlap. If highlight_nodes is given (a list of node ids, e.g. the path
    one prediction took), those boxes get a thick colored border and
    everything else is dimmed, so the real tree itself doubles as the
    "how did this prediction happen" visual instead of a separate
    abstract flowchart.
    """
    n_leaves = tree_model.get_n_leaves()
    depth = tree_model.get_depth()

    px_per_leaf = 1.3
    natural_width = max(6.0, px_per_leaf * n_leaves)
    natural_height = max(4.0, 1.5 * (depth + 1))

    fig_width = min(max_fig_width, natural_width)
    fig_height = min(max_fig_height, natural_height)

    width_shortfall = fig_width / natural_width if natural_width > fig_width else 1.0
    fontsize = int(round(max(6, min(10, 9 * width_shortfall))))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=110)
    artists = plot_tree(
        tree_model, feature_names=feature_names, class_names=[str(c) for c in class_names],
        filled=True, rounded=True, fontsize=fontsize,
        impurity=(n_leaves <= 20), ax=ax,
    )

    if highlight_nodes:
        highlight_set = set(highlight_nodes)
        node_artists = [a for a in artists if "samples" in a.get_text()]
        leaf_id = highlight_nodes[-1]
        for node_id, artist in enumerate(node_artists):
            bbox = artist.get_bbox_patch()
            if bbox is None:
                continue
            if node_id in highlight_set:
                is_leaf = node_id == leaf_id
                bbox.set_edgecolor("#16a34a" if is_leaf else "#d97706")
                bbox.set_linewidth(3.5)
            else:
                bbox.set_alpha(0.25)

    fig.tight_layout()
    return fig, {"n_leaves": n_leaves, "depth": depth, "fontsize": fontsize}


# ---------------------------------------------------------------------------
# "How did this prediction happen?" visuals -- one per algorithm, each shows
# the mechanism (not just the final answer) for a single new query point.
# ---------------------------------------------------------------------------

def _shared_2d_projection(X_train_t, query_t, extra_points=None):
    """Fits one PCA (or passes through if already <=2D) across train points,
    the query point, and any extra points (e.g. centroids/support vectors),
    so everything lands in the same coordinate frame."""
    stack = [X_train_t, query_t.reshape(1, -1)]
    if extra_points is not None:
        stack.append(extra_points)
    combined = np.vstack(stack)
    if combined.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42).fit(combined)
        combined_2d = pca.transform(combined)
        label = "PCA component"
    else:
        combined_2d = combined
        label = "Feature"
    n_train = X_train_t.shape[0]
    train_2d = combined_2d[:n_train]
    query_2d = combined_2d[n_train]
    extra_2d = combined_2d[n_train + 1:] if extra_points is not None else None
    return train_2d, query_2d, extra_2d, label


def plot_knn_neighbors(X_train_t, train_labels, neighbor_idx, query_t):
    """Scatter of all training points (faint), the k chosen neighbors
    (highlighted with a ring), and the new query point (star)."""
    train_2d, query_2d, _, axis_label = _shared_2d_projection(X_train_t, query_t)

    fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=130)
    uniq = sorted(set(train_labels))
    cmap = plt.get_cmap("tab10")
    color_map = {c: cmap(i % 10) for i, c in enumerate(uniq)}
    colors = [color_map[c] for c in train_labels]

    ax.scatter(train_2d[:, 0], train_2d[:, 1], c=colors, s=12, alpha=0.25, linewidths=0)
    neighbor_coords = train_2d[neighbor_idx]
    neighbor_colors = [color_map[train_labels[i]] for i in neighbor_idx]
    ax.scatter(neighbor_coords[:, 0], neighbor_coords[:, 1], c=neighbor_colors, s=110,
               edgecolors="black", linewidths=1.6, label="Chosen neighbors", zorder=3)
    ax.scatter([query_2d[0]], [query_2d[1]], marker="*", s=400, c="gold",
               edgecolors="black", linewidths=1.2, label="Your new point", zorder=4)

    for c in uniq:
        ax.scatter([], [], c=[color_map[c]], label=str(c))
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.set_title("Nearest neighbors used for this prediction")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def plot_svm_support(X_train_t, train_labels, support_idx, query_t):
    """Scatter showing support vectors (the points that define the boundary)
    and where the new query point falls relative to them."""
    train_2d, query_2d, _, axis_label = _shared_2d_projection(X_train_t, query_t)

    fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=130)
    uniq = sorted(set(train_labels))
    cmap = plt.get_cmap("tab10")
    color_map = {c: cmap(i % 10) for i, c in enumerate(uniq)}
    colors = [color_map[c] for c in train_labels]

    ax.scatter(train_2d[:, 0], train_2d[:, 1], c=colors, s=12, alpha=0.2, linewidths=0)
    sv_coords = train_2d[support_idx]
    sv_colors = [color_map[train_labels[i]] for i in support_idx]
    ax.scatter(sv_coords[:, 0], sv_coords[:, 1], c=sv_colors, s=70,
               edgecolors="black", linewidths=1.2, label="Support vectors", zorder=3)
    ax.scatter([query_2d[0]], [query_2d[1]], marker="*", s=400, c="gold",
               edgecolors="black", linewidths=1.2, label="Your new point", zorder=4)

    for c in uniq:
        ax.scatter([], [], c=[color_map[c]], label=str(c))
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.set_title("Support vectors that shape the boundary")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def plot_kmeans_prediction(X_transformed, labels, centroids, query_t):
    """Cluster scatter plus centroids and the new point, so you can see
    which centroid it landed closest to."""
    train_2d, query_2d, centroid_2d, axis_label = _shared_2d_projection(
        X_transformed, query_t, extra_points=centroids
    )

    fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=130)
    scatter = ax.scatter(train_2d[:, 0], train_2d[:, 1], c=labels, cmap="tab10", s=14, alpha=0.35)
    ax.scatter(centroid_2d[:, 0], centroid_2d[:, 1], marker="X", s=220, c="black",
               label="Centroids", zorder=3)
    ax.scatter([query_2d[0]], [query_2d[1]], marker="*", s=400, c="gold",
               edgecolors="black", linewidths=1.2, label="Your new point", zorder=4)
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.set_title("Which cluster centroid is closest?")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper left", fontsize=7)
    ax.add_artist(legend1)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig

