"""
Plotly animation builders -- the "watch it happen" counterpart to viz.py's
static plots. Each function takes the same data explain.py already computes
and returns a go.Figure with animation frames, so app.py just calls
st.plotly_chart(fig).

Kept separate from viz.py (matplotlib, static) because animation needs
Plotly's frame/slider machinery, which matplotlib doesn't have natively.
Design goals, in priority order:
  1. Beginner-friendly: a play button + slider, short on-frame captions,
     nothing requires reading code or math to follow.
  2. Truthful: every animation reflects the *actual* numbers from explain.py
     (real distances, real order, real weights) -- never a generic canned
     animation, since that would mislead rather than teach.
  3. Fast to render: frame counts are capped (<=30) and every frame reuses
     one PCA projection, so this stays snappy even on the 12k-row samples.
"""
import numpy as np
from sklearn.decomposition import PCA

TEMPLATE = "plotly_white"
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _project_2d(X_train_t, query_t, extra_points=None):
    """Same idea as viz._shared_2d_projection: one PCA fit across train
    points, the query point, and any extra points (centroids/support
    vectors), so everything animates in one consistent coordinate frame."""
    stack = [X_train_t, query_t.reshape(1, -1)]
    if extra_points is not None:
        stack.append(extra_points)
    combined = np.vstack(stack)
    if combined.shape[1] > 2:
        combined_2d = PCA(n_components=2, random_state=42).fit_transform(combined)
        axis_label = "PCA component"
    else:
        combined_2d = combined
        axis_label = "Feature"
    n_train = X_train_t.shape[0]
    train_2d = combined_2d[:n_train]
    query_2d = combined_2d[n_train]
    extra_2d = combined_2d[n_train + 1:] if extra_points is not None else None
    return train_2d, query_2d, extra_2d, axis_label


def _color_map(labels):
    uniq = sorted(set(labels))
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(uniq)}


def _base_layout(fig, title, axis_label, height=460):
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        xaxis_title=f"{axis_label} 1",
        yaxis_title=f"{axis_label} 2",
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _play_controls(frame_duration=700):
    return dict(
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=-0.12, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_duration, redraw=True),
                                       fromcurrent=True, transition=dict(duration=200))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.0, y=-0.02, len=1.0,
            currentvalue=dict(prefix="Step: ", font=dict(size=12)),
            steps=[],  # filled in by caller
        )],
    )


def _attach_slider_steps(fig, frame_labels, frame_duration=700):
    fig.layout.sliders[0]["steps"] = [
        dict(label=lbl, method="animate",
             args=[[str(i)], dict(frame=dict(duration=frame_duration, redraw=True),
                                   mode="immediate", transition=dict(duration=200))])
        for i, lbl in enumerate(frame_labels)
    ]


# ---------------------------------------------------------------------------
# KNN: reveal the k neighbors one at a time, closest first, with a running
# vote tally shown in the frame title -- mirrors exactly what model.kneighbors
# returned (already sorted by distance).
# ---------------------------------------------------------------------------

def animate_knn(X_train_t, train_labels, neighbor_idx, neighbor_labels, query_t):
    train_2d, query_2d, _, axis_label = _project_2d(X_train_t, query_t)
    cmap = _color_map(train_labels)
    all_colors = [cmap[c] for c in train_labels]

    import plotly.graph_objects as go

    base_traces = [
        go.Scatter(x=train_2d[:, 0], y=train_2d[:, 1], mode="markers",
                   marker=dict(color=all_colors, size=6, opacity=0.22),
                   name="Training points", hoverinfo="skip", showlegend=False),
        go.Scatter(x=[query_2d[0]], y=[query_2d[1]], mode="markers",
                   marker=dict(symbol="star", size=22, color="gold",
                               line=dict(color="black", width=1.5)),
                   name="New point"),
        go.Scatter(x=[], y=[], mode="markers", marker=dict(size=16, color=[],
                   line=dict(color="black", width=2)), name="Neighbors found so far"),
    ]
    for c in sorted(set(train_labels)):
        base_traces.append(go.Scatter(x=[None], y=[None], mode="markers",
                                       marker=dict(color=cmap[c], size=10),
                                       name=str(c)))

    frames = []
    frame_labels = []
    running = {}
    for step in range(1, len(neighbor_idx) + 1):
        idxs = neighbor_idx[:step]
        coords = train_2d[idxs]
        colors = [cmap[train_labels[i]] for i in idxs]
        lbl = neighbor_labels[step - 1]
        running[lbl] = running.get(lbl, 0) + 1
        leader = max(running, key=running.get)
        tally = ", ".join(f"{k}: {v}" for k, v in running.items())
        frame_labels.append(f"{step}")
        frames.append(go.Frame(
            name=str(step - 1),
            data=[
                go.Scatter(x=train_2d[:, 0], y=train_2d[:, 1], mode="markers",
                           marker=dict(color=all_colors, size=6, opacity=0.22)),
                go.Scatter(x=[query_2d[0]], y=[query_2d[1]], mode="markers",
                           marker=dict(symbol="star", size=22, color="gold",
                                       line=dict(color="black", width=1.5))),
                go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers",
                           marker=dict(size=16, color=colors, line=dict(color="black", width=2))),
            ],
            traces=[0, 1, 2],
            layout=go.Layout(title=f"Neighbor {step}/{len(neighbor_idx)} added (closest so far) — "
                                    f"running vote: {tally} → leading: {leader}"),
        ))

    fig = go.Figure(data=base_traces, frames=frames)
    _base_layout(fig, f"Step 1/{len(neighbor_idx)} — finding the closest neighbors, one at a time",
                 axis_label)
    fig.update_layout(**_play_controls())
    _attach_slider_steps(fig, frame_labels)
    return fig


# ---------------------------------------------------------------------------
# SVM: fade in the support vectors that define the boundary, then drop the
# new point in on the final frame -- shows *why* only some points matter.
# ---------------------------------------------------------------------------

def animate_svm(X_train_t, train_labels, support_idx, query_t, n_frames=14):
    train_2d, query_2d, _, axis_label = _project_2d(X_train_t, query_t)
    cmap = _color_map(train_labels)
    all_colors = [cmap[c] for c in train_labels]

    import plotly.graph_objects as go

    order = list(support_idx)
    n_frames = min(n_frames, max(2, len(order)))
    chunks = np.array_split(np.array(order), n_frames)

    base_traces = [
        go.Scatter(x=train_2d[:, 0], y=train_2d[:, 1], mode="markers",
                   marker=dict(color=all_colors, size=6, opacity=0.18),
                   name="Training points", hoverinfo="skip", showlegend=False),
        go.Scatter(x=[], y=[], mode="markers",
                   marker=dict(size=13, color=[], line=dict(color="black", width=1.5)),
                   name="Support vectors (define the boundary)"),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(symbol="star", size=22, color="gold",
                               line=dict(color="black", width=1.5)), name="New point"),
    ]
    for c in sorted(set(train_labels)):
        base_traces.append(go.Scatter(x=[None], y=[None], mode="markers",
                                       marker=dict(color=cmap[c], size=10), name=str(c)))

    frames = []
    frame_labels = []
    cumulative = []
    for i, chunk in enumerate(chunks):
        cumulative.extend(chunk.tolist())
        coords = train_2d[cumulative]
        colors = [cmap[train_labels[j]] for j in cumulative]
        is_last = i == len(chunks) - 1
        star_x = [query_2d[0]] if is_last else [None]
        star_y = [query_2d[1]] if is_last else [None]
        title = (f"All {len(order)} support vectors found — dropping in your new point"
                 if is_last else
                 f"Finding support vectors: {len(cumulative)}/{len(order)} so far")
        frame_labels.append(str(i + 1))
        frames.append(go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=train_2d[:, 0], y=train_2d[:, 1], mode="markers",
                           marker=dict(color=all_colors, size=6, opacity=0.18)),
                go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers",
                           marker=dict(size=13, color=colors, line=dict(color="black", width=1.5))),
                go.Scatter(x=star_x, y=star_y, mode="markers",
                           marker=dict(symbol="star", size=22, color="gold",
                                       line=dict(color="black", width=1.5))),
            ],
            traces=[0, 1, 2],
            layout=go.Layout(title=title),
        ))

    fig = go.Figure(data=base_traces, frames=frames)
    _base_layout(fig, "Support vectors are the only points that shape the boundary", axis_label)
    fig.update_layout(**_play_controls(frame_duration=350))
    _attach_slider_steps(fig, frame_labels, frame_duration=350)
    return fig


# ---------------------------------------------------------------------------
# K-Means: animate the new point sliding from its raw position toward the
# centroid it's ultimately assigned to, with a faint line to every other
# centroid showing it was considered and ruled out.
# ---------------------------------------------------------------------------

def animate_kmeans(X_transformed, labels, centroids, query_t, predicted_cluster, n_frames=16):
    train_2d, query_2d, centroid_2d, axis_label = _project_2d(
        X_transformed, query_t, extra_points=centroids
    )

    import plotly.graph_objects as go

    winner_2d = centroid_2d[predicted_cluster]
    path_x = np.linspace(query_2d[0], winner_2d[0], n_frames)
    path_y = np.linspace(query_2d[1], winner_2d[1], n_frames)

    cluster_colors = [PALETTE[int(l) % len(PALETTE)] for l in labels]

    base_traces = [
        go.Scatter(x=train_2d[:, 0], y=train_2d[:, 1], mode="markers",
                   marker=dict(color=cluster_colors, size=6, opacity=0.28),
                   name="Data points (colored by cluster)", hoverinfo="skip"),
        go.Scatter(x=centroid_2d[:, 0], y=centroid_2d[:, 1], mode="markers+text",
                   marker=dict(symbol="x", size=16, color="black"),
                   text=[f"C{i}" for i in range(len(centroid_2d))], textposition="top center",
                   name="Centroids"),
        go.Scatter(x=[query_2d[0]], y=[query_2d[1]], mode="markers",
                   marker=dict(symbol="star", size=22, color="gold",
                               line=dict(color="black", width=1.5)), name="Your new point"),
        go.Scatter(x=[query_2d[0], winner_2d[0]], y=[query_2d[1], winner_2d[1]],
                   mode="lines", line=dict(color="gray", width=1, dash="dot"),
                   opacity=0, showlegend=False, hoverinfo="skip"),
    ]

    frames = []
    frame_labels = []
    for i in range(n_frames):
        is_last = i == n_frames - 1
        title = (f"Landed in Cluster {predicted_cluster} (closest centroid)"
                 if is_last else "Moving toward its nearest centroid...")
        frame_labels.append(str(i + 1))
        frames.append(go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=[path_x[i]], y=[path_y[i]], mode="markers",
                           marker=dict(symbol="star", size=22, color="gold",
                                       line=dict(color="black", width=1.5))),
                go.Scatter(x=path_x[:i + 1], y=path_y[:i + 1], mode="lines",
                           line=dict(color="#b45309", width=2, dash="dot"), opacity=0.8,
                           hoverinfo="skip"),
            ],
            traces=[2, 3],
            layout=go.Layout(title=title),
        ))

    fig = go.Figure(data=base_traces, frames=frames)
    _base_layout(fig, "Watching the new point travel to its closest centroid", axis_label)
    fig.update_layout(**_play_controls(frame_duration=180))
    _attach_slider_steps(fig, frame_labels, frame_duration=180)
    return fig


# ---------------------------------------------------------------------------
# Decision Tree: animate the REAL tree -- same node text, same layout as
# viz.plot_decision_tree -- highlighting the actual path this prediction
# took, one split at a time, ending on the real leaf box. Node positions and
# text are extracted from sklearn's own plot_tree layout (via a throwaway,
# never-rendered matplotlib call) so the shape is guaranteed to match the
# real tree exactly, not a hand-drawn abstraction.
# ---------------------------------------------------------------------------

def _extract_tree_layout(tree_model, feature_names, class_names):
    """Runs sklearn's plot_tree on a scratch (never-shown) figure purely to
    read back each node's real (x, y) position and label text, plus the
    real parent/child edges from tree_. Returns them in tree_ node-id order."""
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    scratch_fig, scratch_ax = plt.subplots()
    artists = plot_tree(
        tree_model, feature_names=feature_names, class_names=[str(c) for c in class_names],
        filled=True, ax=scratch_ax,
    )
    plt.close(scratch_fig)

    node_artists = [a for a in artists if "samples" in a.get_text()]
    positions = [a.get_position() for a in node_artists]
    texts = [a.get_text() for a in node_artists]

    t = tree_model.tree_
    edges = []
    for node_id in range(t.node_count):
        left, right = t.children_left[node_id], t.children_right[node_id]
        if left != -1:
            edges.append((node_id, left))
        if right != -1:
            edges.append((node_id, right))
    return positions, texts, edges


def animate_decision_tree_path(tree_model, feature_names, class_names, visited_nodes, predicted_label,
                                max_height=560, max_width=900):
    """Plays the real tree's visited path one node at a time: dims
    everything else, highlights nodes already reached, and finishes on the
    leaf in green. Layout and text come straight from sklearn's own
    plot_tree, so what's animated is the same tree shown elsewhere in the
    app -- just with the path lit up."""
    import plotly.graph_objects as go

    positions, texts, edges = _extract_tree_layout(tree_model, feature_names, class_names)
    n_nodes = len(positions)
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])

    # Short label per node: first line of plot_tree's text (the split
    # condition, or "class = X" for a leaf) plus the sample count --
    # trimmed because animated frames redraw fast and don't need the full
    # gini/value block to stay readable.
    def _short_label(text):
        lines = text.split("\n")
        first = lines[0]
        samples_line = next((l for l in lines if l.startswith("samples")), "")
        class_line = next((l for l in lines if l.startswith("class")), "")
        parts = [first]
        if samples_line:
            parts.append(samples_line)
        if class_line:
            parts.append(class_line)
        return "<br>".join(parts)

    labels = [_short_label(t) for t in texts]
    leaf_id = visited_nodes[-1]

    n_leaves = tree_model.get_n_leaves()
    depth = tree_model.get_depth()
    fig_width = min(max_width, max(420, 70 * n_leaves))
    fig_height = min(max_height, max(280, 90 * (depth + 1)))
    font_size = 10 if n_leaves <= 15 else (8 if n_leaves <= 31 else 7)

    edge_x, edge_y = [], []
    for a, b in edges:
        edge_x += [xs[a], xs[b], None]
        edge_y += [ys[a], ys[b], None]

    base_traces = [
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#cbd5e1", width=1.5),
                   hoverinfo="skip", showlegend=False),
        go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=1, color="rgba(0,0,0,0)"),
                   hoverinfo="skip", showlegend=False),  # invisible, keeps axis range stable
    ]

    frames = []
    frame_labels = []
    for step in range(1, len(visited_nodes) + 1):
        shown_path = visited_nodes[:step]
        shown_set = set(shown_path)
        colors, sizes, border_colors, border_widths, opac = [], [], [], [], []
        for node_id in range(n_nodes):
            is_shown = node_id in shown_set
            is_leaf_now = is_shown and node_id == shown_path[-1] and node_id == leaf_id
            if is_shown:
                colors.append("#bbf7d0" if is_leaf_now else "#fde68a")
                border_colors.append("#16a34a" if is_leaf_now else "#d97706")
                border_widths.append(3)
                sizes.append(34)
                opac.append(1.0)
            else:
                colors.append("#dbeafe")
                border_colors.append("#93c5fd")
                border_widths.append(1)
                sizes.append(24)
                opac.append(0.35)

        title = (f"Reached the prediction: {predicted_label}" if step == len(visited_nodes)
                 else f"Checking split {step}/{len(visited_nodes) - 1}")
        frame_labels.append(str(step))
        frames.append(go.Frame(
            name=str(step - 1),
            data=[
                go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#cbd5e1", width=1.5),
                           hoverinfo="skip"),
                go.Scatter(x=xs, y=ys, mode="markers+text", text=labels, textposition="middle center",
                           textfont=dict(size=font_size), hovertext=texts, hoverinfo="text",
                           marker=dict(size=sizes, color=colors, opacity=opac,
                                       line=dict(color=border_colors, width=border_widths))),
            ],
            traces=[0, 1],
            layout=go.Layout(title=dict(text=title, font=dict(size=13))),
        ))

    fig = go.Figure(data=base_traces, frames=frames)
    fig.data[1].update(mode="markers+text", text=labels, textposition="middle center",
                        textfont=dict(size=font_size), hovertext=texts, hoverinfo="text")
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text=f"Step 1/{len(visited_nodes) - 1} — following the real tree's path",
                    font=dict(size=13)),
        height=fig_height, width=fig_width,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    fig.update_layout(**_play_controls(frame_duration=900))
    _attach_slider_steps(fig, frame_labels, frame_duration=900)
    return fig


# ---------------------------------------------------------------------------
# Perceptron: grow each class's weighted-sum bar from 0 to its final score
# simultaneously, highlighting the winner once the growth finishes.
# ---------------------------------------------------------------------------

def animate_perceptron(breakdown: dict, n_frames=16):
    import plotly.graph_objects as go

    labels = list(breakdown.keys())
    finals = np.array([breakdown[l] for l in labels], dtype=float)
    winner = labels[int(np.argmax(finals))]
    progress = np.linspace(0, 1, n_frames)

    frames = []
    frame_labels = []
    for i, p in enumerate(progress):
        vals = finals * p
        is_last = i == n_frames - 1
        colors = ["#16a34a" if (is_last and l == winner) else "#3b82f6" for l in labels]
        title = (f"Highest score wins: {winner}" if is_last else "Computing weighted sums...")
        frame_labels.append(str(i + 1))
        frames.append(go.Frame(
            name=str(i),
            data=[go.Bar(x=labels, y=vals, marker_color=colors,
                          text=[f"{v:.2f}" for v in vals], textposition="outside")],
            layout=go.Layout(title=title),
        ))

    fig = go.Figure(data=[go.Bar(x=labels, y=finals * 0, marker_color="#3b82f6")], frames=frames)
    fig.update_layout(
        template=TEMPLATE, title="Step 1 — each class's score starts at 0",
        height=420, margin=dict(l=10, r=10, t=60, b=10),
        yaxis_title="Weighted-sum score", showlegend=False,
    )
    fig.update_layout(**_play_controls(frame_duration=120))
    _attach_slider_steps(fig, frame_labels, frame_duration=120)
    return fig
