"""
ML Playground - a beginner-friendly interface for KNN, SVM, Decision Tree,
Perceptron, and K-Means. Upload a file (or use a bundled 12,000-row sample),
train a model, and see exactly how each prediction was made.

Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from core.data_loader import load_file, DataLoadError, infer_column_types
from core.models import REGISTRY, UNSUPERVISED
from core.trainer import train
from core.predictor import predict_single
from core.explanations import ALGO_INFO
from core import explain
from utils import viz, animate
from utils.metrics import elbow_data
from utils.naming import clean_feature_names

st.set_page_config(page_title="Machine Learning Lab", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
    .algo-badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; margin-left: 8px;
    }
    .badge-supervised { background: #dbeafe; color: #1d4ed8; }
    .badge-unsupervised { background: #fef3c7; color: #b45309; }
    .step-title { font-size: 1.35rem; font-weight: 700; margin-top: 0.4rem; }
    .one-liner { color: #4b5563; font-size: 0.95rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    div[data-testid="stPlotlyChart"], div[data-testid="stImage"] {
        border: 1px solid #e5e7eb; border-radius: 10px; padding: 6px;
    }
</style>
""", unsafe_allow_html=True)

SAMPLE_DATASETS = {
    "classification_sample": {
        "path": "sample_data/classification_sample.csv",
        "label": "🛍️ Customer Segments (12,000 rows) — for KNN / SVM / Decision Tree / Perceptron",
        "default_target": "customer_segment",
    },
    "fruit_sample": {
        "path": "sample_data/fruit_sample.csv",
        "label": "🍎 Fruit Classifier (12,000 rows) — 4 measurements, 6 fruits — great for Perceptron",
        "default_target": "label",
    },
    "clustering_sample": {
        "path": "sample_data/clustering_sample.csv",
        "label": "🎯 Shopper Behavior (12,000 rows) — no labels, for K-Means",
        "default_target": None,
    },
}

for key, default in {
    "df": None, "trained": None, "algorithm": None,
    "feature_cols": None, "target_col": None, "picked_algo": None,
    "last_prediction": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------- header
st.title("⚙️ Machine Learning Lab")
st.caption("Pick an algorithm, feed it data, and watch exactly how it makes each prediction.")

# ---------------------------------------------------------------- Step 1: pick algorithm
st.markdown('<div class="step-title">Step 1 · Choose an algorithm</div>', unsafe_allow_html=True)
algo_names = list(REGISTRY.keys())
cols = st.columns(len(algo_names))
for i, name in enumerate(algo_names):
    info = ALGO_INFO[name]
    with cols[i]:
        with st.container(border=True):
            st.markdown(f"### {info['icon']}")
            st.markdown(f"**{name}**")
            st.caption(info["one_liner"])
            if st.button("Select", key=f"pick_{name}", use_container_width=True,
                         type="primary" if st.session_state.picked_algo == name else "secondary"):
                st.session_state.picked_algo = name
                st.session_state.trained = None  # switching algos invalidates any prior training
                st.session_state.last_prediction = None

algorithm = st.session_state.picked_algo
if algorithm is None:
    st.info("👆 Pick an algorithm above to get started.")
    st.stop()

is_unsupervised = algorithm in UNSUPERVISED
info = ALGO_INFO[algorithm]
badge_class = "badge-unsupervised" if is_unsupervised else "badge-supervised"
badge_text = "Unsupervised (no target needed)" if is_unsupervised else "Supervised (needs a target column)"
st.markdown(f"**You picked: {info['icon']} {algorithm}** "
            f'<span class="algo-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

with st.expander("ℹ️ How does this algorithm work?", expanded=False):
    st.markdown(f"**How it learns:** {info['how_it_learns']}")
    st.markdown(f"**How it predicts:** {info['how_it_predicts']}")

st.divider()

# ---------------------------------------------------------------- Step 2: get data
st.markdown('<div class="step-title">Step 2 · Provide training data</div>', unsafe_allow_html=True)
data_source = st.radio("Where should the data come from?",
                        ["Use a bundled sample dataset", "Upload my own file"], horizontal=True)

sample_choice = None
if data_source == "Upload my own file":
    uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            st.session_state.df = load_file(uploaded)
            st.success(f"Loaded {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns.")
        except DataLoadError as e:
            st.error(str(e))
            st.session_state.df = None
else:
    suggested_key = "clustering_sample" if is_unsupervised else (
        "fruit_sample" if algorithm == "Perceptron" else "classification_sample"
    )
    sample_keys = list(SAMPLE_DATASETS.keys())
    sample_choice = st.selectbox(
        "Sample dataset", sample_keys,
        index=sample_keys.index(suggested_key),
        format_func=lambda k: SAMPLE_DATASETS[k]["label"],
    )
    st.session_state.df = pd.read_csv(SAMPLE_DATASETS[sample_choice]["path"])
    st.caption(f"Loaded **{SAMPLE_DATASETS[sample_choice]['label']}** "
               f"({len(st.session_state.df):,} rows).")

df = st.session_state.df
if df is None:
    st.stop()

with st.expander("👀 Preview the data (first 20 rows)"):
    st.dataframe(df.head(20), use_container_width=True)

st.divider()

# ---------------------------------------------------------------- Step 3: columns
st.markdown('<div class="step-title">Step 3 · Choose columns</div>', unsafe_allow_html=True)
all_cols = df.columns.tolist()

if is_unsupervised:
    feature_cols = st.multiselect("Feature columns (used to find groups)", all_cols, default=all_cols)
    target_col = None
else:
    default_target = SAMPLE_DATASETS.get(sample_choice, {}).get("default_target") if sample_choice else None
    default_idx = all_cols.index(default_target) if default_target in all_cols else len(all_cols) - 1
    target_col = st.selectbox("Target column (what you want to predict)", all_cols, index=default_idx)
    feature_default = [c for c in all_cols if c != target_col]
    feature_cols = st.multiselect("Feature columns (inputs)", feature_default, default=feature_default)

st.divider()

# ---------------------------------------------------------------- Step 4: hyperparameters
st.markdown('<div class="step-title">Step 4 · Set hyperparameters</div>', unsafe_allow_html=True)
model_module = REGISTRY[algorithm]
params = {}
hp_cols = st.columns(min(3, len(model_module.PARAM_SPEC)))
for i, (key, spec) in enumerate(model_module.PARAM_SPEC.items()):
    with hp_cols[i % len(hp_cols)]:
        if spec["type"] in ("int", "float"):
            params[key] = st.slider(spec["label"], spec["min"], spec["max"], spec["default"], spec["step"], help=spec.get("help"))
        elif spec["type"] == "select":
            params[key] = st.selectbox(spec["label"], spec["options"], index=spec["options"].index(spec["default"]), help=spec.get("help"))

st.divider()

# ---------------------------------------------------------------- Step 5: train
st.markdown('<div class="step-title">Step 5 · Train</div>', unsafe_allow_html=True)
can_train = len(feature_cols) > 0 and (is_unsupervised or target_col is not None)
if not can_train:
    st.warning("Select at least one feature column" + ("" if is_unsupervised else " and a target column") + ".")

if st.button(f"🚀 Train {algorithm}", disabled=not can_train, type="primary"):
    X = df[feature_cols].copy()
    y = None if is_unsupervised else df[target_col].copy()
    with st.spinner(f"Training on {len(df):,} rows..."):
        result = train(algorithm, params, X, y)
    st.session_state.trained = result
    st.session_state.algorithm = algorithm
    st.session_state.feature_cols = feature_cols
    st.session_state.target_col = target_col
    st.success("Model trained.")

trained = st.session_state.trained
if trained is None or st.session_state.algorithm != algorithm:
    st.stop()

# ---------------------------------------------------------------- Step 6: results
st.divider()
st.markdown('<div class="step-title">Step 6 · Results</div>', unsafe_allow_html=True)
m = trained["metrics"]

if trained["task"] == "classification":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", m["accuracy"])
    c2.metric("Precision", m["precision"])
    c3.metric("Recall", m["recall"])
    c4.metric("F1 score", m["f1"])
    st.caption("Computed on a held-out 20% test split the model never saw during training.")

    plot_col, extra_col = st.columns(2)
    with plot_col:
        fig = viz.plot_confusion_matrix(np.array(m["confusion_matrix"]), trained["extra"]["class_labels"])
        st.pyplot(fig, use_container_width=False)

    if algorithm == "Perceptron":
        with extra_col:
            st.write("**Training curve** (misclassifications per class, per epoch):")
            hist_df = pd.DataFrame({
                str(cls): pd.Series(h["errors_per_epoch"])
                for cls, h in trained["pipeline"].named_steps["model"].history_.items()
            })
            st.line_chart(hist_df)
        with st.expander("See per-class weight evolution and learned formulas"):
            model_obj = trained["pipeline"].named_steps["model"]
            try:
                feat_names = clean_feature_names(trained["pipeline"].named_steps["prep"].get_feature_names_out())
            except Exception:
                feat_names = feature_cols
            for cls in model_obj.classes_:
                h = model_obj.history_[cls]
                label = trained["target_encoder"].inverse_transform([cls])[0] if trained["target_encoder"] else cls
                converged = h["errors_per_epoch"][-1] == 0
                st.markdown(f"**{label}** — "
                            + ("✅ converged in " if converged else "⚠️ stopped after ")
                            + f"{len(h['errors_per_epoch'])} epoch(s)")
                w_df = pd.DataFrame(h["w_history"], columns=list(feat_names))
                w_df["bias"] = h["b_history"]
                st.line_chart(w_df)
                terms = " + ".join(
                    f"{w:.2f}\\cdot\\text{{{fn.replace('_', chr(92)+'_')}}}"
                    for w, fn in zip(model_obj.weights_[cls], feat_names)
                )
                st.latex(f"\\text{{score}} = {terms} + ({model_obj.biases_[cls]:.2f})")

    if algorithm == "Decision Tree":
        st.write("**The full tree** (this is the real, trained model — every split it can make):")
        try:
            feat_names = clean_feature_names(trained["pipeline"].named_steps["prep"].get_feature_names_out())
        except Exception:
            feat_names = feature_cols
        tree_model = trained["pipeline"].named_steps["model"]
        fig2, tree_info = viz.plot_decision_tree(tree_model, feat_names, trained["extra"]["class_labels"])
        st.pyplot(fig2, use_container_width=False)
        if tree_info["n_leaves"] > 20 or tree_info["depth"] > 5:
            st.caption("This tree is large, so labels use a smaller font to keep every box readable. "
                       "Lower Max Depth or raise Min Samples to Split in the sidebar for a simpler tree.")



else:  # clustering
    c1, c2 = st.columns(2)
    c1.metric("Clusters found", m["n_clusters_found"])
    c2.metric("Silhouette score", m["silhouette_score"] if m["silhouette_score"] is not None else "n/a")
    st.caption("Silhouette score ranges from -1 to 1 — higher means better-separated, more distinct clusters.")

    plot_col, elbow_col = st.columns(2)
    with plot_col:
        fig = viz.plot_clusters_2d(trained["extra"]["X_transformed"], trained["extra"]["labels"])
        st.pyplot(fig, use_container_width=False)
    with elbow_col:
        if st.checkbox("Show elbow plot (helps pick k)"):
            k_vals, inertias = elbow_data(trained["extra"]["X_transformed"])
            st.pyplot(viz.plot_elbow(k_vals, inertias), use_container_width=False)

# ---------------------------------------------------------------- Step 7: predict
st.divider()
st.markdown('<div class="step-title">Step 7 · Try a prediction</div>', unsafe_allow_html=True)
st.caption("Enter values below, then see not just the answer but exactly how the model got there.")

col_types = infer_column_types(df[st.session_state.feature_cols])
input_cols = st.columns(3)
raw_values = {}
for i, col in enumerate(st.session_state.feature_cols):
    with input_cols[i % 3]:
        if col in col_types["numeric"]:
            default_val = float(df[col].median())
            raw_values[col] = st.number_input(col, value=default_val)
        else:
            options = sorted(df[col].dropna().unique().tolist())
            raw_values[col] = st.selectbox(col, options)

if st.button("🔮 Predict", type="primary"):
    st.session_state.last_prediction = {"raw_values": dict(raw_values), "algorithm": algorithm}

if st.session_state.get("last_prediction") and st.session_state.last_prediction["algorithm"] == algorithm:
    raw_values = st.session_state.last_prediction["raw_values"]
    feature_cols_saved = st.session_state.feature_cols

    view_mode = st.radio(
        "How do you want to see the result?",
        ["📷 Static picture", "🎬 Watch it happen (animated)"],
        horizontal=True, key="view_mode",
    )
    animated = view_mode.startswith("🎬")
    if animated:
        st.caption("Press ▶ Play, or drag the slider to step through it yourself.")

    if trained["task"] == "clustering":
        result = explain.explain_kmeans(trained, raw_values)
        st.success(f"Predicted cluster: **Cluster {result['predicted_cluster']}**")
        st.write("**Distance to each centroid** (closest wins):")
        st.bar_chart(pd.Series(result["distances"]))
        if animated:
            fig = animate.animate_kmeans(
                trained["extra"]["X_transformed"], trained["extra"]["labels"],
                result["centroids"], result["query_t"], result["predicted_cluster"],
            )
            st.plotly_chart(fig, use_container_width=False)
        else:
            fig = viz.plot_kmeans_prediction(
                trained["extra"]["X_transformed"], trained["extra"]["labels"],
                result["centroids"], result["query_t"],
            )
            st.pyplot(fig, use_container_width=False)

    else:
        out = predict_single(trained["pipeline"], feature_cols_saved, raw_values, trained["target_encoder"])
        st.success(f"Predicted: **{out['label']}**")

        if algorithm == "K-Nearest Neighbors":
            ex = explain.explain_knn(trained, feature_cols_saved, raw_values)
            st.write(f"**The {len(ex['neighbor_idx'])} closest training points and how they voted:**")
            st.bar_chart(ex["vote_counts"])
            if animated:
                fig = animate.animate_knn(ex["X_train_t"], ex["train_labels_all"], ex["neighbor_idx"],
                                           ex["neighbor_labels"], ex["query_t"])
                st.plotly_chart(fig, use_container_width=False)
            else:
                fig = viz.plot_knn_neighbors(ex["X_train_t"], ex["train_labels_all"], ex["neighbor_idx"], ex["query_t"])
                st.pyplot(fig, use_container_width=False)

        elif algorithm == "Support Vector Machine":
            ex = explain.explain_svm(trained, feature_cols_saved, raw_values)
            st.write("**Decision score per class** (highest wins; the sign/size reflects distance from the boundary):")
            st.bar_chart(pd.Series(ex["scores"]))
            if animated:
                fig = animate.animate_svm(ex["X_train_t"], ex["train_labels_all"], ex["support_idx"], ex["query_t"])
                st.plotly_chart(fig, use_container_width=False)
            else:
                fig = viz.plot_svm_support(ex["X_train_t"], ex["train_labels_all"], ex["support_idx"], ex["query_t"])
                st.pyplot(fig, use_container_width=False)

        elif algorithm == "Decision Tree":
            ex = explain.explain_decision_tree(trained, feature_cols_saved, raw_values)
            st.write(f"**{len(ex['steps'])} split(s) checked** — the highlighted path below is the exact "
                     "route your input took through the real tree:")
            try:
                feat_names = clean_feature_names(trained["pipeline"].named_steps["prep"].get_feature_names_out())
            except Exception:
                feat_names = feature_cols_saved
            tree_model = trained["pipeline"].named_steps["model"]
            if animated:
                fig = animate.animate_decision_tree_path(
                    tree_model, feat_names, trained["extra"]["class_labels"],
                    ex["visited_nodes"], out["label"],
                )
                st.plotly_chart(fig, use_container_width=False)
            else:
                fig, tree_info = viz.plot_decision_tree(
                    tree_model, feat_names, trained["extra"]["class_labels"],
                    highlight_nodes=ex["visited_nodes"],
                )
                st.pyplot(fig, use_container_width=False)
                st.caption("Orange = a split your input passed through. Green = the leaf it landed on. "
                           "Faded boxes were never reached.")

        elif algorithm == "Perceptron":
            ex = explain.explain_perceptron(trained, feature_cols_saved, raw_values)
            st.write("**Weighted-sum score per class** (highest wins):")
            st.bar_chart(pd.Series(ex["breakdown"]))
            if animated:
                fig = animate.animate_perceptron(ex["breakdown"])
                st.plotly_chart(fig, use_container_width=False)

        if "probabilities" in out:
            st.write("**Model confidence per class:**")
            st.bar_chart(pd.Series(out["probabilities"]))
