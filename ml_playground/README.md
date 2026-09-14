# ML Playground

A beginner-friendly web app for training and using five classic machine
learning algorithms — **K-Nearest Neighbors**, **Support Vector Machine**,
**Decision Tree**, **Perceptron**, and **K-Means Clustering** — on your own
data, or on three bundled 12,000-row sample datasets.

Every prediction comes with a visual explanation of *how* the model arrived
at that answer (nearest neighbors highlighted, decision-tree path traced,
support vectors shown, perceptron weighted sums broken down, K-Means
distances to each centroid) — not just the final label.

## How it works

1. **Pick an algorithm** from the five cards, and read the plain-language
   "how does this work?" explanation.
2. **Provide data** — use one of the bundled 12,000-row sample datasets, or
   upload your own CSV/Excel file.
3. **Choose columns** and **hyperparameters** (sliders/dropdowns with
   sensible defaults and tooltips).
4. **Train** — see accuracy/metrics, a confusion matrix, and (depending on
   the algorithm) a decision-tree diagram or a perceptron learning curve.
5. **Predict** — enter new values and watch the actual mechanism: which
   neighbors voted, which support vectors mattered, which yes/no path the
   tree took, which centroid was closest, or which class scored highest.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Bundled sample datasets (12,000 rows each)

| File | Used for | Notes |
|---|---|---|
| `sample_data/classification_sample.csv` | KNN, SVM, Decision Tree, Perceptron | Synthetic customer segmentation; mixes numeric + categorical columns, has missing values and label noise on purpose, so it exercises the full preprocessing pipeline. |
| `sample_data/fruit_sample.csv` | Perceptron (and any classifier) | Reworked from the uploaded fruit-classification example — continuous noisy shape/texture/weight/size measurements across 6 fruits, instead of clean ±1 codes. |
| `sample_data/clustering_sample.csv` | K-Means | Synthetic shopper-behavior blobs (visits vs. basket size), no labels. |

Each has a matching `generate_*.py` script in `sample_data/` so you can see
(or tweak) exactly how it was produced.

## Project structure

```
ml_playground/
├── app.py                    # Streamlit UI — the only file that touches the screen
├── core/
│   ├── data_loader.py        # File upload + validation
│   ├── preprocessing.py      # Imputation, scaling, encoding (sklearn ColumnTransformer)
│   ├── trainer.py            # Unified train/evaluate for all 5 algorithms
│   ├── predictor.py          # Scores a single new input row
│   ├── explain.py            # Computes the "why" behind a prediction (neighbors, support vectors, tree path, etc.)
│   ├── explanations.py       # Plain-language descriptions of each algorithm
│   └── models/
│       ├── knn.py            # Model + hyperparameter spec
│       ├── svm.py
│       ├── decision_tree.py
│       ├── perceptron.py     # From-scratch one-vs-rest hardlim perceptron (sklearn-compatible)
│       └── kmeans.py
├── utils/
│   ├── metrics.py            # Accuracy/F1/silhouette/elbow calculations
│   ├── viz.py                # Matplotlib figures, incl. "how prediction worked" visuals
│   └── naming.py             # Strips ColumnTransformer prefixes for display
├── sample_data/               # 3 bundled 12k-row datasets + their generator scripts
└── requirements.txt
```

## Why it's built this way

- **One Pipeline per model**: preprocessing (imputer/scaler/encoder) is fused
  with the estimator into a single `sklearn.pipeline.Pipeline`. This guarantees
  whatever transformation was applied during training is applied identically
  at prediction time — no manual bookkeeping, no train/serve mismatch.
- **Registry pattern** (`core/models/__init__.py`): adding a 6th algorithm
  later means writing one new file with a `build_model()` function and a
  `PARAM_SPEC` dict — `app.py` and `trainer.py` don't need to change.
- **Declarative hyperparameters** (`PARAM_SPEC`): each model describes its own
  tunable parameters (type, range, default, help text). The UI renders sliders
  and dropdowns generically from this spec instead of hardcoding widgets per
  algorithm.
- **The Perceptron is a real sklearn estimator, not a script**: the uploaded
  fruit-classifier app's training loop was rebuilt as a `BaseEstimator` /
  `ClassifierMixin` subclass, so it can sit inside the same
  `ColumnTransformer` + `Pipeline` as every other algorithm and work on any
  uploaded dataset — while still keeping the full per-epoch weight/error
  history needed to visualize how it converged.
- **`explain.py` is separate from `predictor.py`**: getting a label is cheap;
  computing "why" (nearest neighbors, support vectors, a tree's decision path,
  distances to every centroid) is extra work that only runs when you actually
  ask to see it.
- **Separation of concerns**: `app.py` only handles layout/interaction; all
  ML logic lives in `core/`, so it can be tested or reused (e.g. behind an
  API) without Streamlit at all.

## Extending

To add a new algorithm:
1. Create `core/models/your_algo.py` with `TASK`, `PARAM_SPEC`, and `build_model(params)`.
2. Register it in `core/models/__init__.py`'s `REGISTRY` (and `UNSUPERVISED` if it has no target column).
3. Optionally add a `core/explanations.py` entry and an `explain_*` function in `core/explain.py` for a custom "how it predicted" visual.

That's it — the algorithm picker, training, and prediction flow all pick it up automatically.

