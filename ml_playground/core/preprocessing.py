"""Builds reusable preprocessing pipelines (imputation, scaling, encoding).

Using a single fitted ColumnTransformer guarantees that the exact same
transformations applied during training are applied again at prediction
time -- no manual bookkeeping, no train/serve skew.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from .data_loader import infer_column_types


def build_feature_preprocessor(X):
    """ColumnTransformer: numeric -> impute+scale, categorical -> impute+one-hot."""
    col_types = infer_column_types(X)
    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers), col_types


def encode_target_if_needed(y):
    """Label-encode the target if it's categorical/text; returns (y_encoded, encoder_or_None).

    Uses is_numeric_dtype rather than checking for `object` specifically, since
    pandas' newer Arrow-backed string dtype (pandas >= 2.x with the string
    backend, default in pandas 3.0) is not `object` but is also not numeric.
    """
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        return le.fit_transform(y.astype(str)), le
    return y.to_numpy(), None
