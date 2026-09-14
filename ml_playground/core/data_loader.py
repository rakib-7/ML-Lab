"""Handles reading uploaded files into a validated pandas DataFrame."""
import pandas as pd


class DataLoadError(Exception):
    pass


def load_file(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel file-like object into a DataFrame with basic validation."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            raise DataLoadError("Unsupported file type. Please upload a .csv or .xlsx file.")
    except Exception as exc:
        raise DataLoadError(f"Could not read the file: {exc}") from exc

    if df.empty:
        raise DataLoadError("The uploaded file has no rows.")
    if df.shape[1] < 2:
        raise DataLoadError("Need at least 2 columns (at least one feature + one target).")

    # Drop fully-empty rows/columns which commonly appear in exported spreadsheets
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df


def infer_column_types(df: pd.DataFrame) -> dict:
    """Split columns into numeric vs categorical for downstream preprocessing."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return {"numeric": numeric_cols, "categorical": categorical_cols}
