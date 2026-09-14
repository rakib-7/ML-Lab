"""Cosmetic helper: sklearn's ColumnTransformer prefixes output feature names
with 'num__' / 'cat__' (and one-hot columns look like 'cat__city_Boston').
Beginners shouldn't have to see that -- strip it for anything shown in the UI.
"""


def clean_feature_name(name: str) -> str:
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def clean_feature_names(names):
    return [clean_feature_name(n) for n in names]
