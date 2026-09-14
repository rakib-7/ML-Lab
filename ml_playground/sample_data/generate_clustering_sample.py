"""
Generates a synthetic customer-spending dataset for demoing K-Means.

Uses well-separated-but-overlapping Gaussian blobs (unequal spread per
group, like real segments would have) rather than a hand-labeled target --
K-Means never sees labels, only the raw numeric features.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


def generate(n_samples=12000, seed=11):
    centers = np.array([
        [20, 2],    # occasional shoppers, small basket
        [80, 15],   # frequent shoppers, small basket
        [30, 40],   # occasional shoppers, big basket
        [90, 45],   # frequent shoppers, big basket
    ])
    X, _true_cluster = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=[6, 9, 7, 10],
        random_state=seed,
    )
    annual_visits = np.clip(X[:, 0], 1, None).round(1)
    avg_basket_size = np.clip(X[:, 1], 1, None).round(2)

    df = pd.DataFrame({
        "annual_visits": annual_visits,
        "avg_basket_size": avg_basket_size,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("clustering_sample.csv", index=False)
    print(f"clustering_sample.csv created with {len(df)} rows.")
    print(df.describe())
