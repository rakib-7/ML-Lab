"""
Generates a synthetic customer-segmentation dataset for demoing the
classification algorithms (KNN, SVM, Decision Tree, Perceptron).

Deliberately mixes numeric + categorical columns and injects label noise,
so the bundled demo actually exercises the preprocessing pipeline
(imputation, scaling, one-hot encoding) the same way a messy real-world
CSV would.
"""
import numpy as np
import pandas as pd


def generate(n_samples=12000, seed=7, noise_frac=0.06):
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 70, n_samples).astype(float)
    annual_income = rng.normal(55000, 20000, n_samples).clip(12000, None)
    monthly_visits = rng.poisson(6, n_samples).astype(float)
    device = rng.choice(["mobile", "desktop", "tablet"], n_samples, p=[0.55, 0.35, 0.10])
    region = rng.choice(["North", "South", "East", "West"], n_samples)

    # A noisy-but-learnable rule defines the "true" segment, so every
    # algorithm has real signal to find (not pure noise, not trivially linear).
    score = (
        0.00003 * annual_income
        + 0.15 * monthly_visits
        - 0.01 * age
        + np.where(device == "desktop", 0.5, 0.0)
    )
    segment = np.select(
        [score < 1.5, score < 3.0],
        ["Budget", "Regular"],
        default="Premium",
    )

    # Flip a fraction of labels at random to simulate real-world label noise
    flip_mask = rng.random(n_samples) < noise_frac
    flipped = rng.choice(["Budget", "Regular", "Premium"], flip_mask.sum())
    segment = segment.copy()
    segment[flip_mask] = flipped

    # A few realistic missing values, since real CSVs are never perfectly clean
    income_arr = annual_income.copy()
    missing_idx = rng.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    income_arr[missing_idx] = np.nan

    df = pd.DataFrame({
        "age": age,
        "annual_income": income_arr.round(2),
        "monthly_visits": monthly_visits,
        "device": device,
        "region": region,
        "customer_segment": segment,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("classification_sample.csv", index=False)
    print(f"classification_sample.csv created with {len(df)} rows.")
    print(df["customer_segment"].value_counts())
