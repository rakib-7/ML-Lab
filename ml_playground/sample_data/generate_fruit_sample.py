"""
Generates a fruit-classification dataset for the Perceptron demo.

This is inspired by the uploaded generate_dataset.py (Hagan-style fruit
prototypes) but deliberately reworked rather than reused as-is:
  - Prototypes are continuous, noisy measurements (real-valued shape/texture/
    weight/size scores) instead of clean +1/-1 codes with random sign flips.
    This makes the classes only *mostly* linearly separable, which is a more
    honest demo of what a perceptron can and can't do.
  - A 4th feature (size) is added.
  - Two more fruits are added (Grape, Lemon) to make it a 6-class problem.
  - Noise is added per-feature via Gaussian jitter around each prototype
    rather than discrete sign flips.
"""
import numpy as np
import pandas as pd

# Prototype centers per fruit: (shape, texture, weight, size), roughly in [-1, 1]
FRUIT_PROTOTYPES = {
    "Watermelon": (1.0, -1.0, 1.0, 1.0),
    "Banana": (-1.0, 1.0, -1.0, 0.3),
    "Orange": (1.0, -1.0, -0.3, -0.2),
    "Apple": (1.0, 1.0, -0.3, -0.3),
    "Grape": (1.0, -0.6, -1.0, -1.0),
    "Lemon": (0.6, -1.0, -0.6, -0.5),
}


def generate(n_samples=12000, seed=42, noise_std=0.35):
    rng = np.random.default_rng(seed)
    fruit_names = list(FRUIT_PROTOTYPES.keys())
    labels = rng.choice(fruit_names, size=n_samples)

    prototypes = np.array([FRUIT_PROTOTYPES[label] for label in labels])
    noisy = prototypes + rng.normal(0, noise_std, prototypes.shape)
    noisy = np.clip(noisy, -2, 2)

    df = pd.DataFrame(noisy, columns=["shape", "texture", "weight", "size"]).round(3)
    df["label"] = labels
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("fruit_sample.csv", index=False)
    print(f"fruit_sample.csv created with {len(df)} rows.")
    print(df["label"].value_counts())
