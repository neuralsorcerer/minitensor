"""K-means clustering on the Iris dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from random import Random

import numpy as np

import minitensor as mt

DATA_PATH = Path(__file__).resolve().parent / "data" / "iris.csv"


def load_features():
    with open(DATA_PATH, newline="") as f:
        reader = csv.reader(f)
        feats = [[float(v) for v in row[:4]] for row in reader if row]
    return mt.Tensor(feats, dtype="float32")


def kmeans(x: mt.Tensor, k: int = 3, iters: int = 10):
    rng = Random(0)
    indices = list(range(x.shape[0]))
    rng.shuffle(indices)
    centers = [x[i] for i in indices[:k]]
    for _ in range(iters):
        dists = []
        for c in centers:
            diff = x - c
            dists.append((diff * diff).sum(dim=1).unsqueeze(1))
        dist_mat = mt.from_numpy(np.concatenate([d.numpy() for d in dists], axis=1))
        assignments = dist_mat.argmin(dim=1)
        new_centers = []
        for j in range(k):
            mask = assignments.eq(j).astype("float32").unsqueeze(1)
            denom = mask.sum(dim=0)
            new_centers.append((x * mask).sum(dim=0) / denom)
        centers = new_centers
    return centers, assignments


def main() -> None:  # pragma: no cover - example script
    x = load_features()
    centers, assignments = kmeans(x)
    print("Cluster centers:")
    for c in centers:
        print(c.numpy().tolist())
    counts = []
    for j in range(len(centers)):
        count = assignments.eq(j).astype("float32").sum().numpy().ravel()[0]
        counts.append(int(count))
    print("Cluster counts:", counts)


if __name__ == "__main__":  # pragma: no cover - example script
    main()
