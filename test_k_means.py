import os
import random
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans as SkKMeans
from models.KMeans import KmeansClassifier

DIGITS_FILE_PATH = r"data\digits.csv"


def import_digits(
    filepath: str, test_size: float = 0.33
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load digits.csv where column 0 is the label and columns 1.. are features.
    Keeps behavior consistent with your runKMeans().
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist")

    data = pd.read_csv(filepath, header=0).values
    y = data[:, 0]
    if isinstance(y[0], str):
        classes = np.unique(y)
        mapping = dict(zip(classes, range(len(classes))))
        y = np.vectorize(mapping.get)(y)
    X = data[:, 1:].astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=1
    )
    return X_train, X_test, y_train, y_test


def _majority_centroid_assignments(
    centers: np.ndarray, X: np.ndarray, y: np.ndarray, n_classes: int = 10
) -> list[int]:
    """
    Map each centroid to a label by majority vote of training samples
    nearest to that centroid.
    """
    # Pairwise squared distances: (n_samples, k)
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    cluster_idx = dists.argmin(axis=1)

    mapping = np.zeros(centers.shape[0], dtype=int)
    global_majority = int(np.bincount(y.astype(int)).argmax())

    for c in range(centers.shape[0]):
        labels_c = y[cluster_idx == c].astype(int)
        if labels_c.size == 0:
            mapping[c] = global_majority  # rare fallback
        else:
            mapping[c] = int(np.bincount(labels_c, minlength=n_classes).argmax())

    return mapping.tolist()


def test_kmeans_digits_accuracy_vs_baseline_and_sklearn() -> None:
    """
    Train your KmeansClassifier (k=10) on digits.csv, derive centroid→label mapping
    by majority vote on the TRAIN set, and check:
      1) Accuracy beats a naive baseline.
      2) Accuracy is not wildly worse than scikit-learn's KMeans.
      3) Shapes/sanity are as expected.
    """
    random.seed(1)
    np.random.seed(1)

    X_train, X_test, y_train, y_test = import_digits(DIGITS_FILE_PATH)

    # --- My KMeans ---
    model = KmeansClassifier(n_clusters=10, max_iter=500, threshold=1e-6)
    model.train(X_train)
    assert model.cluster_centers.shape == (10, X_train.shape[1])

    my_assign = _majority_centroid_assignments(model.cluster_centers, X_train, y_train, n_classes=10)
    my_preds = model.predict(X_test, centroid_assignments=my_assign)
    my_acc = float(np.mean(my_preds == y_test))

    # Baseline = majority class in training labels
    baseline = float(np.max(np.bincount(y_train.astype(int))) / y_train.shape[0])

    # Must handily beat baseline and be at least reasonable
    assert my_acc >= max(0.5, baseline * 2.0)

    # --- scikit-learn KMeans reference ---
    sk = SkKMeans(n_clusters=10, n_init=10, random_state=1)
    sk.fit(X_train)

    sk_assign = _majority_centroid_assignments(sk.cluster_centers_, X_train, y_train, n_classes=10)
    # Predict for sklearn by nearest center
    sk_dists = ((X_test[:, None, :] - sk.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    sk_clusters = sk_dists.argmin(axis=1)
    sk_preds = np.array([sk_assign[c] for c in sk_clusters])
    sk_acc = float(np.mean(sk_preds == y_test))

    # Parity: not more than ~20 points worse than sklearn
    assert my_acc >= sk_acc - 0.20

    # Sanity: finite
    assert np.isfinite(my_acc) and np.isfinite(sk_acc)
