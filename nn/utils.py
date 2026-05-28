"""
utils.py
--------
Helper functions:
  - load_mnist()          : download and cache MNIST via scikit-learn
  - one_hot_encode()      : convert integer labels to one-hot matrices
  - normalize()           : scale pixel values to [0, 1]
  - train_test_split_nn() : split data into train/test sets
  - plot_history()        : plot loss & accuracy curves
  - plot_confusion_matrix(): plot confusion matrix heatmap
  - show_sample_predictions(): visualize a grid of predictions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Tuple, Optional


# ─────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────

def load_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the MNIST dataset via scikit-learn (downloads on first call).

    Returns
    -------
    X : (70000, 784) float64 array of pixel values
    y : (70000,)     uint8  array of digit labels 0-9
    """
    from sklearn.datasets import fetch_openml
    print("Loading MNIST (may download ~12 MB on first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float64)
    y = mnist.target.astype(np.uint8)
    print(f"  Loaded: X={X.shape}, y={y.shape}")
    return X, y


def normalize(X: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0, 1]."""
    return X / 255.0


def one_hot_encode(y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """
    Convert integer labels to one-hot matrix.

    Parameters
    ----------
    y         : (m,) integer labels
    n_classes : number of output classes

    Returns
    -------
    Y : (n_classes, m) one-hot matrix
    """
    m = y.shape[0]
    Y = np.zeros((n_classes, m))
    Y[y, np.arange(m)] = 1.0
    return Y


def train_test_split_nn(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into train and test subsets.

    Note: X and y here follow the (features, samples) convention,
    so columns are samples.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    np.random.seed(seed)
    m = X.shape[1]
    idx = np.random.permutation(m)
    n_test = int(m * test_size)
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]
    return X[:, train_idx], X[:, test_idx], y[:, train_idx], y[:, test_idx]


# ─────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────

def plot_history(
    costs: list,
    train_accs: list,
    val_accs: Optional[list] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot training loss and accuracy curves side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    epochs = range(1, len(costs) + 1)

    # Loss curve
    axes[0].plot(epochs, costs, color="#7b68ee", linewidth=2, label="Loss")
    axes[0].set_title("Training Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(facecolor="#222", labelcolor="white")

    # Accuracy curve
    axes[1].plot(epochs, train_accs, color="#00d4ff", linewidth=2, label="Train")
    if val_accs:
        axes[1].plot(epochs, val_accs, color="#ff6b6b", linewidth=2, label="Validation")
    axes[1].set_title("Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(facecolor="#222", labelcolor="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[list] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a confusion matrix heatmap."""
    n = len(np.unique(y_true))
    if classes is None:
        classes = [str(i) for i in range(n)]

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(cm, interpolation="nearest", cmap="magma")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="white")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] < thresh else "black",
                    fontsize=8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, color="white")
    ax.set_yticklabels(classes, color="white")
    ax.set_xlabel("Predicted label", color="white", fontsize=12)
    ax.set_ylabel("True label", color="white", fontsize=12)
    ax.set_title("Confusion Matrix", color="white", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


def show_sample_predictions(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 25,
    save_path: Optional[str] = None,
) -> None:
    """
    Display a grid of sample images with predicted vs. true labels.
    Correct predictions are framed in green, wrong ones in red.

    Parameters
    ----------
    X      : (784, m) pixel matrix  — columns are individual images
    y_true : (m,) true labels
    y_pred : (m,) predicted labels
    n      : number of samples to display
    """
    cols = 5
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(cols * 2.2, rows * 2.4))
    fig.patch.set_facecolor("#0f0f1a")

    indices = np.random.choice(X.shape[1], n, replace=False)
    for idx, sample_idx in enumerate(indices):
        ax = fig.add_subplot(rows, cols, idx + 1)
        img = X[:, sample_idx].reshape(28, 28)
        ax.imshow(img, cmap="gray")

        true_l = y_true[sample_idx]
        pred_l = y_pred[sample_idx]
        correct = true_l == pred_l

        color  = "#00ff88" if correct else "#ff4444"
        border = plt.Rectangle((0, 0), 1, 1, fill=False,
                                edgecolor=color, linewidth=3,
                                transform=ax.transAxes)
        ax.add_patch(border)
        ax.set_title(f"T:{true_l} P:{pred_l}", color=color, fontsize=9,
                     fontweight="bold")
        ax.axis("off")

    plt.suptitle("Sample Predictions  (Green=Correct  Red=Wrong)",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()
