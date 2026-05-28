"""
losses.py
---------
Loss functions and their gradients with respect to the output layer activations.

Supported losses
  - cross_entropy  : for multi-class classification with softmax output
  - binary_cross_entropy : for binary / sigmoid output
  - mse            : mean squared error
"""

import numpy as np


# ─────────────────────────────────────────────
#  Categorical Cross-Entropy
#  L = -1/m * Σ y * log(ŷ)   (ŷ from softmax)
# ─────────────────────────────────────────────
def cross_entropy(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    Parameters
    ----------
    y_hat : (n_classes, m) – softmax probabilities
    y     : (n_classes, m) – one-hot encoded labels

    Returns
    -------
    scalar loss averaged over the batch.
    """
    m = y.shape[1]
    # Clip to avoid log(0)
    y_hat = np.clip(y_hat, 1e-12, 1.0 - 1e-12)
    return -np.sum(y * np.log(y_hat)) / m


def cross_entropy_backward(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient of CE loss w.r.t. the *pre-softmax* logits (z) when the
    output activation is softmax.  The combined softmax + CE gradient is:
        dL/dz = (ŷ - y) / m
    """
    m = y.shape[1]
    return (y_hat - y) / m


# ─────────────────────────────────────────────
#  Binary Cross-Entropy   (sigmoid output)
#  L = -1/m * Σ [y*log(ŷ) + (1-y)*log(1-ŷ)]
# ─────────────────────────────────────────────
def binary_cross_entropy(y_hat: np.ndarray, y: np.ndarray) -> float:
    m = y.shape[1]
    y_hat = np.clip(y_hat, 1e-12, 1.0 - 1e-12)
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m


def binary_cross_entropy_backward(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = y.shape[1]
    y_hat = np.clip(y_hat, 1e-12, 1.0 - 1e-12)
    return (-(y / y_hat) + (1 - y) / (1 - y_hat)) / m


# ─────────────────────────────────────────────
#  Mean Squared Error
#  L = 1/(2m) * Σ (ŷ - y)²
# ─────────────────────────────────────────────
def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    m = y.shape[1]
    return np.sum((y_hat - y) ** 2) / (2 * m)


def mse_backward(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = y.shape[1]
    return (y_hat - y) / m


# ─────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────
LOSSES = {
    "cross_entropy":         (cross_entropy,         cross_entropy_backward),
    "binary_cross_entropy":  (binary_cross_entropy,  binary_cross_entropy_backward),
    "mse":                   (mse,                   mse_backward),
}
