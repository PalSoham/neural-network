"""
activations.py
--------------
Collection of activation functions and their derivatives.

Each activation is implemented as a pair:
  - forward(z)  -> activated output
  - backward(z) -> derivative w.r.t. the pre-activation z
"""

import numpy as np


# ─────────────────────────────────────────────
#  ReLU  f(z) = max(0, z)
# ─────────────────────────────────────────────
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_backward(z: np.ndarray) -> np.ndarray:
    """d(ReLU)/dz = 1 where z > 0, else 0."""
    return (z > 0).astype(float)


# ─────────────────────────────────────────────
#  Sigmoid  f(z) = 1 / (1 + exp(-z))
# ─────────────────────────────────────────────
def sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip for numerical stability
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_backward(z: np.ndarray) -> np.ndarray:
    """d(sigmoid)/dz = sigmoid(z) * (1 - sigmoid(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)


# ─────────────────────────────────────────────
#  Tanh  f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
# ─────────────────────────────────────────────
def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_backward(z: np.ndarray) -> np.ndarray:
    """d(tanh)/dz = 1 - tanh²(z)."""
    return 1.0 - np.tanh(z) ** 2


# ─────────────────────────────────────────────
#  Leaky ReLU  f(z) = z if z > 0, else alpha * z
# ─────────────────────────────────────────────
def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)


def leaky_relu_backward(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, 1.0, alpha)


# ─────────────────────────────────────────────
#  Softmax  (used in the output layer for multi-class)
# ─────────────────────────────────────────────
def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z_shifted)
    return e / np.sum(e, axis=0, keepdims=True)


# ─────────────────────────────────────────────
#  Registry: map string name → (forward, backward)
# ─────────────────────────────────────────────
ACTIVATIONS = {
    "relu":       (relu,       relu_backward),
    "sigmoid":    (sigmoid,    sigmoid_backward),
    "tanh":       (tanh,       tanh_backward),
    "leaky_relu": (leaky_relu, leaky_relu_backward),
    "softmax":    (softmax,    None),   # backward handled specially in output layer
}
