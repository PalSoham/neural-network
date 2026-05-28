"""
network.py
----------
Core NeuralNetwork class.

Architecture
============
The network is a fully-connected (dense) multi-layer perceptron.
You specify the layer sizes as a list:

    layer_sizes = [784, 128, 32, 10]
    # → input layer: 784 neurons (flattened 28×28 MNIST pixels)
    # → hidden layer 1: 128 neurons
    # → hidden layer 2: 32 neurons
    # → output layer: 10 neurons (one per digit class)

Each hidden layer uses the chosen activation function.
The output layer uses softmax for multi-class classification.

Forward Pass
============
For each layer l:
    Z[l] = W[l] @ A[l-1] + b[l]     (linear step)
    A[l] = g(Z[l])                   (activation step)

Backpropagation
===============
Starting from the output layer and moving backwards:
    dZ[L]   = dL/dZ[L]              (from loss gradient)
    dW[l]   = dZ[l] @ A[l-1].T / m
    db[l]   = mean(dZ[l], axis=1)
    dA[l-1] = W[l].T @ dZ[l]
    dZ[l-1] = dA[l-1] * g'(Z[l-1])

Gradient Descent Update
=======================
    W[l] -= lr * dW[l]
    b[l] -= lr * db[l]
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from nn.activations import ACTIVATIONS, softmax
from nn.losses import LOSSES


class NeuralNetwork:
    """
    Fully-connected Neural Network built from scratch using only NumPy.

    Parameters
    ----------
    layer_sizes      : List of ints. Number of neurons in each layer
                       (including the input and output layers).
    hidden_activation: Activation function for hidden layers.
                       One of: 'relu', 'sigmoid', 'tanh', 'leaky_relu'.
    loss             : Loss function. One of: 'cross_entropy', 'mse'.
    seed             : Random seed for reproducibility.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        hidden_activation: str = "relu",
        loss: str = "cross_entropy",
        seed: Optional[int] = 42,
    ):
        if seed is not None:
            np.random.seed(seed)

        if hidden_activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{hidden_activation}'. "
                             f"Choose from: {list(ACTIVATIONS.keys())}")
        if loss not in LOSSES:
            raise ValueError(f"Unknown loss '{loss}'. "
                             f"Choose from: {list(LOSSES.keys())}")

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1   # number of weight matrices
        self.hidden_activation = hidden_activation
        self._act_fn, self._act_fn_backward = ACTIVATIONS[hidden_activation]
        self._loss_fn, self._loss_backward = LOSSES[loss]

        # Training history
        self.costs: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []

        # Parameters will be initialized in fit()
        self.params: Dict[str, np.ndarray] = {}

    # ─────────────────────────────────────────────────────────────────────────
    #  Parameter Initialization (He initialization for ReLU family,
    #  Xavier/Glorot for sigmoid/tanh)
    # ─────────────────────────────────────────────────────────────────────────
    def _initialize_parameters(self) -> None:
        """Initialize weights (He or Xavier) and biases (zeros)."""
        self.params = {}
        for l in range(1, self.n_layers + 1):
            n_in  = self.layer_sizes[l - 1]
            n_out = self.layer_sizes[l]

            # He initialization → good for ReLU
            # Xavier initialization → good for sigmoid / tanh
            if self.hidden_activation in ("relu", "leaky_relu"):
                scale = np.sqrt(2.0 / n_in)
            else:
                scale = np.sqrt(1.0 / n_in)   # Xavier

            self.params[f"W{l}"] = np.random.randn(n_out, n_in) * scale
            self.params[f"b{l}"] = np.zeros((n_out, 1))

    # ─────────────────────────────────────────────────────────────────────────
    #  Forward Pass
    # ─────────────────────────────────────────────────────────────────────────
    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Perform a full forward pass through the network.

        Parameters
        ----------
        X : (n_features, m) input matrix

        Returns
        -------
        y_hat   : output activations (predictions)
        cache   : dict storing Z and A for each layer (needed for backprop)
        """
        cache = {"A0": X}
        A = X

        # Hidden layers
        for l in range(1, self.n_layers):
            Z = self.params[f"W{l}"] @ A + self.params[f"b{l}"]
            A = self._act_fn(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        # Output layer → always softmax for multi-class CE
        l = self.n_layers
        Z = self.params[f"W{l}"] @ A + self.params[f"b{l}"]
        A = softmax(Z)          # softmax output
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A

        return A, cache

    # ─────────────────────────────────────────────────────────────────────────
    #  Backward Pass (Backpropagation)
    # ─────────────────────────────────────────────────────────────────────────
    def _backward(self, y_hat: np.ndarray, y: np.ndarray, cache: Dict) -> Dict:
        """
        Backpropagate the loss gradient through all layers.

        Parameters
        ----------
        y_hat : (n_classes, m) – network output from forward pass
        y     : (n_classes, m) – one-hot ground truth labels
        cache : dict from _forward()

        Returns
        -------
        grads : dict with dW{l} and db{l} for each layer l
        """
        grads = {}
        L = self.n_layers

        # ── Output layer gradient ────────────────────────────────────────────
        # Combined softmax + cross-entropy gradient: dZ = (ŷ - y) / m
        # NOTE: _loss_backward already includes the 1/m normalization,
        #       so we do NOT divide by m again when computing dW and db.
        dZ = self._loss_backward(y_hat, y)

        # ── Propagate from L down to 1 ───────────────────────────────────────
        for l in range(L, 0, -1):
            A_prev = cache[f"A{l-1}"]

            # dW = dZ @ A_prev.T  (1/m already in dZ from loss_backward)
            grads[f"dW{l}"] = dZ @ A_prev.T
            # db = sum over batch dimension (1/m already in dZ)
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True)

            if l > 1:
                # Gradient w.r.t. previous layer's activation
                dA_prev = self.params[f"W{l}"].T @ dZ
                # Element-wise multiply with derivative of activation
                Z_prev  = cache[f"Z{l-1}"]
                dZ      = dA_prev * self._act_fn_backward(Z_prev)

        return grads

    # ─────────────────────────────────────────────────────────────────────────
    #  Parameter Update (Gradient Descent)
    # ─────────────────────────────────────────────────────────────────────────
    def _update_parameters(self, grads: Dict, lr: float) -> None:
        """Vanilla gradient descent update."""
        for l in range(1, self.n_layers + 1):
            self.params[f"W{l}"] -= lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= lr * grads[f"db{l}"]

    # ─────────────────────────────────────────────────────────────────────────
    #  Predict
    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted class indices for each sample.

        Parameters
        ----------
        X : (n_features, m)

        Returns
        -------
        predictions : (m,) array of predicted class indices
        """
        y_hat, _ = self._forward(X)
        return np.argmax(y_hat, axis=0)

    # ─────────────────────────────────────────────────────────────────────────
    #  Accuracy
    # ─────────────────────────────────────────────────────────────────────────
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy.

        Parameters
        ----------
        X : (n_features, m)
        y : (n_classes, m) one-hot labels  OR  (m,) integer labels

        Returns
        -------
        accuracy : float in [0, 1]
        """
        preds = self.predict(X)
        if y.ndim == 2:
            true_labels = np.argmax(y, axis=0)
        else:
            true_labels = y
        return np.mean(preds == true_labels)

    # ─────────────────────────────────────────────────────────────────────────
    #  Fit (Training Loop)
    # ─────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        print_every: int = 10,
        batch_size: Optional[int] = None,
    ) -> "NeuralNetwork":
        """
        Train the network.

        Parameters
        ----------
        X_train    : (n_features, m_train)
        y_train    : (n_classes,  m_train) – one-hot encoded
        epochs     : number of training epochs
        lr         : learning rate
        X_val      : optional validation features (n_features, m_val)
        y_val      : optional validation labels   (n_classes,  m_val)
        print_every: print progress every N epochs (0 to silence)
        batch_size : mini-batch size (None → full-batch gradient descent)

        Returns
        -------
        self (for method chaining)
        """
        self._initialize_parameters()
        m = X_train.shape[1]

        for epoch in range(1, epochs + 1):
            # ── Mini-batch or full-batch ──────────────────────────────────────
            if batch_size is not None:
                indices = np.random.permutation(m)
                X_shuffled = X_train[:, indices]
                y_shuffled = y_train[:, indices]
                epoch_loss = 0.0
                n_batches  = int(np.ceil(m / batch_size))

                for i in range(n_batches):
                    X_batch = X_shuffled[:, i * batch_size : (i + 1) * batch_size]
                    y_batch = y_shuffled[:, i * batch_size : (i + 1) * batch_size]

                    y_hat, cache = self._forward(X_batch)
                    epoch_loss  += self._loss_fn(y_hat, y_batch)
                    grads        = self._backward(y_hat, y_batch, cache)
                    self._update_parameters(grads, lr)

                cost = epoch_loss / n_batches

            else:
                # Full-batch gradient descent
                y_hat, cache = self._forward(X_train)
                cost = self._loss_fn(y_hat, y_train)
                grads = self._backward(y_hat, y_train, cache)
                self._update_parameters(grads, lr)

            # ── Record metrics ────────────────────────────────────────────────
            self.costs.append(cost)
            train_acc = self.accuracy(X_train, y_train)
            self.train_accuracies.append(train_acc)

            if X_val is not None and y_val is not None:
                val_acc = self.accuracy(X_val, y_val)
                self.val_accuracies.append(val_acc)
            else:
                val_acc = None

            # ── Logging ───────────────────────────────────────────────────────
            if print_every > 0 and (epoch % print_every == 0 or epoch == 1):
                msg = (f"Epoch {epoch:>4}/{epochs}  |  "
                       f"Loss: {cost:.4f}  |  "
                       f"Train Acc: {train_acc:.4f}")
                if val_acc is not None:
                    msg += f"  |  Val Acc: {val_acc:.4f}"
                print(msg)

        return self

    # ─────────────────────────────────────────────────────────────────────────
    #  String Representation
    # ─────────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        lines = ["NeuralNetwork"]
        lines.append(f"  Activation : {self.hidden_activation}")
        lines.append(f"  Architecture:")
        for l, (n_in, n_out) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:]), start=1
        ):
            label = "Output" if l == self.n_layers else f"Hidden {l}"
            lines.append(f"    Layer {l} ({label}): {n_in} -> {n_out}")
        total = sum(
            self.params[f"W{l}"].size + self.params[f"b{l}"].size
            for l in range(1, self.n_layers + 1)
        ) if self.params else 0
        lines.append(f"  Total parameters: {total:,}")
        return "\n".join(lines)
