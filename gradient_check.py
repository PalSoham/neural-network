"""
gradient_check.py
-----------------
Numerically verifies that the analytical backpropagation gradients are
correct by comparing them with finite-difference approximations.

Run with:
    python gradient_check.py

A relative error < 1e-5 is considered acceptable.
"""

import numpy as np
from nn.network import NeuralNetwork
from nn.utils import one_hot_encode


def compute_numerical_gradient(
    model: NeuralNetwork,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float = 1e-5,
) -> dict:
    """
    Compute numerical gradient for all parameters using central differences:
        ∂L/∂θ ≈ [L(θ+ε) - L(θ-ε)] / (2ε)
    """
    numerical_grads = {}

    for key in model.params:
        param = model.params[key]
        grad  = np.zeros_like(param)

        with np.nditer(param, flags=["multi_index"], op_flags=["readwrite"]) as it:
            for _ in it:
                idx = it.multi_index

                # θ + ε
                param[idx] += epsilon
                y_hat_plus, _ = model._forward(X)
                loss_plus = model._loss_fn(y_hat_plus, Y)

                # θ - ε
                param[idx] -= 2 * epsilon
                y_hat_minus, _ = model._forward(X)
                loss_minus = model._loss_fn(y_hat_minus, Y)

                # Restore
                param[idx] += epsilon

                grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        numerical_grads[f"d{key}"] = grad

    return numerical_grads


def relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    num   = np.linalg.norm(analytical - numerical)
    denom = np.linalg.norm(analytical) + np.linalg.norm(numerical) + 1e-12
    return num / denom


def run_gradient_check():
    print("\n" + "="*55)
    print("  Gradient Check")
    print("="*55)

    # Use a tiny network and a tiny dataset for speed
    np.random.seed(0)
    layer_sizes = [4, 5, 3]
    m = 8   # samples

    X = np.random.randn(4, m)
    y_int = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    Y = one_hot_encode(y_int, n_classes=3)

    model = NeuralNetwork(layer_sizes=layer_sizes,
                          hidden_activation="tanh",
                          loss="cross_entropy",
                          seed=1)
    model._initialize_parameters()

    # Analytical gradients
    y_hat, cache = model._forward(X)
    analytical   = model._backward(y_hat, Y, cache)

    # Numerical gradients
    numerical = compute_numerical_gradient(model, X, Y)

    print(f"\n  {'Parameter':<12}  {'Analytic norm':>14}  {'Numeric norm':>12}  {'Rel. Error':>12}  {'Status'}")
    print("  " + "-"*70)

    all_ok = True
    for key in sorted(analytical.keys()):
        a = analytical[key]
        n = numerical[key]
        err = relative_error(a, n)
        status = "[OK]" if err < 1e-5 else "[FAIL]"
        if err >= 1e-5:
            all_ok = False
        print(f"  {key:<12}  {np.linalg.norm(a):>14.6f}  "
              f"{np.linalg.norm(n):>12.6f}  {err:>12.2e}  {status}")

    print("\n  " + ("All gradients correct!" if all_ok else "Some gradients FAILED!"))
    print("="*55 + "\n")


if __name__ == "__main__":
    run_gradient_check()
