"""
train.py
--------
End-to-end training script for the Neural Network on MNIST.

Run with:
    python train.py

Configurable parameters are at the top of the file in the CONFIG section.
"""

import numpy as np
import os

from nn import NeuralNetwork
from nn.utils import (
    load_mnist,
    normalize,
    one_hot_encode,
    train_test_split_nn,
    plot_history,
    plot_confusion_matrix,
    show_sample_predictions,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG — tweak these to experiment
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Network architecture: [input_size, hidden_1, ..., output_size]
    "layer_sizes":        [784, 128, 32, 10],

    # Activation for hidden layers: 'relu', 'sigmoid', 'tanh', 'leaky_relu'
    "hidden_activation":  "relu",

    # Training hyper-parameters
    "learning_rate":      0.03,
    "epochs":             200,
    "batch_size":         None,   # None → full-batch gradient descent

    # Train / Test split
    "test_size":          0.2,

    # How often to print epoch progress
    "print_every":        20,

    # Output directory for saved plots
    "output_dir":         "output",

    # Random seed
    "seed":               42,
}
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ── 1. Load & Preprocess Data ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Neural Network from Scratch — MNIST")
    print("="*60)

    X_raw, y_raw = load_mnist()

    # Normalize pixel values to [0, 1]
    X = normalize(X_raw)

    # Transpose to (features, samples) convention  →  (784, 70000)
    X = X.T

    # One-hot encode labels  →  (10, 70000)
    Y = one_hot_encode(y_raw, n_classes=10)

    # Train / test split
    X_train, X_test, Y_train, Y_test = train_test_split_nn(
        X, Y, test_size=CONFIG["test_size"], seed=CONFIG["seed"]
    )
    y_test_int   = np.argmax(Y_test, axis=0)

    print(f"\n  Train samples : {X_train.shape[1]:,}")
    print(f"  Test  samples : {X_test.shape[1]:,}")
    print(f"  Features      : {X_train.shape[0]}")

    # ── 2. Build Network ─────────────────────────────────────────────────────
    print("\n" + "-"*60)
    model = NeuralNetwork(
        layer_sizes      = CONFIG["layer_sizes"],
        hidden_activation= CONFIG["hidden_activation"],
        loss             = "cross_entropy",
        seed             = CONFIG["seed"],
    )
    # Initialize parameters so __repr__ shows param count
    model._initialize_parameters()
    print(model)
    print("-"*60)

    # ── 3. Train ─────────────────────────────────────────────────────────────
    print(f"\n  Training for {CONFIG['epochs']} epochs "
          f"(lr={CONFIG['learning_rate']}, "
          f"batch={'full' if CONFIG['batch_size'] is None else CONFIG['batch_size']})...\n")

    model.fit(
        X_train = X_train,
        y_train = Y_train,
        epochs  = CONFIG["epochs"],
        lr      = CONFIG["learning_rate"],
        X_val   = X_test,
        y_val   = Y_test,
        print_every = CONFIG["print_every"],
        batch_size  = CONFIG["batch_size"],
    )

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    final_train_acc = model.accuracy(X_train, Y_train)
    final_test_acc  = model.accuracy(X_test,  Y_test)
    print(f"\n  [OK] Final Train Accuracy : {final_train_acc:.4f}  ({final_train_acc*100:.2f}%)")
    print(f"  [OK] Final Test  Accuracy : {final_test_acc:.4f}  ({final_test_acc*100:.2f}%)")

    # ── 5. Visualize ─────────────────────────────────────────────────────────
    print("\n  Generating plots...")

    # Training curves
    plot_history(
        costs      = model.costs,
        train_accs = model.train_accuracies,
        val_accs   = model.val_accuracies if model.val_accuracies else None,
        save_path  = os.path.join(CONFIG["output_dir"], "training_history.png"),
    )

    # Confusion matrix on test set
    y_pred = model.predict(X_test)
    plot_confusion_matrix(
        y_true    = y_test_int,
        y_pred    = y_pred,
        classes   = [str(i) for i in range(10)],
        save_path = os.path.join(CONFIG["output_dir"], "confusion_matrix.png"),
    )

    # Sample predictions grid
    show_sample_predictions(
        X        = X_test,
        y_true   = y_test_int,
        y_pred   = y_pred,
        n        = 25,
        save_path= os.path.join(CONFIG["output_dir"], "sample_predictions.png"),
    )

    print("\n  Done! All plots saved to ./" + CONFIG["output_dir"])
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
