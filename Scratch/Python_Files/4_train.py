"""
Phase 4 — Training
===================
Trains the CNN with:
 - Adam optimizer
 - ReduceLROnPlateau (patience=3, factor=0.5)
 - EarlyStopping (patience=5, restore best weights)
 - ModelCheckpoint (save best val_accuracy)
 - Training history plots

Run: python 4_train.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

from data_pipeline import make_generators, make_tf_datasets, CLASSES, NUM_CLASSES
from model import build_model, compile_model

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS      = 5
LR_INITIAL  = 1e-3
BATCH_SIZE  = 64


# ─── Callbacks ────────────────────────────────────────────────────────────────
def make_callbacks():
    return [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # Reduce LR when val_loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        # Stop early if no improvement
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        # Log to CSV
        tf.keras.callbacks.CSVLogger("outputs/training_log.csv"),
        # TensorBoard (optional)
        tf.keras.callbacks.TensorBoard(
            log_dir="outputs/tensorboard_logs",
            histogram_freq=1,
        ),
    ]


# ─── Plot training history ────────────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    metrics = [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("loss",     "val_loss",     "Loss"),
        ("precision","val_precision","Precision"),
    ]
    for ax, (train_key, val_key, title) in zip(axes, metrics):
        if train_key in history:
            ax.plot(history[train_key], label=f"Train {title}", linewidth=2)
        if val_key in history:
            ax.plot(history[val_key], label=f"Val {title}",   linewidth=2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/training_history.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[✓] Saved outputs/training_history.png")


# ─── Main training loop ───────────────────────────────────────────────────────
def train():
    print("=== Phase 4: Training ===\n")

    # Build data generators
    train_gen, val_gen, test_gen = make_generators()

    # Build model
    model = build_model(num_classes=NUM_CLASSES)
    compile_model(model, learning_rate=LR_INITIAL)
    model.summary()

    # Class weights to handle imbalance
    BASE_PATH = r"C:\Users\User\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage"
    total = sum(len(list(Path(f"{BASE_PATH}/{c}").glob("*.jpg")))
                for c in CLASSES if Path(f"{BASE_PATH}/{c}").exists())
    class_weights = {}
    for i, cls in enumerate(CLASSES):
        p = Path(f"{BASE_PATH}/{cls}")
        n = len(list(p.glob("*.jpg"))) if p.exists() else 1
        class_weights[i] = (total / (NUM_CLASSES * n)) if n > 0 else 1.0

    print(f"\nClass weights: {class_weights}\n")

    # Train
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=make_callbacks(),
        class_weight=class_weights,
        verbose=1,
    )

    # Save history
    with open("outputs/history.json", "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print("[✓] Saved outputs/history.json")

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    test_loss, test_acc, *rest = model.evaluate(test_gen, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    # Plot
    plot_history(history.history)

    return model, history


if __name__ == "__main__":
    # Allow importing 1_data_pipeline.py despite the numeric prefix
    import importlib.util, sys
    for module, path in [("datapipeline", "data_pipeline.py"),
                         ("model",          "model.py")]:
        spec = importlib.util.spec_from_file_location(module, path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[module] = mod

    from data_pipeline import make_generators, make_tf_datasets, CLASSES, NUM_CLASSES
    from model import build_model, compile_model

    model, history = train()
