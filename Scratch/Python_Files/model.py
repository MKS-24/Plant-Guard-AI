"""
Phase 3 — CNN Architecture (from scratch, no pretrained weights)
=================================================================
Key Spec Requirements:
 - Key pretrained model EXCEPT YOLO frozen feature extraction (NOT used)
 - Built from scratch: 3 Conv blocks + BatchNorm + Dropout + Dense head
 - ReduceLROnPlateau scheduler
 - First Conv2D for healthy vs diseased leaf classification

Architecture:
  Input (224, 224, 3)
    ↓
  [Conv Block 1] Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
    ↓
  [Conv Block 2] Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
    ↓
  [Conv Block 3] Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.40)
    ↓
  Flatten → Dense(256, ReLU) → Dropout(0.5)
    ↓
  Dense(5, Softmax)   ← 5 disease classes
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import plot_model
import os


NUM_CLASSES = 5
IMG_SIZE    = (64, 64, 3)


def conv_block(x, filters: int, dropout_rate: float = 0.25, l2: float = 1e-4):
    """
    A single convolutional block:
      Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool → Dropout
    """
    x = layers.Conv2D(
        filters, (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(l2),
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters, (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(l2),
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_model(num_classes: int = NUM_CLASSES,
                input_shape: tuple = IMG_SIZE,
                l2: float = 1e-4) -> tf.keras.Model:
    """
    Build CNN from scratch.
    3 conv blocks + dense head.
    """
    inputs = layers.Input(shape=input_shape, name="leaf_input")

    # ── Block 1: learn basic edges/colors ─────────────────────────────────────
    x = conv_block(inputs, filters=32, dropout_rate=0.25, l2=l2)   # → 112×112×32

    # ── Block 2: learn mid-level texture patterns ──────────────────────────────
    x = conv_block(x, filters=64, dropout_rate=0.25, l2=l2)        # → 56×56×64

    # ── Block 3: learn high-level disease signatures ───────────────────────────
    x = conv_block(x, filters=128, dropout_rate=0.40, l2=l2)       # → 28×28×128

    # ── Dense head ────────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)        # 28×28×128 → 128  (better than Flatten)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2),
        name="dense_features",
    )(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="disease_class")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PlantDiseaseNet")
    return model


def compile_model(model: tf.keras.Model,
                  learning_rate: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Phase 3: CNN Architecture ===\n")
    model = build_model()
    compile_model(model)
    model.summary()

    os.makedirs("outputs", exist_ok=True)
    try:
        plot_model(model, to_file="outputs/model_architecture.png",
                   show_shapes=True, show_layer_names=True, dpi=100)
        print("\n[✓] Saved outputs/model_architecture.png")
    except Exception:
        print("[i] Install pydot + graphviz to generate model diagram.")

    total_params = model.count_params()
    print(f"\n[✓] Total parameters: {total_params:,}")
    print(f"[✓] Trainable:        {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")
