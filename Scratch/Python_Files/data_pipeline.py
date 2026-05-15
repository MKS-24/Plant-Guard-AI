

import os
from pathlib import Path

import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = path = r"C:\Users\User\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1"
print(f"Dataset Path : {path}")
DATA_DIR = Path(path) / "PlantVillage"
IMG_SIZE    = (64, 64)
BATCH_SIZE  = 32
SEED        = 42

CLASSES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
]

CLASS_LABELS = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
]

NUM_CLASSES = len(CLASSES)

# ─── Augmentation generators ──────────────────────────────────────────────────
def make_generators():
    """
    Returns (train_gen, val_gen, test_gen) as Keras ImageDataGenerators.
    Train: heavy augmentation.
    Val/Test: rescale only.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
        validation_split=0.2,   # 20% of training goes to val
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode="categorical",
        subset="training",
        seed=SEED,
        shuffle=True,
    )

    val_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
        shuffle=False,
    )

    # For test set, use a dedicated test split folder if available,
    # otherwise re-use val with shuffle=False
    test_gen = val_test_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode="categorical",
        seed=SEED,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# ─── tf.data pipeline (faster training) ──────────────────────────────────────
def make_tf_datasets(train_gen, val_gen):
    """Convert Keras generators to tf.data.Dataset for prefetching."""
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES),  dtype=tf.float32),
        ),
    ).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES),  dtype=tf.float32),
        ),
    ).prefetch(AUTOTUNE)

    return train_ds, val_ds


# ─── Quick dataset check ──────────────────────────────────────────────────────
def show_samples(gen, n=10):
    """Display n sample images from the generator with class labels."""
    images, labels = next(gen)
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle("Sample Training Images (augmented)", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i >= n:
            break
        ax.imshow(images[i])
        class_idx = np.argmax(labels[i])
        ax.set_title(CLASS_LABELS[class_idx], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/sample_images.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[✓] Saved outputs/sample_images.png")


def class_distribution(gen):
    """Print how many images each class has."""
    print("\nClass distribution:")
    for cls, idx in gen.class_indices.items():
        label = CLASS_LABELS[idx]
        count = sum(1 for f in gen.filenames if f.startswith(cls))
        print(f"  [{idx}] {label:<25} — {count} images")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Phase 1: Data Pipeline ===\n")

    if not DATA_DIR.exists():
        print(f"[!] Dataset not found at '{DATA_DIR}'")
        print("    Download PlantVillage from Kaggle and place the 5 class folders there.")
        print("    Folders needed:")
        for c in CLASSES:
            print(f"      data/PlantVillage/{c}/")
    else:
        train_gen, val_gen, test_gen = make_generators()
        print(f"[✓] Train samples : {train_gen.samples}")
        print(f"[✓] Val   samples : {val_gen.samples}")
        print(f"[✓] Test  samples : {test_gen.samples}")
        class_distribution(train_gen)
        show_samples(train_gen)
