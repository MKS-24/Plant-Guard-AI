"""
Phase 5 — Evaluation
=====================
Key Tasks (from spec):
 - Build and compare CNN from scratch vs any pretrained model: report time for all three
 - Run on 20 unseen leaf images
 - Annotate each with predicted disease class and confidence score using cv2
 - Display the Sobel filter response grid across 5 disease classes
 - Discuss what texture differences are visible

Run: python 5_evaluate.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
import tensorflow as tf

os.makedirs("outputs/annotated", exist_ok=True)

CLASS_LABELS = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
]
IMG_SIZE = (64, 64)


# ─── Load saved model ─────────────────────────────────────────────────────────
def load_model(path: str = "models/best_model.keras") -> tf.keras.Model:
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found at {path}. Run 4_train.py first.")
    return tf.keras.models.load_model(path)


# ─── Preprocess a single image ────────────────────────────────────────────────
def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    return img_resized.astype(np.float32) / 255.0


# ─── Predict single image ─────────────────────────────────────────────────────
def predict(model: tf.keras.Model, img_bgr: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Returns (class_label, confidence, all_probabilities)."""
    img = preprocess_image(img_bgr)
    batch = np.expand_dims(img, 0)
    probs = model.predict(batch, verbose=0)[0]
    idx   = np.argmax(probs)
    return CLASS_LABELS[idx], float(probs[idx]), probs


# ─── Annotate image with OpenCV ───────────────────────────────────────────────
def annotate_image(img_bgr: np.ndarray,
                   label: str,
                   confidence: float,
                   probs: np.ndarray) -> np.ndarray:
    """
    Draw prediction label, confidence bar, and per-class probability bars
    onto the image using OpenCV drawing functions.
    """
    h, w = img_bgr.shape[:2]
    annotated = img_bgr.copy()

    # Confidence color: green if healthy, red if diseased
    color = (0, 200, 0) if label == "Healthy" else (0, 60, 220)

    # Top banner
    cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(annotated, f"{label}  ({confidence*100:.1f}%)",
                (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # Probability bars at bottom
    bar_h    = 12
    bar_area = 18 * len(CLASS_LABELS) + 10
    cv2.rectangle(annotated, (0, h - bar_area), (w, h), (20, 20, 20), -1)

    for i, (cls, prob) in enumerate(zip(CLASS_LABELS, probs)):
        y   = h - bar_area + 8 + i * 18
        bar_w = int(prob * (w - 90))
        bar_c = (0, 180, 0) if cls == "Healthy" else (0, 80, 200)
        cv2.rectangle(annotated, (85, y), (85 + bar_w, y + bar_h), bar_c, -1)
        cv2.putText(annotated, f"{cls[:8]}", (2, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
        cv2.putText(annotated, f"{prob*100:.1f}%", (85 + bar_w + 3, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

    return annotated


# ─── Run on 20 unseen images ─────────────────────────────────────────────────
def evaluate_unseen_images(model: tf.keras.Model,
                            image_paths: list,
                            true_labels: list = None) -> dict:
    """
    Predict on up to 20 images, annotate with cv2, save to outputs/annotated/.
    Returns dict with y_true, y_pred, confidences.
    """
    print(f"\n--- Evaluating {len(image_paths)} unseen images ---")
    y_pred, y_conf = [], []

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle("Predictions on 20 Unseen Leaf Images", fontsize=14)

    for i, path in enumerate(image_paths[:20]):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, (224,224))

        label, conf, probs = predict(model, img_bgr)
        y_pred.append(CLASS_LABELS.index(label))
        y_conf.append(conf)

        # Annotate and save
        annotated = annotate_image(img_bgr, label, conf, probs)
        out_path  = f"outputs/annotated/pred_{i:02d}_{label.replace(' ', '_')}.jpg"
        cv2.imwrite(out_path, annotated)

        # Show in grid
        ax = axes[i // 5, i % 5]
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{label}\n{conf*100:.1f}%", fontsize=8,
                     color="green" if label == "Healthy" else "red")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/annotated_predictions_grid.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("[✓] Saved outputs/annotated_predictions_grid.png")
    print(f"[✓] Individual annotated images in outputs/annotated/")

    return {
        "y_true": true_labels,
        "y_pred": y_pred,
        "confidences": y_conf,
    }


# ─── Confusion matrix ─────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true: list, y_pred: list):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Raw Counts", "Normalized (row %)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="YlOrRd",
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
            linewidths=0.5, ax=ax, cbar=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[✓] Saved outputs/confusion_matrix.png")


# ─── Sobel texture comparison across classes ──────────────────────────────────
def plot_sobel_across_classes(class_image_paths: dict):
    """
    Show original + Sobel magnitude for one image per disease class.
    Discuss texture differences in console output.
    """
    from filter_visualization import fast_convolve, SOBEL_X, SOBEL_Y

    n   = len(class_image_paths)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    fig.suptitle("Sobel Filter Response Across Disease Classes\n"
                 "(reveals texture patterns unique to each disease)", fontsize=12)

    for i, (label, path) in enumerate(class_image_paths.items()):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img    = cv2.resize(img, IMG_SIZE)
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        sx     = fast_convolve(gray, SOBEL_X)
        sy     = fast_convolve(gray, SOBEL_Y)
        mag    = np.sqrt(sx ** 2 + sy ** 2)
        mag    = mag / mag.max() if mag.max() > 0 else mag

        # LoG
        from filter_visualization import fast_convolve, gaussian_kernel, LAPLACIAN
        blurred = fast_convolve(gray, gaussian_kernel(5, 1.4))
        log     = np.abs(fast_convolve(blurred, LAPLACIAN))
        log     = log / log.max() if log.max() > 0 else log

        axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(label, fontsize=9, fontweight="bold")
        axes[0, i].axis("off")

        axes[1, i].imshow(mag, cmap="hot")
        axes[1, i].set_title("Sobel Magnitude", fontsize=8)
        axes[1, i].axis("off")

        axes[2, i].imshow(log, cmap="gray")
        axes[2, i].set_title("Laplacian of Gaussian", fontsize=8)
        axes[2, i].axis("off")

    row_labels = ["Original", "Sobel Magnitude", "LoG"]
    for ax, lbl in zip(axes[:, 0], row_labels):
        ax.set_ylabel(lbl, fontsize=10, rotation=90, labelpad=10)

    plt.tight_layout()
    plt.savefig("outputs/sobel_class_comparison.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[✓] Saved outputs/sobel_class_comparison.png")

    print("\n=== Texture Analysis — What the filters reveal ===")
    print("""
  Healthy:           Smooth, uniform Sobel response. Low edge density.
                     LoG shows clean leaf venation only.

  Early Blight:      Concentric ring patterns. Sobel edges cluster around
                     necrotic rings with yellow halos.

  Late Blight:       Large dark water-soaked lesions. High-intensity Sobel
                     edges at lesion boundaries. LoG captures lesion texture.

  Leaf Mold:         Pale green/yellow upper surface, olive-green to gray
                     fuzzy patches on underside. Diffuse Sobel response.

  Septoria Leaf Spot: Small circular spots (1-2mm) with dark borders.
                     Very high Sobel edge density, many small rings.
                     LoG shows dense spot pattern clearly.
    """)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import importlib.util, sys

    for module, path in [("filtervisualization", "filter_visualization.py")]:
        spec = importlib.util.spec_from_file_location(module, path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[module] = mod

    print("=== Phase 5: Evaluation ===\n")

    try:
        model = load_model()
    except FileNotFoundError as e:
        print(f"[!] {e}")
        print("    Creating a dummy model for demo purposes...")
        import tensorflow as tf
        from tensorflow.keras import layers, models
        inputs = layers.Input(shape=(64, 64, 3))
        x = layers.GlobalAveragePooling2D()(inputs)
        x = layers.Dense(5, activation="softmax")(x)
        model = models.Model(inputs, x)
        model.compile(optimizer="adam", loss="categorical_crossentropy")

    # Gather test images
    DATA_DIR = Path(r"C:\Users\User\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage")
    CLASSES  = [
        "Tomato_healthy",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
    ]

    test_images, true_labels, class_images_for_sobel = [], [], {}
    for cls_idx, cls in enumerate(CLASSES):
        cls_path = DATA_DIR / cls
        if not cls_path.exists():
            continue
        imgs = sorted(cls_path.glob("*.jpg"))
        # Use last 4 images per class as "unseen"
        for img in imgs[-4:]:
            test_images.append(img)
            true_labels.append(cls_idx)
        if imgs:
            class_images_for_sobel[CLASS_LABELS[cls_idx]] = imgs[-1]

    if test_images:
        results = evaluate_unseen_images(model, test_images, true_labels)
        if results["y_true"] and results["y_pred"]:
            y_true = results["y_true"]
            y_pred = results["y_pred"]
            plot_confusion_matrix(y_true, y_pred)
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

        if class_images_for_sobel:
            plot_sobel_across_classes(class_images_for_sobel)
    else:
        print("[!] No test images found. Showing filter discussion only.")
        print("\nRun this after downloading the PlantVillage dataset.")
