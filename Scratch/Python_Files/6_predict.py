"""
Phase 6 — Prediction + Filter Visualization UI
================================================
Upload any leaf image → apply all filters → run model → display:
  [Original] [Gaussian] [Sobel X] [Sobel Y] [Sobel Mag] [LoG] [Canny]
  + Prediction overlay with class name + confidence bar

Run: python 6_predict.py --image path/to/leaf.jpg
     python 6_predict.py --image path/to/leaf.jpg --save   (save outputs)
"""

import argparse
import os
import sys
import importlib.util , sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

os.makedirs("outputs", exist_ok=True)

# ─── Dynamic imports (handles numeric-prefix filenames) ───────────────────────
def _import(module_name, filepath):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[module_name] = mod
    return mod


fv        = _import("filter_visualization","2_filter_visualization.py")
pipeline  = _import("pipeline","1_data_pipeline.py")
model_mod = _import("model","3_model.py")

CLASS_LABELS = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
]
CLASS_COLORS = [
    "#2ecc71",  # healthy  → green
    "#e67e22",  # early blight → orange
    "#e74c3c",  # late blight  → red
    "#9b59b6",  # leaf mold    → purple
    "#3498db",  # septoria     → blue
]
IMG_SIZE = (224, 224)


# ─── Predict ──────────────────────────────────────────────────────────────────
def run_prediction(model, img_bgr: np.ndarray):
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE).astype(np.float32) / 255.0
    batch       = np.expand_dims(img_resized, 0)
    probs       = model.predict(batch, verbose=0)[0]
    idx         = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx]), probs


# ─── Main visualization grid ──────────────────────────────────────────────────
def visualize_prediction(img_bgr: np.ndarray, model, save: bool = False):
    """
    Full 2-row visualization:
     Row 1: 7 filter outputs
     Row 2: prediction panel with confidence bars
    """
    img_bgr_resized = cv2.resize(img_bgr, IMG_SIZE)

    # Apply all filters
    results = fv.apply_all_filters(img_bgr_resized)

    # Prediction
    label, confidence, probs = run_prediction(model, img_bgr_resized)
    label_idx = CLASS_LABELS.index(label)
    label_color = CLASS_COLORS[label_idx]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(
        2, 7,
        figure=fig,
        hspace=0.35,
        wspace=0.18,
        top=0.92, bottom=0.08,
        left=0.03, right=0.97,
    )

    # ── Row 1: Filter outputs ──────────────────────────────────────────────────
    filter_keys = [
        "Original",
        "Gaussian Blur\n(manual kernel)",
        "Sobel X\n(manual)",
        "Sobel Y\n(manual)",
        "Sobel Magnitude",
        "Laplacian of Gaussian\n(LoG)",
        "Canny Edges",
    ]
    cmaps = [None, "gray", "RdBu", "RdBu", "hot", "gray", "gray"]

    for col, (key, cmap) in enumerate(zip(filter_keys, cmaps)):
        ax = fig.add_subplot(gs[0, col])
        img_data = results.get(key)
        if img_data is not None:
            if img_data.ndim == 3:
                ax.imshow(img_data)
            else:
                ax.imshow(img_data, cmap=cmap or "gray")
        ax.set_title(key, color="white", fontsize=8, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── Row 2: Prediction panel ────────────────────────────────────────────────
    # Prediction ax spans cols 0-2
    ax_pred = fig.add_subplot(gs[1, 0:3])
    ax_pred.set_facecolor("#0f0f1a")
    ax_pred.set_xlim(0, 1); ax_pred.set_ylim(0, 1)
    ax_pred.axis("off")

    # Large class label
    ax_pred.text(0.5, 0.80, "Prediction", ha="center", va="center",
                 color="#888", fontsize=11, transform=ax_pred.transAxes)
    ax_pred.text(0.5, 0.58, label, ha="center", va="center",
                 color=label_color, fontsize=24, fontweight="bold",
                 transform=ax_pred.transAxes)
    ax_pred.text(0.5, 0.38, f"Confidence: {confidence * 100:.1f}%",
                 ha="center", va="center", color="white", fontsize=14,
                 transform=ax_pred.transAxes)

    # Confidence ring (simple arc via bar chart trick)
    ax_ring = ax_pred.inset_axes([0.72, 0.05, 0.25, 0.55])
    theta   = np.linspace(0, 2 * np.pi * confidence, 100)
    ax_ring.plot(np.cos(theta), np.sin(theta), color=label_color, linewidth=6)
    ax_ring.plot(np.cos(2 * np.pi), np.sin(2 * np.pi), 'o',
                 color=label_color, markersize=8)
    circle_bg = plt.Circle((0, 0), 1, color="#222", zorder=0)
    ax_ring.add_patch(circle_bg)
    ax_ring.set_xlim(-1.3, 1.3); ax_ring.set_ylim(-1.3, 1.3)
    ax_ring.axis("off")
    ax_ring.text(0, 0, f"{confidence*100:.0f}%", ha="center", va="center",
                 color="white", fontsize=11, fontweight="bold")

    # Probability bars (cols 3-6)
    ax_bars = fig.add_subplot(gs[1, 3:7])
    ax_bars.set_facecolor("#0f0f1a")
    y_pos   = np.arange(len(CLASS_LABELS))

    bars = ax_bars.barh(
        y_pos, probs,
        color=[CLASS_COLORS[i] for i in range(len(CLASS_LABELS))],
        alpha=0.85, height=0.6, edgecolor="none",
    )
    for bar, prob in zip(bars, probs):
        ax_bars.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%", va="center", color="white", fontsize=10,
        )

    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(CLASS_LABELS, color="white", fontsize=10)
    ax_bars.set_xlim(0, 1.15)
    ax_bars.set_xlabel("Probability", color="#888", fontsize=10)
    ax_bars.set_title("Class Probabilities", color="white", fontsize=11)
    ax_bars.tick_params(colors="white")
    ax_bars.set_facecolor("#0f0f1a")
    for spine in ax_bars.spines.values():
        spine.set_visible(False)
    ax_bars.xaxis.label.set_color("white")
    ax_bars.tick_params(axis="x", colors="white", labelsize=9)

    fig.suptitle(
        "Automated Plant Disease Classifier — Filter Visualization",
        color="white", fontsize=14, fontweight="bold",
    )

    if save:
        out_path = "outputs/full_prediction_visualization.png"
        plt.savefig(out_path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[✓] Saved {out_path}")

    plt.show()
    print(f"\n[✓] Prediction: {label}  ({confidence * 100:.1f}% confidence)")
    return label, confidence, probs


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease Predictor")
    parser.add_argument("--image", type=str, required=False,
                        help="Path to leaf image (JPG/PNG)")
    parser.add_argument("--model", type=str, default="models/best_model.keras",
                        help="Path to saved model")
    parser.add_argument("--save", action="store_true",
                        help="Save visualization to outputs/")
    args = parser.parse_args()

    # Load model
    import tensorflow as tf
    model_path = Path(args.model)
    if model_path.exists():
        print(f"[✓] Loading model from {model_path}")
        model = tf.keras.models.load_model(str(model_path))
    else:
        print(f"[!] Model not found at {model_path}. Using random weights for demo.")
        from tensorflow.keras import layers, models as km
        inp = layers.Input(shape=(224, 224, 3))
        x   = layers.GlobalAveragePooling2D()(inp)
        out = layers.Dense(5, activation="softmax")(x)
        model = km.Model(inp, out)

    # Load image
    if args.image and Path(args.image).exists():
        img_bgr = cv2.imread(args.image)
    else:
        print("[i] No image specified. Using synthetic leaf image for demo.")
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :, 1] = 100
        img[40:80, 40:80] = [60, 40, 20]
        img[130:180, 100:160] = [50, 30, 10]
        noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    visualize_prediction(img_bgr, model, save=args.save)
