"""
Phase 2 — Filter Visualization
================================
Applies Sobel, Laplacian, Gaussian, and manual kernels to leaf images.
Visualizes texture patterns that help distinguish disease classes.

Key Tasks (from spec):
 - Gaussian blur (NumPy kernel)
 - Sobel edge detection (cv2 + manual kernel)
 - Laplacian of Gaussian
 - Manual kernel convolution from scratch (NumPy)
 - Visualize filter responses across disease classes

Run: python 2_filter_visualization.py --image path/to/leaf.jpg
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import os
import argparse

os.makedirs("outputs", exist_ok=True)

# ─── Manual kernel convolution (NumPy, from scratch) ─────────────────────────
def manual_convolve(image_gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Pure NumPy 2D convolution (no cv2.filter2D).
    Pads image with zeros (same padding).
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image_gray, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)

    out = np.zeros_like(image_gray, dtype=np.float32)
    for i in range(image_gray.shape[0]):
        for j in range(image_gray.shape[1]):
            region = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(region * kernel)

    return out


def fast_convolve(image_gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Vectorized convolution using stride tricks (much faster for large images)."""
    from numpy.lib.stride_tricks import sliding_window_view
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image_gray.astype(np.float32),
                    ((ph, ph), (pw, pw)), mode="constant")
    windows = sliding_window_view(padded, (kh, kw))
    return (windows * kernel).sum(axis=(-2, -1))


# ─── Filter bank ─────────────────────────────────────────────────────────────
def gaussian_kernel(size=5, sigma=1.0) -> np.ndarray:
    """Build a Gaussian blur kernel from scratch."""
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()


SOBEL_X = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)

SOBEL_Y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float32)

LAPLACIAN = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]], dtype=np.float32)

SHARPEN = np.array([[ 0, -1,  0],
                     [-1,  5, -1],
                     [ 0, -1,  0]], dtype=np.float32)

EMBOSS = np.array([[-2, -1, 0],
                   [-1,  1, 1],
                   [ 0,  1, 2]], dtype=np.float32)


# ─── Apply all filters to a single image ─────────────────────────────────────
def apply_all_filters(img_bgr: np.ndarray) -> dict:
    """
    Given a BGR OpenCV image, return a dict of {filter_name: result_image}.
    All results are uint8 for display.
    """
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    results = {"Original": img_rgb}

    # 1. Gaussian blur (manual kernel)
    g_kernel = gaussian_kernel(size=9, sigma=2.0)
    g_blur   = fast_convolve(img_gray, g_kernel)
    results["Gaussian Blur\n(manual kernel)"] = np.clip(g_blur, 0, 1)

    # 2. Gaussian blur via OpenCV (for comparison)
    g_ocv = cv2.GaussianBlur(img_gray, (9, 9), sigmaX=2.0)
    results["Gaussian Blur\n(OpenCV)"] = g_ocv

    # 3. Sobel X (manual)
    sx = fast_convolve(img_gray, SOBEL_X)
    results["Sobel X\n(manual)"] = np.abs(sx)

    # 4. Sobel Y (manual)
    sy = fast_convolve(img_gray, SOBEL_Y)
    results["Sobel Y\n(manual)"] = np.abs(sy)

    # 5. Sobel magnitude
    mag = np.sqrt(sx ** 2 + sy ** 2)
    results["Sobel Magnitude"] = mag / mag.max() if mag.max() > 0 else mag

    # 6. Sobel via OpenCV
    sx_ocv = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sy_ocv = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag_ocv = np.sqrt(sx_ocv ** 2 + sy_ocv ** 2)
    results["Sobel Magnitude\n(OpenCV)"] = mag_ocv / mag_ocv.max()

    # 7. Laplacian (manual)
    lap = fast_convolve(img_gray, LAPLACIAN)
    results["Laplacian\n(manual)"] = np.abs(lap)

    # 8. Laplacian of Gaussian (LoG)
    blurred = fast_convolve(img_gray, gaussian_kernel(size=5, sigma=1.4))
    log     = fast_convolve(blurred, LAPLACIAN)
    results["Laplacian of Gaussian\n(LoG)"] = np.abs(log)

    # 9. Canny edge detection
    gray_u8 = (img_gray * 255).astype(np.uint8)
    canny   = cv2.Canny(gray_u8, threshold1=50, threshold2=150)
    results["Canny Edges"] = canny / 255.0

    # 10. Sharpen (manual)
    sharp = fast_convolve(img_gray, SHARPEN)
    results["Sharpen\n(manual)"] = np.clip(sharp, 0, 1)

    # 11. Emboss (manual)
    emb = fast_convolve(img_gray, EMBOSS)
    results["Emboss\n(manual)"] = np.clip(emb + 0.5, 0, 1)

    return results


# ─── Plot filter grid ─────────────────────────────────────────────────────────
def plot_filter_grid(results: dict, title: str = "Filter Visualization", save_path: str = None):
    """Display all filter outputs in a grid."""
    n     = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(16, nrows * 4))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.4, wspace=0.3)

    for idx, (name, img) in enumerate(results.items()):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[✓] Saved {save_path}")
    plt.show()


# ─── Visualize one filter across multiple disease classes ─────────────────────
def compare_filter_across_classes(class_image_paths: dict, filter_name: str = "Sobel Magnitude"):
    """
    Show a single filter's output across different disease classes.
    class_image_paths: {class_label: image_path}
    """
    n = len(class_image_paths)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle(f"Filter: {filter_name} across disease classes", fontsize=13)

    for i, (label, path) in enumerate(class_image_paths.items()):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        results = apply_all_filters(img)

        axes[0, i].imshow(results["Original"])
        axes[0, i].set_title(label, fontsize=9)
        axes[0, i].axis("off")

        if filter_name in results:
            fimg = results[filter_name]
            axes[1, i].imshow(fimg, cmap="gray" if fimg.ndim == 2 else None)
        axes[1, i].set_title(filter_name, fontsize=8)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/filter_comparison_classes.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[✓] Saved outputs/filter_comparison_classes.png")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Visualization")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single leaf image")
    args = parser.parse_args()

    if args.image:
        img_path = args.image
    else:
        # Try to find any image in the dataset
        from pathlib import Path
        candidates = list(Path("data/PlantVillage").rglob("*.jpg"))[:1]
        if not candidates:
            print("[!] No image found. Pass --image path/to/leaf.jpg")
            print("    Generating a synthetic test image instead...")
            # Create a synthetic green leaf-like image for demo
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[:, :, 1] = 120  # green base
            img[50:80, 50:80, :] = [80, 50, 20]   # brown spot (disease sim)
            img[120:160, 100:150, :] = [60, 40, 10]
            # Add some texture noise
            noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.imread(str(candidates[0]))
    
    if "img_bgr" not in dir():
        img_bgr = cv2.imread(str(args.image) if args.image else str(candidates[0]))
        img_bgr = cv2.resize(img_bgr, (224, 224))

    print("=== Phase 2: Filter Visualization ===\n")
    results = apply_all_filters(img_bgr)
    plot_filter_grid(results,
                     title="Plant Leaf — Filter Bank Output",
                     save_path="outputs/filter_grid.png")
    print("[✓] All filters applied and visualized.")
    print(f"[i] Filters available: {list(results.keys())}")
