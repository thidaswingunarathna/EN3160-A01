"""EN3160 A01 - Q5: Custom histogram equalization.

NOTE: The assignment's Fig. 5 (a dark, dotted leaf photograph) was not
among the files supplied to this script, so a standard low-contrast
grayscale test image (skimage's `moon`) is used as a stand-in so the code
path can be demonstrated end-to-end. Swap in the real Fig. 5 image before
final submission -- the function itself does not change.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalize(im):
    """Custom histogram equalization for an 8-bit grayscale image."""
    hist = np.bincount(im.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min()
    total = im.size
    lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255).clip(0, 255).astype(np.uint8)
    return lut[im], hist, cdf, lut

if __name__ == "__main__":
    img = cv2.imread("/home/claude/en3160_a01/images/q5_placeholder.jpeg", cv2.IMREAD_GRAYSCALE)
    out, hist, cdf, lut = histogram_equalize(img)
    hist_eq = np.bincount(out.ravel(), minlength=256)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(img, cmap="gray"); axes[0, 0].axis("off")
    axes[0, 0].set_title("(a) Original")
    axes[0, 1].imshow(out, cmap="gray"); axes[0, 1].axis("off")
    axes[0, 1].set_title("(b) Histogram-equalized")
    axes[0, 2].plot(lut, color="#8B1A1A"); axes[0, 2].set_title("(c) LUT (from CDF)")
    axes[0, 2].set_xlim(0, 255); axes[0, 2].set_ylim(0, 255)
    axes[0, 2].set_xlabel("Input"); axes[0, 2].set_ylabel("Output")

    axes[1, 0].bar(np.arange(256), hist, color="#555", width=1)
    axes[1, 0].set_title("(d) Original histogram")
    axes[1, 1].bar(np.arange(256), hist_eq, color="#8B1A1A", width=1)
    axes[1, 1].set_title("(e) Equalized histogram")
    axes[1, 2].plot(cdf / cdf.max(), color="#2A6F97")
    axes[1, 2].set_title("(f) Original CDF (normalized)")
    axes[1, 2].set_xlim(0, 255)

    plt.tight_layout()
    plt.savefig("/home/claude/en3160_a01/figures/q5_result.png", dpi=150)
    print("saved q5_result.png")
