"""EN3160 A01 - Q1: Piecewise-linear intensity transformation"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def intensity_transform(im, breakpoints):
    """Apply a piecewise-linear intensity transformation to a grayscale image.

    Parameters
    ----------
    im : np.ndarray (uint8, single channel)
    breakpoints : array-like of shape (N, 2)
        Each row is (input_intensity, output_intensity). Must be sorted by
        input intensity and cover the full [0, 255] range.

    Returns
    -------
    np.ndarray (uint8) - transformed image
    lut : np.ndarray (256,) - the lookup table used (for plotting)
    """
    breakpoints = np.asarray(breakpoints, dtype=np.float64)
    xs, ys = breakpoints[:, 0], breakpoints[:, 1]
    full_range = np.arange(256)
    lut = np.interp(full_range, xs, ys).astype(np.uint8)
    return cv2.LUT(im, lut), lut

if __name__ == "__main__":
    img = cv2.imread("/home/claude/en3160_a01/images/q1_emma.jpeg", cv2.IMREAD_GRAYSCALE)

    breakpoints = np.array([
        [0, 0],
        [50, 50],
        [100, 150],
        [150, 255],
        [255, 255],
    ])

    out, lut = intensity_transform(img, breakpoints)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].plot(np.arange(256), lut, color="#8B1A1A", lw=2)
    axes[0].plot(breakpoints[:, 0], breakpoints[:, 1], "ko", ms=4)
    axes[0].set_xlim(0, 255); axes[0].set_ylim(0, 255)
    axes[0].set_xlabel("Input intensity"); axes[0].set_ylabel("Output intensity")
    axes[0].set_title("(a) Intensity transformation")
    axes[0].set_aspect('equal')

    axes[1].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("(b) Original"); axes[1].axis("off")

    axes[2].imshow(out, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("(c) Transformed"); axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("/home/claude/en3160_a01/figures/q1_result.png", dpi=150)
    print("saved q1_result.png")
