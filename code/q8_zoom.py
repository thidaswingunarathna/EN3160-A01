"""EN3160 A01 - Q8: Custom image zoom (nearest-neighbor & bilinear interpolation).

NOTE: the course-provided "two large originals + three zoomed-out versions"
test set was not supplied to this script. As a stand-in, a large image is
downsampled by 4x (by area averaging, to emulate a pre-shrunk small image)
and then the custom zoom function is used to scale it back up by 4x; the
result is compared with the true original via normalized SSD. Swap in the
actual course-provided small/large image pairs before final submission --
the zoom() function itself does not change.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def zoom(im, s, method="bilinear"):
    """Zoom a (grayscale or color) image by factor s in (0, 10].

    method: 'nearest' or 'bilinear'
    """
    assert 0 < s <= 10
    h, w = im.shape[:2]
    new_h, new_w = max(1, round(h * s)), max(1, round(w * s))
    im_f = im.astype(np.float64)
    out_shape = (new_h, new_w) if im.ndim == 2 else (new_h, new_w, im.shape[2])
    out = np.zeros(out_shape, dtype=np.float64)

    # Map output pixel centers back to input coordinates
    ys = (np.arange(new_h) + 0.5) / s - 0.5
    xs = (np.arange(new_w) + 0.5) / s - 0.5

    if method == "nearest":
        yi = np.clip(np.round(ys), 0, h - 1).astype(int)
        xi = np.clip(np.round(xs), 0, w - 1).astype(int)
        out = im_f[yi][:, xi]
    elif method == "bilinear":
        y0 = np.clip(np.floor(ys), 0, h - 1).astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x0 = np.clip(np.floor(xs), 0, w - 1).astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1)
        wy = np.clip(ys - y0, 0, 1)[:, None]
        wx = np.clip(xs - x0, 0, 1)[None, :]

        if im.ndim == 3:
            wy = wy[:, :, None]
            wx = wx[:, :, None]
        Ia = im_f[y0][:, x0]
        Ib = im_f[y0][:, x1]
        Ic = im_f[y1][:, x0]
        Id = im_f[y1][:, x1]
        top = Ia * (1 - wx) + Ib * wx
        bot = Ic * (1 - wx) + Id * wx
        out = top * (1 - wy) + bot * wy
    else:
        raise ValueError("method must be 'nearest' or 'bilinear'")

    return np.clip(out, 0, 255).astype(np.uint8)

def normalized_ssd(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return np.sum((a - b) ** 2) / a.size

if __name__ == "__main__":
    original = cv2.imread("/home/claude/en3160_a01/images/q9_daisy.jpeg")
    h, w = original.shape[:2]
    h4, w4 = h - (h % 4), w - (w % 4)
    original = original[:h4, :w4]

    small = cv2.resize(original, (w4 // 4, h4 // 4), interpolation=cv2.INTER_AREA)

    zoomed_nn = zoom(small, 4.0, method="nearest")
    zoomed_bl = zoom(small, 4.0, method="bilinear")

    ssd_nn = normalized_ssd(zoomed_nn, original)
    ssd_bl = normalized_ssd(zoomed_bl, original)
    print(f"Normalized SSD  nearest-neighbor: {ssd_nn:.3f}")
    print(f"Normalized SSD  bilinear        : {ssd_bl:.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, im, title in zip(
        axes,
        [small, original, zoomed_nn, zoomed_bl],
        ["(a) Small (1/4x)", "(b) Original (ground truth)",
         f"(c) Zoomed x4 nearest\nSSD={ssd_nn:.2f}", f"(d) Zoomed x4 bilinear\nSSD={ssd_bl:.2f}"],
    ):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.axis("off"); ax.set_title(title)
    plt.tight_layout()
    plt.savefig("/home/claude/en3160_a01/figures/q8_result.png", dpi=150)
    print("saved q8_result.png")
