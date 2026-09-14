"""EN3160 A01 - Q10: Bilateral filtering - OpenCV vs Gaussian vs a custom implementation."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

img = cv2.imread("/home/claude/en3160_a01/images/q10_lake.jpeg")
img_f = img.astype(np.float64)

SIGMA_S = 9     # spatial sigma (pixels)
SIGMA_R = 40    # range/intensity sigma
D = 15          # diameter, roughly 3*sigma_s

t0 = time.time()
cv_bilateral = cv2.bilateralFilter(img, d=D, sigmaColor=SIGMA_R, sigmaSpace=SIGMA_S)
t_cv = time.time() - t0

gauss = cv2.GaussianBlur(img, (D, D), sigmaX=SIGMA_S)

def bilateral_filter_manual(im, sigma_s, sigma_r, radius):
    h, w, c = im.shape
    padded = np.pad(im, ((radius, radius), (radius, radius), (0, 0)), mode="reflect").astype(np.float64)
    out = np.zeros_like(im, dtype=np.float64)

    ys, xs = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing="ij")
    spatial_w = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma_s ** 2))  # (k, k)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i + 2 * radius + 1, j:j + 2 * radius + 1, :]     # (k,k,c)
            center = padded[i + radius, j + radius, :]                        # (c,)
            diff = patch - center
            range_w = np.exp(-np.sum(diff ** 2, axis=2) / (2 * sigma_r ** 2))  # (k,k)
            weight = spatial_w * range_w                                       # (k,k)
            wsum = weight.sum()
            out[i, j, :] = (patch * weight[:, :, None]).sum(axis=(0, 1)) / wsum
    return np.clip(out, 0, 255).astype(np.uint8)

# The naive O(H*W*k^2) Python loop is very slow at full resolution, so run
# the custom implementation on a downsized copy for a feasible runtime, and
# compare against cv2's bilateral filter run on the same downsized image.
scale = 0.35
small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
radius = 5
t0 = time.time()
my_bilateral_small = bilateral_filter_manual(small, sigma_s=SIGMA_S * scale, sigma_r=SIGMA_R, radius=radius)
t_mine = time.time() - t0
cv_bilateral_small = cv2.bilateralFilter(small, d=2 * radius + 1, sigmaColor=SIGMA_R, sigmaSpace=SIGMA_S * scale)
gauss_small = cv2.GaussianBlur(small, (2 * radius + 1, 2 * radius + 1), sigmaX=SIGMA_S * scale)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / mse)

ssd_mine_cv = np.sum((my_bilateral_small.astype(np.float64) - cv_bilateral_small.astype(np.float64)) ** 2) / my_bilateral_small.size
psnr_mine_cv = psnr(my_bilateral_small, cv_bilateral_small)
print(f"cv2.bilateralFilter (full res): {t_cv*1000:.1f} ms")
print(f"custom bilateral (small, {radius=}): {t_mine*1000:.1f} ms")
print(f"normalized SSD (mine vs cv2, small): {ssd_mine_cv:.4f}")
print(f"PSNR        (mine vs cv2, small): {psnr_mine_cv:.2f} dB")

fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0, 0].axis("off")
axes[0, 0].set_title("(a) Original")
axes[0, 1].imshow(cv2.cvtColor(gauss, cv2.COLOR_BGR2RGB)); axes[0, 1].axis("off")
axes[0, 1].set_title("(b) Gaussian blur (same kernel size)")
axes[0, 2].imshow(cv2.cvtColor(cv_bilateral, cv2.COLOR_BGR2RGB)); axes[0, 2].axis("off")
axes[0, 2].set_title(f"(c) cv2.bilateralFilter\n{t_cv*1000:.1f} ms")

axes[1, 0].imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)); axes[1, 0].axis("off")
axes[1, 0].set_title("(d) Downsized input (for (e)-(f))")
axes[1, 1].imshow(cv2.cvtColor(cv_bilateral_small, cv2.COLOR_BGR2RGB)); axes[1, 1].axis("off")
axes[1, 1].set_title("(e) cv2.bilateralFilter (small)")
axes[1, 2].imshow(cv2.cvtColor(my_bilateral_small, cv2.COLOR_BGR2RGB)); axes[1, 2].axis("off")
axes[1, 2].set_title(f"(f) Custom bilateral (small)\nPSNR vs (e) = {psnr_mine_cv:.1f} dB")

plt.tight_layout()
plt.savefig("/home/claude/en3160_a01/figures/q10_result.png", dpi=150)
print("saved q10_result.png")
