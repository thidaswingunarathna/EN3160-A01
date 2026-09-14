"""EN3160 A01 - Q7: Sobel filtering (filter2D, manual convolution, separable)."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

img = cv2.imread("/home/claude/en3160_a01/images/q7_einstein.jpeg", cv2.IMREAD_GRAYSCALE).astype(np.float64)

Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
Gy = Gx.T

# (a) cv2.filter2D
t0 = time.time()
sx_a = cv2.filter2D(img, -1, Gx)
sy_a = cv2.filter2D(img, -1, Gy)
mag_a = np.hypot(sx_a, sy_a)
t_a = time.time() - t0

# (b) Manual 2D convolution ("from scratch")
def conv2d_manual(im, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(im, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(im)
    k = np.flipud(np.fliplr(kernel))  # true convolution flips the kernel
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            region = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(region * k)
    return out

t0 = time.time()
sx_b = conv2d_manual(img, Gx)
sy_b = conv2d_manual(img, Gy)
mag_b = np.hypot(sx_b, sy_b)
t_b = time.time() - t0

# (c) Separable convolution: Gx = [1,2,1]^T * [1,0,-1], Gy = [1,0,-1]^T * [1,2,1]
def conv1d_axis(im, kernel, axis):
    k = kernel[::-1]
    pad = len(kernel) // 2
    if axis == 0:
        padded = np.pad(im, ((pad, pad), (0, 0)), mode="reflect")
        out = np.zeros_like(im)
        for i in range(im.shape[0]):
            out[i, :] = sum(k[m] * padded[i + m, :] for m in range(len(k)))
        return out
    else:
        padded = np.pad(im, ((0, 0), (pad, pad)), mode="reflect")
        out = np.zeros_like(im)
        for j in range(im.shape[1]):
            out[:, j] = sum(k[m] * padded[:, j + m] for m in range(len(k)))
        return out

t0 = time.time()
smooth_col = np.array([1, 2, 1], dtype=np.float64)
deriv_row = np.array([1, 0, -1], dtype=np.float64)
sx_c = conv1d_axis(conv1d_axis(img, smooth_col, axis=0), deriv_row, axis=1)
sy_c = conv1d_axis(conv1d_axis(img, deriv_row, axis=0), smooth_col, axis=1)
mag_c = np.hypot(sx_c, sy_c)
t_c = time.time() - t0

print(f"filter2D: {t_a*1000:.2f} ms | manual 2D conv: {t_b*1000:.2f} ms | separable: {t_c*1000:.2f} ms")
print("max abs diff filter2D vs manual :", np.max(np.abs(mag_a - mag_b)))
print("max abs diff filter2D vs separable:", np.max(np.abs(mag_a - mag_c)))

fig, axes = plt.subplots(1, 4, figsize=(17, 5))
axes[0].imshow(img, cmap="gray"); axes[0].axis("off"); axes[0].set_title("(a) Original")
axes[1].imshow(mag_a, cmap="gray"); axes[1].axis("off"); axes[1].set_title(f"(b) cv2.filter2D\n{t_a*1000:.1f} ms")
axes[2].imshow(mag_b, cmap="gray"); axes[2].axis("off"); axes[2].set_title(f"(c) Manual conv.\n{t_b*1000:.1f} ms")
axes[3].imshow(mag_c, cmap="gray"); axes[3].axis("off"); axes[3].set_title(f"(d) Separable conv.\n{t_c*1000:.1f} ms")
plt.tight_layout()
plt.savefig("/home/claude/en3160_a01/figures/q7_result.png", dpi=150)
print("saved q7_result.png")
