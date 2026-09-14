"""EN3160 A01 - Q6: Histogram-equalize only the foreground of an image."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from q5_histogram_equalization import histogram_equalize

img_bgr = cv2.imread("/home/claude/en3160_a01/images/q6_jlaw.jpeg")
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# The background is a low-saturation neutral-gray gradient, while the hair
# and skin are considerably more saturated -> threshold the S plane.
_, mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_val, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
foreground = cv2.bitwise_and(gray, gray, mask=mask)
background_mask = cv2.bitwise_not(mask)
background = cv2.bitwise_and(gray, gray, mask=background_mask)

# Histogram + cumsum of the foreground pixels only (ignore the masked-out zeros)
fg_pixels = gray[mask > 0]
hist = np.bincount(fg_pixels, minlength=256).astype(np.float64)
cdf = np.cumsum(hist)
cdf_min = cdf[cdf > 0].min()
lut = np.round((cdf - cdf_min) / (fg_pixels.size - cdf_min) * 255).clip(0, 255).astype(np.uint8)

fg_eq = cv2.LUT(gray, lut)
fg_eq = cv2.bitwise_and(fg_eq, fg_eq, mask=mask)
result = cv2.add(background, fg_eq)

fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
axes[0, 0].imshow(h, cmap="gray"); axes[0, 0].axis("off"); axes[0, 0].set_title("(a) Hue")
axes[0, 1].imshow(s, cmap="gray"); axes[0, 1].axis("off"); axes[0, 1].set_title("(b) Saturation")
axes[0, 2].imshow(v, cmap="gray"); axes[0, 2].axis("off"); axes[0, 2].set_title("(c) Value")
axes[0, 3].imshow(mask, cmap="gray"); axes[0, 3].axis("off")
axes[0, 3].set_title(f"(d) Foreground mask (Otsu on S, t={otsu_val:.0f})")

axes[1, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); axes[1, 0].axis("off")
axes[1, 0].set_title("(e) Original")
axes[1, 1].imshow(foreground, cmap="gray"); axes[1, 1].axis("off")
axes[1, 1].set_title("(f) Foreground only")
axes[1, 2].bar(np.arange(256), hist, color="#555", width=1)
axes[1, 2].set_title("(g) Foreground histogram")
axes[1, 3].imshow(result, cmap="gray"); axes[1, 3].axis("off")
axes[1, 3].set_title("(h) Foreground histogram-equalized")

plt.tight_layout()
plt.savefig("/home/claude/en3160_a01/figures/q6_result.png", dpi=150)
print("saved q6_result.png, otsu threshold =", otsu_val)
