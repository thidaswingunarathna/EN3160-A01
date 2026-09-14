"""EN3160 A01 - Q9: grabCut foreground segmentation + background blur (bokeh)."""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("/home/claude/en3160_a01/images/q9_daisy.jpeg")
h, w = img.shape[:2]

mask = np.zeros((h, w), np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
rect = (int(0.06 * w), int(0.02 * h), int(0.90 * w), int(0.85 * h))

cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")

foreground = img * binary_mask[:, :, None]
background_only = img * (1 - binary_mask[:, :, None])  # zeros where the flower was
mask_f = binary_mask.astype(np.float64)[:, :, None]

# Correct approach: blur the *full* image, then composite with the mask.
blurred_full = cv2.GaussianBlur(img, (0, 0), sigmaX=15)
enhanced_correct = (img.astype(np.float64) * mask_f + blurred_full.astype(np.float64) * (1 - mask_f)).astype(np.uint8)

# Naive approach: blur the already-masked-out background (which has hard
# black zeros where the flower used to be) then composite. The Gaussian
# kernel mixes in those zeros near the boundary, pulling the blurred
# background dark in a halo just outside the flower's silhouette.
blurred_naive = cv2.GaussianBlur(background_only, (0, 0), sigmaX=15)
enhanced_naive = (img.astype(np.float64) * mask_f + blurred_naive.astype(np.float64) * (1 - mask_f)).astype(np.uint8)

fig, axes = plt.subplots(3, 3, figsize=(13, 12.5))
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0, 0].axis("off")
axes[0, 0].set_title("(a) Original")
rect_vis = img.copy()
cv2.rectangle(rect_vis, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 3)
axes[0, 1].imshow(cv2.cvtColor(rect_vis, cv2.COLOR_BGR2RGB)); axes[0, 1].axis("off")
axes[0, 1].set_title("(b) grabCut init rectangle")
axes[0, 2].imshow(binary_mask, cmap="gray"); axes[0, 2].axis("off")
axes[0, 2].set_title("(c) Segmentation mask")

axes[1, 0].imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)); axes[1, 0].axis("off")
axes[1, 0].set_title("(d) Foreground")
axes[1, 1].imshow(cv2.cvtColor(background_only, cv2.COLOR_BGR2RGB)); axes[1, 1].axis("off")
axes[1, 1].set_title("(e) Background")
axes[1, 2].imshow(cv2.cvtColor(enhanced_correct, cv2.COLOR_BGR2RGB)); axes[1, 2].axis("off")
axes[1, 2].set_title("(f) Bokeh - blur whole image first (correct)")

axes[2, 0].imshow(cv2.cvtColor(enhanced_naive, cv2.COLOR_BGR2RGB)); axes[2, 0].axis("off")
axes[2, 0].set_title("(g) Bokeh - blur masked background (naive)")
# Zoomed crops around the flower boundary to make the halo obvious
y0, y1, x0, x1 = int(0.28*h), int(0.42*h), int(0.05*w), int(0.30*w)
axes[2, 1].imshow(cv2.cvtColor(enhanced_correct[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
axes[2, 1].axis("off"); axes[2, 1].set_title("(h) Zoom of (f) - no halo")
axes[2, 2].imshow(cv2.cvtColor(enhanced_naive[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
axes[2, 2].axis("off"); axes[2, 2].set_title("(i) Zoom of (g) - dark halo")

plt.tight_layout()
plt.savefig("/home/claude/en3160_a01/figures/q9_result.png", dpi=150)
print("saved q9_result.png")
