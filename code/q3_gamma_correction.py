"""EN3160 A01 - Q3: Gamma correction on the L plane in L*a*b* space."""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("/home/claude/en3160_a01/images/q3_gamma.jpeg")
lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
L, a, b = cv2.split(lab)

GAMMA = 0.6  # < 1 brightens shadows / recovers detail in the dark rocks

L_norm = L.astype(np.float64) / 255.0
L_corrected = np.clip((L_norm ** GAMMA) * 255.0, 0, 255).astype(np.uint8)

lab_corrected = cv2.merge([L_corrected, a, b])
img_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); axes[0, 0].axis("off")
axes[0, 0].set_title("(a) Original")
axes[0, 1].imshow(cv2.cvtColor(img_corrected, cv2.COLOR_BGR2RGB)); axes[0, 1].axis("off")
axes[0, 1].set_title(f"(b) Gamma corrected ($\\gamma$ = {GAMMA})")

xs = np.linspace(0, 255, 256)
ys = np.clip(((xs / 255.0) ** GAMMA) * 255.0, 0, 255)
axes[0, 2].plot(xs, ys, color="#8B1A1A", lw=2)
axes[0, 2].plot([0, 255], [0, 255], "k--", lw=1, alpha=0.4)
axes[0, 2].set_xlim(0, 255); axes[0, 2].set_ylim(0, 255)
axes[0, 2].set_xlabel("Input L"); axes[0, 2].set_ylabel("Output L")
axes[0, 2].set_title(f"(c) $L' = 255(L/255)^{{{GAMMA}}}$")

axes[1, 0].hist(L.ravel(), bins=256, range=(0, 255), color="#555")
axes[1, 0].set_title("(d) Original L histogram")
axes[1, 1].hist(L_corrected.ravel(), bins=256, range=(0, 255), color="#8B1A1A")
axes[1, 1].set_title("(e) Corrected L histogram")

axes[1, 2].hist(L.ravel(), bins=256, range=(0, 255), color="#555", alpha=0.5, label="original")
axes[1, 2].hist(L_corrected.ravel(), bins=256, range=(0, 255), color="#8B1A1A", alpha=0.5, label="corrected")
axes[1, 2].legend(); axes[1, 2].set_title("(f) Overlaid")

plt.tight_layout()
plt.savefig("/home/claude/en3160_a01/figures/q3_result.png", dpi=150)
print("saved q3_result.png, gamma =", GAMMA)
