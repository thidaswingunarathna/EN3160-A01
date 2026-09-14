"""EN3160 A01 - Q4: Vibrance enhancement via a Gaussian bump on the saturation plane."""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("/home/claude/en3160_a01/images/q4_vibrance.jpeg")
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

A = 0.7
SIGMA = 70

x = np.arange(256, dtype=np.float64)
f = np.minimum(x + A * 128 * np.exp(-((x - 128) ** 2) / (2 * SIGMA ** 2)), 255).astype(np.uint8)

s_new = cv2.LUT(s, f)
hsv_new = cv2.merge([h, s_new, v])
img_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); axes[0].axis("off")
axes[0].set_title("(a) Original")
axes[1].imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)); axes[1].axis("off")
axes[1].set_title(f"(b) Vibrance-enhanced (a = {A})")
axes[2].plot(x, x, "k--", lw=1, alpha=0.4, label="identity")
axes[2].plot(x, f, color="#8B1A1A", lw=2, label="f(x)")
axes[2].set_xlim(0, 255); axes[2].set_ylim(0, 255)
axes[2].set_xlabel("Input saturation"); axes[2].set_ylabel("Output saturation")
axes[2].set_title("(c) Saturation transform"); axes[2].legend()

plt.tight_layout()
plt.savefig("/home/claude/en3160_a01/figures/q4_result.png", dpi=150)
print("saved q4_result.png, a =", A)
