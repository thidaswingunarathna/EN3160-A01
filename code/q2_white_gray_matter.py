"""EN3160 A01 - Q2: Accentuating white / gray matter in a brain proton-density slice."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from q1_intensity_transform import intensity_transform

img = cv2.imread("/home/claude/en3160_a01/images/q2_brain.jpeg", cv2.IMREAD_GRAYSCALE)

# Gray matter sits roughly in the 140-180 mid-gray band; white matter sits
# roughly in the 195-235 bright band (found by inspecting the histogram).
# A "bump" LUT that is boosted only inside the target band and suppressed
# outside it isolates/accentuates that tissue type.
gray_matter_bp = np.array([
    [0, 0], [120, 0], [145, 235], [178, 235], [205, 0], [255, 0]
])
white_matter_bp = np.array([
    [0, 0], [175, 0], [197, 0], [212, 255], [238, 255], [255, 80]
])

gray_out, gray_lut = intensity_transform(img, gray_matter_bp)
white_out, white_lut = intensity_transform(img, white_matter_bp)

fig, axes = plt.subplots(2, 2, figsize=(9, 9))
axes[0, 0].plot(np.arange(256), gray_lut, color="#2A6F97", lw=2)
axes[0, 0].plot(gray_matter_bp[:, 0], gray_matter_bp[:, 1], "ko", ms=3)
axes[0, 0].set_title("(a) Gray-matter accentuation transform")
axes[0, 0].set_xlabel("Input intensity"); axes[0, 0].set_ylabel("Output intensity")
axes[0, 0].set_xlim(0, 255); axes[0, 0].set_ylim(0, 255)

axes[0, 1].imshow(gray_out, cmap="gray"); axes[0, 1].axis("off")
axes[0, 1].set_title("(b) Gray matter accentuated")

axes[1, 0].plot(np.arange(256), white_lut, color="#8B1A1A", lw=2)
axes[1, 0].plot(white_matter_bp[:, 0], white_matter_bp[:, 1], "ko", ms=3)
axes[1, 0].set_title("(c) White-matter accentuation transform")
axes[1, 0].set_xlabel("Input intensity"); axes[1, 0].set_ylabel("Output intensity")
axes[1, 0].set_xlim(0, 255); axes[1, 0].set_ylim(0, 255)

axes[1, 1].imshow(white_out, cmap="gray"); axes[1, 1].axis("off")
axes[1, 1].set_title("(d) White matter accentuated")

plt.tight_layout()
plt.savefig("/home/claude/en3160_a01/figures/q2_result.png", dpi=150)
print("saved q2_result.png")
