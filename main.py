import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load main image
image = cv2.imread("./images/opencv-1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]

# ---- Reduce Saturation and Keep Warm Tone ----
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
hsv[:,:,1] = np.clip(hsv[:,:,1] * 0.6, 0, 255)  # Slightly reduce saturation
hsv[:,:,0] = (hsv[:,:,0] + 10) % 180  # Maintain warm tone
image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Increase Clarity (Enhance Contrast)
lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
l = cv2.equalizeHist(l)  # Histogram equalization to enhance details
image = cv2.merge([l, a, b])
image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

# ---- Road-Aligned Text ----
text = "PATH BREAKER"
font = cv2.FONT_HERSHEY_DUPLEX
thickness = 10

# Dynamic font scaling
target_width = int(w * 0.8)
for scale in np.linspace(6.0, 0.5, 100):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw < target_width:
        break

# Perspective transform matrix (Right-Aligned)
src_pts = np.float32([[0, 0], [tw, 0], [0, th], [tw, th]])
dst_pts = np.float32([
    [1647, 1202],  # Top-left position
    [1647 + int(tw * 0.8), 1202],  # Top-right
    [1647 + int(tw * 0.2), 1500],  # Bottom-left
    [1647 + int(tw * 1.0), 1500]   # Bottom-right
])

M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Create and warp text
text_layer = np.zeros((h, w, 3), dtype=np.uint8)
cv2.putText(text_layer, text, (0, th), font, scale, (40, 40, 40), thickness+5, cv2.LINE_AA)
cv2.putText(text_layer, text, (0, th), font, scale, (255, 200, 100), thickness, cv2.LINE_AA)
warped_text = cv2.warpPerspective(text_layer, M, (w, h))
image = cv2.addWeighted(image, 1, warped_text, 0.9, 0)

# ---- Final Adjustments ----
# Warm vignette
y, x = np.ogrid[:h, :w]
vignette = 1 - np.sqrt((x - w/2)**2 + (y - h/2)**2)/w
vignette = np.clip(vignette**3 * 1.5, 0, 1)[..., np.newaxis]
image = np.clip(image * (0.8 + 0.2*vignette), 0, 255).astype(np.uint8)

plt.figure(figsize=(16, 10))
plt.imshow(image)
plt.axis('off')
plt.show()

cv2.imwrite("./images/final_poster.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
