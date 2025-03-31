import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image
image = cv2.imread("Palimpsest.jpe")
if image is None:
    raise FileNotFoundError("Image file 'Palimpsest.jpe' not found or could not be opened.")
# Convert image from BGR (OpenCV default) to RGB for visualization
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Split into R, G, B channels
r, g, b = cv2.split(image_rgb)

# 3. Show channels
figure, axes = plt.subplots(1, 4, figsize=(10, 5))

# Display the original RGB image in the first subplot
axes[0].imshow(image_rgb)
axes[0].set_title("Original")

# Display the Red channel in grayscale in the second subplot
axes[1].imshow(r, cmap='gray')
axes[1].set_title("Red Channel")

# Display the Green channel in grayscale in the third subplot
axes[2].imshow(g, cmap='gray')
axes[2].set_title("Green Channel")

# Display the Blue channel in grayscale in the fourth subplot
axes[3].imshow(b, cmap='gray')
axes[3].set_title("Blue Channel")

for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
