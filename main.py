import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------- TASK A --------

# Load image
image = cv2.imread("Palimpsest.jpe")
if image is None:
    raise FileNotFoundError("Image file 'Palimpsest.jpe' not found or could not be opened.")
# Convert image from BGR (OpenCV default) to RGB for visualization
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split into R, G, B channels
red, green, blue = cv2.split(image_rgb)

# Show channels
figure_RGB_split = plt.figure("RGB Split")
axes_RGB_split = figure_RGB_split.subplots(1, 4)

axes_RGB_split[0].imshow(image_rgb)
axes_RGB_split[0].set_title("Original")
axes_RGB_split[0].axis("off")

axes_RGB_split[1].imshow(red, cmap='gray')
axes_RGB_split[1].set_title("Red Channel")
axes_RGB_split[1].axis("off")

axes_RGB_split[2].imshow(green, cmap='gray')
axes_RGB_split[2].set_title("Green Channel")
axes_RGB_split[2].axis("off")

axes_RGB_split[3].imshow(blue, cmap='gray')
axes_RGB_split[3].set_title("Blue Channel")
axes_RGB_split[3].axis("off")

figure_RGB_split.tight_layout()
figure_RGB_split.show()

# Convert RGB to float
red_float = red.astype(np.float32)
green_float = green.astype(np.float32)
blue_float = blue.astype(np.float32)

# Select a patch of parchment
x, y, w, h = 120, 30, 100, 80

# Compute alpha from selected patch
patch_red = red_float[y:y+h, x:x+w]
patch_green = green_float[y:y+h, x:x+w]
alpha = np.mean(patch_green) / np.mean(patch_red)

# Combine channels for suppression
overwrite_estimate = (red_float + green_float) / 2
underwriting = blue_float - alpha * overwrite_estimate

# Isolate underwriting
underwriting = cv2.normalize(underwriting, None, 0, 255, cv2.NORM_MINMAX)
underwriting = underwriting.astype(np.uint8)

figure_underwriting = plt.figure("Isolated Underwriting")
plt.imshow(underwriting, cmap='gray')
plt.title("Isolated Underwriting")
plt.axis("off")
plt.show()