import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from sklearn.metrics import precision_score, recall_score, f1_score
import glob
import pandas as pd

# -------- TASK A: Separate the Classes --------

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the image
image_path = os.path.join(script_dir, "Palimpsest.jpe")

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found or could not be opened.")
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
plt.show(block=False)

# -------- TASK B: Su et al. Binarization --------

# Compute the contrast image based on local max and min in 3x3 neighborhood
def compute_contrast_image(img, eps=1e-5):
    kernel_size = 3

    # Maximum filter and minimum filter (dilation and erosion)
    f_max = cv2.dilate(img, np.ones((kernel_size, kernel_size)))
    f_min = cv2.erode(img, np.ones((kernel_size, kernel_size)))

    # Contrast formula from Su et al.
    contrast = (f_max - f_min) / (f_max + f_min + eps)
    contrast = np.nan_to_num(contrast, nan=0.0)  # Replace NaN with 0

    # Scale contrast to 0-255 and convert to uint8
    contrast_scaled = (contrast * 255).clip(0, 255).astype(np.uint8)
    return contrast_scaled

# Detect high contrast pixels using Otsu's global thresholding
def detect_high_contrast(contrast_img):
    # Find Otsu's threshold
    thresh = threshold_otsu(contrast_img)

    # Create binary mask for high contrast pixels
    high_contrast_mask = contrast_img > thresh

    # Convert boolean mask to integer (0 or 1)
    return high_contrast_mask.astype(np.uint8)

# Estimate window size and minimum number of points
def estimate_window_size(contrast_img):
    high_contrast = detect_high_contrast(contrast_img)

    # Find list of pixel coordinates (y, x) where text edges were detected
    points = np.argwhere(high_contrast == 1)
    # If image is very empty, return default values
    if len(points) < 2:
        return 15, 5

    # Sort points by row and measure distances between neighboring points on the same text line
    points = points[np.argsort(points[:, 0])]
    distances = []
    for i in range(1, len(points)):
        if points[i-1, 0] == points[i, 0]: # same row
            dist = abs(points[i,1] - points[i-1,1])
            if dist > 0:
                distances.append(dist)

    # If no valid horizontal distances (blank page, isolated points) return default values
    if len(distances) == 0:
        return 15, 5

    # Estimate stroke width
    stroke_width = int(np.median(distances))

    # Calculate window size with Su et al. formula
    window_size = max(3 * stroke_width, 15)
    if window_size % 2 == 0:
        window_size += 1  # make odd to ensure center pixel

    # Estimate N_min
    if stroke_width <= 2:
        N_min = 5  # for tiny strokes we need more points to trust classification
    elif stroke_width <= 5:
        N_min = 3 # for normal strokes we use moderate number of points
    else:
        N_min = 2 # for thick strokes we can use less points

    return window_size, N_min

# Perform Su et al. binarization on a grayscale document image
def su_binarization(img):
    # Build the contrast image
    contrast_img = compute_contrast_image(img)

    # Estimate window size and N_min dynamically
    window_size, N_min = estimate_window_size(contrast_img)
    print(f"Estimated window_size: {window_size}, N_min: {N_min}")

    # Build the contrast image
    E = detect_high_contrast(contrast_img)

    # Pad the images to handle edges when sliding window
    padded_img = cv2.copyMakeBorder(img, window_size//2, window_size//2,
                                    window_size//2, window_size//2, cv2.BORDER_REFLECT)
    padded_E = cv2.copyMakeBorder(E, window_size//2, window_size//2,
                                window_size//2, window_size//2, cv2.BORDER_REFLECT)

    # Create an empty image to store binarized result (same size as input)
    bin_img = np.zeros_like(img, dtype=np.uint8)

    # Iterate over the image with a sliding window
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            x0, x1 = x, x + window_size
            y0, y1 = y, y + window_size

            # Extract the patch
            patch = padded_img[y0:y1, x0:x1] # Intensity values
            patch_E = padded_E[y0:y1, x0:x1] # High contrast mask values (0 or 1)

            # Count number of high contrast points in the window
            Ne = np.sum(patch_E == 1)

            # If there are enough high contrast points:
            if Ne >= N_min:
                # Take the pixel values at those points
                vals = patch[patch_E == 1]

                # Compute the mean and std of the neighboring high contrast pixels
                mean = np.mean(vals)
                std = np.std(vals)

                # Pixel classification rule (Equation 2 from Su et al.)
                if img[y, x] <= mean + std / 2:
                    bin_img[y, x] = 0 # Foreground (text)
                else:
                    bin_img[y, x] = 255 # Background
            else:
                bin_img[y, x] = 255 # Not enough high contrast neighbors - assume background

    return bin_img

# Compute evaluation metrics (Precision, Recall, F-Score, PSNR)
def evaluate_metrics(pred, gt):
    pred_bin = (pred < 128).astype(np.uint8) # Binarize prediction
    gt_bin = (gt < 128).astype(np.uint8) # Binarize ground truth

    precision = precision_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    recall = recall_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    fscore = f1_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    psnr = compare_psnr(gt_bin * 255, pred_bin * 255, data_range=255)

    return {
        "Precision": precision,
        "Recall": recall,
        "F-Score": fscore,
        "PSNR": psnr
    }

# -------- Batch Binarization of DIBCO2009 --------

# Paths to input and ground truth files
input_folder = "dibco2009/input"
gt_folder = "dibco2009/gt"

# Find all the input images
input_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.tif")))

# Collect metrics for all the images
all_metrics = []

for input_path, gt_path in zip(input_files, gt_files):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    bin_img = su_binarization(img)

    metrics = evaluate_metrics(bin_img, gt)
    metrics["Image"] = os.path.basename(input_path)
    all_metrics.append(metrics)

    print(f"Done: {os.path.basename(input_path)}")

results_df = pd.DataFrame(all_metrics)
print("\nAverage metrics over all images:")
print(results_df.mean(numeric_only=True))

# Save to CSV
results_df.to_csv("binarization_results.csv", index=False)
print("\nResults saved to binarization_results.csv")