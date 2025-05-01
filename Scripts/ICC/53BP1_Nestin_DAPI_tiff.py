import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, white_tophat, disk
from skimage.segmentation import clear_border
import cv2
from tifffile import imread  # Use tifffile to read .tiff files

# === Step 1: Load the image ===
tiff_path = "/Users/nikalu/Downloads/ICC after fiji (T4.13GC P21)/53b1 + nestin + dapi.tif"
data = imread(tiff_path)  # Load the .tiff file

print(f"Shape of image: {data.shape}")  # Check the shape of the image

# === Step 2: Extract and preprocess the image ===
# Assuming the .tiff file has channels in the third dimension (XYC order)
data_53bp1 = data[:, :, 0]  # 53BP1 channel (C=0)
data_nestin = data[:, :, 1]   # DAPI channel (C=1)
data_dapi = data[:, :, 2] # Nestin channel (C=2)

# === Step 3: Normalize and enhance the images ===
def normalize_and_enhance(image):
    norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    return clahe.apply(norm)

norm_53bp1 = normalize_and_enhance(data_53bp1)
norm_dapi = normalize_and_enhance(data_dapi)
norm_nestin = normalize_and_enhance(data_nestin)

# Visualize normalized images
plt.figure()
plt.title("Normalized 53BP1")
plt.imshow(norm_53bp1, cmap='gray')
plt.show()

plt.figure()
plt.title("Normalized DAPI")
plt.imshow(norm_dapi, cmap='gray')
plt.show()

plt.figure()
plt.title("Normalized Nestin")
plt.imshow(norm_nestin, cmap='gray')
plt.show()

# === Step 4: Segment cell regions ===
# DAPI is a nuclear marker, so use it to detect nuclei
# Use adaptive thresholding for DAPI
dapi_mask = cv2.adaptiveThreshold(
    norm_dapi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
dapi_mask = dapi_mask > 0  # Convert to boolean mask

# Morphological processing for DAPI
dapi_mask = remove_small_objects(dapi_mask, min_size=10)  # Retain smaller nuclei
dapi_mask = clear_border(dapi_mask)  # Remove objects touching the border

# Visualize the original DAPI image
plt.figure()
plt.title("Original DAPI Image")
plt.imshow(norm_dapi, cmap='gray')
plt.show()

# Visualize the thresholded DAPI mask
plt.figure()
plt.title("Thresholded DAPI Mask")
plt.imshow(dapi_mask, cmap='gray')
plt.show()

# Visualize the processed DAPI mask
plt.figure()
plt.title("Processed DAPI Mask")
plt.imshow(dapi_mask, cmap='gray')
plt.show()

# Labeling and filtering for DAPI
labels_dapi = label(dapi_mask)
regions_dapi = []
for region in regionprops(labels_dapi):
    if 5 < region.area < 1000:  # Relaxed area range
        regions_dapi.append(region)

# Count total cells
cell_count = len(regions_dapi)
print(f"Total number of cells (DAPI): {cell_count}")

# Visualize detected nuclei
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(norm_dapi, cmap='gray')
for region in regions_dapi:
    y, x = region.centroid
    ax.plot(x, y, 'b.', markersize=5)
ax.set_title(f"Total Cells (DAPI): {cell_count}")
ax.axis("off")
plt.show()

# Nestin is a membrane marker, so use it to detect cell boundaries
nestin_thresh_val = threshold_otsu(norm_nestin)
nestin_mask = norm_nestin > nestin_thresh_val

# Visualize Nestin mask
plt.figure()
plt.title("Nestin Mask (Cell Boundaries)")
plt.imshow(nestin_mask, cmap='gray')
plt.show()

# === Step 5: Apply multi-scale Top-Hat filter for 53BP1 ===
tophat_small = white_tophat(norm_53bp1, footprint=disk(3))  # Small foci
tophat_large = white_tophat(norm_53bp1, footprint=disk(5))  # Larger foci
tophat_53bp1 = tophat_small + tophat_large

# Visualize top-hat filtered image for 53BP1
plt.figure()
plt.title("Top-Hat Filtered Image (53BP1)")
plt.imshow(tophat_53bp1, cmap='gray')
plt.show()

# === Step 6: Thresholding and masking for 53BP1 ===
thresh_val_53bp1 = np.percentile(tophat_53bp1, 98)
binary_53bp1 = tophat_53bp1 > thresh_val_53bp1
binary_53bp1 = binary_53bp1 & dapi_mask  # Apply DAPI mask to focus on nuclei

# Visualize binary image for 53BP1 after masking
plt.figure()
plt.title("Binary Image After Applying DAPI Mask (53BP1)")
plt.imshow(binary_53bp1, cmap='gray')
plt.show()

# === Step 7: Morphological processing for 53BP1 ===
binary_53bp1 = remove_small_objects(binary_53bp1, min_size=10)
binary_53bp1 = closing(binary_53bp1, disk(1))
binary_53bp1 = clear_border(binary_53bp1)

# Visualize binary after morphological processing
plt.figure()
plt.title("Binary Image After Morphological Processing (53BP1)")
plt.imshow(binary_53bp1, cmap='gray')
plt.show()

# === Step 8: Labeling and filtering regions for 53BP1 ===
labels_53bp1 = label(binary_53bp1)
regions_53bp1 = []
for region in regionprops(labels_53bp1, intensity_image=norm_53bp1):
    if 5 < region.area < 100 and region.eccentricity < 0.9:
        regions_53bp1.append(region)

# Count 53BP1 foci
foci_count = len(regions_53bp1)
print(f"Automatically counted 53BP1 foci: {foci_count}")

# Visualize detected 53BP1 foci
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(norm_53bp1, cmap='gray')
for region in regions_53bp1:
    y, x = region.centroid
    ax.plot(x, y, 'r.', markersize=5)
ax.set_title(f"53BP1 Foci Counting: {foci_count}")
ax.axis("off")
plt.show()

# Debugging: Visualize all detected regions
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(norm_dapi, cmap='gray')
for region in regionprops(label(dapi_mask)):
    y, x = region.centroid
    ax.plot(x, y, 'r.', markersize=5)  # Mark detected nuclei in red
ax.set_title("Detected Nuclei (DAPI)")
ax.axis("off")
plt.show()