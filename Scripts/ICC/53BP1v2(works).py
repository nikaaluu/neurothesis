import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, square, white_tophat
from skimage.segmentation import clear_border
from skimage.morphology import disk
import cv2
from aicsimageio import AICSImage

# === Step 1: Load the image ===
# Path to the ND2 file
nd2_path = "/Users/nikalu/Downloads/ICC_Images/20250429_Staining/20250429_C6_53bp1.nd2"

# Load image (aicsimageio is ND2 compatible)
img = AICSImage(nd2_path)

print(f"Shape of image: {img.dims}")  # Shows available dimensions in the file
print(f"Available channels: {img.shape}")  # Displays the shape of the image. E.g. (T, C, Z, Y, X)
"""
T: Number of time points (time dimension)
C: Number of channels (e.g., different fluorescence markers)
Z: Number of z-slices (depth, or optical sections along the z-axis)
Y: Height (pixels along the vertical axis)
X: Width (pixels along the horizontal axis)

E.g. If the shape is (10, 3, 20, 512, 512), it means:
- 10 time points
- 3 channels
- 20 z-slices
- Each slice is 512 x 512 pixels

"""
# === Step 2: Extract and preprocess the image ===
# Select channel, Z, time (here: channel 0, Z-Max, time 0)
# If there are multiple channels: foci_data = img.get_image_data("ZYX", T=0, C=1)
data = img.get_image_data("ZYX", T=0, C=0)  # Extracts image data for the first time point and the first channel in the (Z, Y, X) shape
proj = np.max(data, axis=0)  # Z-Projection by taking the max intensity along the Z-axis. This creates a 2D image

# === Step 3: Normalize and enhance the image ===
# This normalizes the image and then converts to 8-bit
norm = cv2.normalize(proj, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# Enhance contrast with Contrast Limited Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8)) # The clipLimit controls the contrast enhancement strength. While the tileGridSize defines the size of the grid for local contrast enhancement
norm = clahe.apply(norm)

# Visualize enhanced image
plt.figure()
plt.title("Enhanced Image with CLAHE")
plt.imshow(norm, cmap='gray')
plt.show()

# === Step 4: Segment cell regions ===
# Using Otsu's thresholding
cell_thresh_val = threshold_otsu(norm) # Calculates optimal gobal threshold value
cell_mask = norm > cell_thresh_val # Creates binary mask wehre pixel values greater than the threshold are "TRUE" (indicate cell regions)

# Visualize cell mask
plt.figure()
plt.title("Cell Mask")
plt.imshow(cell_mask, cmap='gray')
plt.show()

# === Step 5: Apply multi-scale Top-Hat filter ===
"""
white_tophat enhances the small bright spots (foci) by removing uneven backgroung light
The disk size define the circular structuring elements for detecting foci with different sizes
"""
tophat_small = white_tophat(norm, footprint=disk(3))  # Small foci
tophat_large = white_tophat(norm, footprint=disk(5))  # Larger foci
tophat = tophat_small + tophat_large  # Combine results

# Visualize top-hat filtered image
plt.figure()
plt.title("Top-Hat Filtered Image (Multi-Scale)")
plt.imshow(tophat, cmap='gray')
plt.show()

# === Step 6: Threholding and masking
# Apply global thresholding (Otsu's method)
thresh_val = np.percentile(tophat, 98)  # Calculates the nth percentile of pixl intensities in top-hat filtered image
binary = tophat > thresh_val # Creates binary mask wehre pixel values greater than the threshold are "TRUE" (indicate cell regions)
binary = binary & cell_mask # Applies cell mask to exclude regions outisde cells

# Visualize binary image after applying cell mask
plt.figure()
plt.title("Binary Image After Applying Cell Mask")
plt.imshow(binary, cmap='gray')
plt.show()

# === Step 7: Morphological processing ===
binary = remove_small_objects(binary, min_size=10)  # removes noise from binary mask. 
    # The min_size parameter defines the minimum size of objects to keep
binary = closing(binary, disk(1))  # fills in small holes in the binary mask with circular structuring
binary = clear_border(binary) # removes objects touching the edge of the image

# Visualize binary after morphological processing
plt.figure()
plt.title("Binary Image After Morphological Processing")
plt.imshow(binary, cmap='gray')
plt.show()

# === Step 8: Labeling and filtering regions ===
labels = label(binary)
regions = []
for region in regionprops(labels, intensity_image=norm):  # Pass norm as intensity_image
    print(f"Region Area: {region.area}, Eccentricity: {region.eccentricity}, Intensity: {region.mean_intensity}")
    if 5 < region.area < 100 and region.eccentricity < 0.9:  # Relaxed area and eccentricity range
        regions.append(region)
"""
For each labeled region, the properties of area, eccentricity and intensity are extracted. 
There are some filtering criteria that retain regions with an area between 5-100 pixels (region.area) and also regions with eccentricity < 0.9 (region.eccentricity)
"""

# === Step 9: Debugging: Visualize all detected regions ===
# Display all detected regions before filtering in blue
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(norm, cmap='gray')
for region in regionprops(labels, intensity_image=norm):
    y, x = region.centroid
    ax.plot(x, y, 'b.', markersize=5)  # Mark all detected regions in blue
ax.set_title("All Detected Regions (Blue)")
ax.axis("off")
plt.show()

# === Counting & Visualizing foci ===
foci_count = len(regions)
print(f"Automatically counted foci: {foci_count}")

# Visualize detected foci
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(norm, cmap='gray')
for region in regions:
    y, x = region.centroid
    ax.plot(x, y, 'r.', markersize=5)  # Increase marker size for better visibility
ax.set_title(f"Foci counting: {foci_count}")
ax.axis("off")
plt.show()


