# ======================================================
# Author: Nika Lu Yang
# Date: April 2025
# Bachelor Thesis Research (BTR): "Unraveling the Impact of Aging on Parkinson’s Disease-derived Neuroepithelial Stem Cells"
# Maastricht Science Programme (MSP)
# ======================================================

import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, disk
from skimage.color import gray2rgb
from skimage.io import imsave
from skimage.segmentation import clear_border, watershed
from skimage.feature import peak_local_max
import cv2
from aicsimageio import AICSImage
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# === Step 1: Define input and output directories ===
images_folder = "/Volumes/NO NAME/ICC_Cleaned"
save_path_main_directory = "/Users/nikalu/Downloads/53BP1_Results"

# Get script name
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Create timestamped results folder with script name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = os.path.join(save_path_main_directory, f'Analysis_{timestamp}_{script_name}')
os.makedirs(save_path, exist_ok=True)
output_root = save_path

# === Step 2: Define parameters for 53BP1 Foci Detection ===
# This can be changed to adjust the detection parameters 
params = {
    "min_size": 5,  # Minimum size of detected foci
    "area_min": 2,  # Minimum area of detected foci
    "area_max": 20,  # Maximum area of detected foci
    "ecc_max": 0.95,  # Maximum eccentricity of detected foci
    "intensity_threshold": 70,  # Intensity threshold for foci detection
    "use_clahe": False,  # Whether to apply CLAHE for contrast enhancement
    "use_sum_projection": True,  # Whether to use sum projection for Z-stacks
    "use_tophat": True,  # Whether to apply top-hat filtering
    "tophat_disk_small": 1,  # Small disk size for top-hat filtering
    "tophat_disk_large": 2,  # Large disk size for top-hat filtering
    "tophat_thresh_percentile": 85,  # Percentile threshold for top-hat filtering
}

# In case the initial parameters fail to detect foci, this is another set of parameters that will try to detect foci again with a differnt approach
fallback_params = params.copy()
fallback_params.update({
    "min_size": 1,  # Reduced minimum size for fallback
    "area_min": 0.3,  # Reduced minimum area for fallback
    "area_max": 50,  # Increased maximum area for fallback
    "ecc_max": 0.99,  # Increased maximum eccentricity for fallback
    "intensity_threshold": 10,  # Lower intensity threshold for fallback
    "tophat_thresh_percentile": 40,  # Lower percentile threshold for fallback
    "use_clahe": True,  # Enable CLAHE for fallback
    "tophat_disk_small": 1,
    "tophat_disk_large": 7, # Increased disk size for fallback
})

# === Step 3: Helper Functions ===
# Save preview images for raw, processed, and detected regions
def save_previews(image, signal, thresholded, binary_mask, regions, image_path, preview_paths):
    base_name = os.path.basename(image_path).replace('.nd2', '.png')
    imsave(os.path.join(preview_paths['raw'], base_name), image, check_contrast=False)
    imsave(os.path.join(preview_paths['tophat'], base_name), signal.astype(np.uint8), check_contrast=False)
    imsave(os.path.join(preview_paths['threshold'], base_name), thresholded.astype(np.uint8) * 255, check_contrast=False)
    imsave(os.path.join(preview_paths['mask'], base_name), binary_mask.astype(np.uint8) * 255, check_contrast=False)

    overlay = gray2rgb(image / 255.0)
    for r in regions:
        coords = r.coords
        overlay[coords[:, 0], coords[:, 1]] = [1, 0, 0]
    preview = (overlay * 255).astype(np.uint8)
    imsave(os.path.join(preview_paths['rp'], base_name), preview, check_contrast=False)

def run_analysis(image_data, cell_mask, param_set):
    signal = image_data
    thresholded = (signal > param_set['intensity_threshold'])
    binary = thresholded & cell_mask
    binary = remove_small_objects(binary, min_size=param_set['min_size'])
    binary = closing(binary, disk(1))
    binary = clear_border(binary)

    labels = label(binary)
    regions = [
        r for r in regionprops(labels, intensity_image=signal)
        if param_set['area_min'] < r.area < param_set['area_max'] and r.eccentricity < param_set['ecc_max']
    ]

    return regions, signal, thresholded, binary

# === Step 4: Batch Processing ===
input_root = images_folder

for date_folder in os.listdir(input_root):
    if date_folder.lower() == "Bad": # Skip folder that has the images that are not possible to analyze (acquisiton was bad)
        continue
    date_path = os.path.join(input_root, date_folder)
    if not os.path.isdir(date_path):
        continue

    for type_folder in os.listdir(date_path):
        if type_folder.lower() == "Bad": # Skip folder
            continue

        type_path = os.path.join(date_path, type_folder)
        if not os.path.isdir(type_path):
            continue

        print(f"\n--- Processing: {date_folder}/{type_folder} ---")
        results_base = os.path.join(output_root, date_folder, type_folder)
        os.makedirs(results_base, exist_ok=True)

        channel_results = []

        # Define paths for saving preview images
        preview_paths_template = lambda channel: {
            'raw': os.path.join(results_base, channel, 'Raw'),
            'tophat': os.path.join(results_base, channel, 'Tophat'),
            'threshold': os.path.join(results_base, channel, 'Threshold'),
            'mask': os.path.join(results_base, channel, 'Mask'),
            'rp': os.path.join(results_base, channel, 'Rp'),
            'failed': os.path.join(results_base, channel, 'Failed')
        }

        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.nd2')
        
        # Collect all valid image files
        all_files = []
        for root, dirs, files in os.walk(type_path):
            if os.path.basename(root).lower() == "shit":
                continue
            for file in files:
                if file.lower().endswith(valid_extensions) and not file.startswith("._"):
                    all_files.append(os.path.join(root, file))

        # Process each image file
        for image_path in all_files:
            try:
                # Load image data and preprocess
                img = AICSImage(image_path)
                dapi_stack = img.get_image_data("ZYX", T=0, C=0)
                foci1_stack = img.get_image_data("ZYX", T=0, C=1)
                foci2_stack = img.get_image_data("ZYX", T=0, C=2)

                # Perform Z-projection
                proj_fn = np.sum if params.get("use_sum_projection", False) else np.max
                dapi_proj = proj_fn(dapi_stack, axis=0)
                foci1_proj = proj_fn(foci1_stack, axis=0)
                foci2_proj = proj_fn(foci2_stack, axis=0)
                # foci_proj = foci1_proj if foci1_proj.mean() > foci2_proj.mean() else foci2_proj
                if np.percentile(foci1_proj, 99.5) > np.percentile(foci2_proj, 99.5):
                    foci_proj = foci1_proj
                else:
                    foci_proj = foci2_proj

                # Normalize and enhance contrast
                norm_dapi = cv2.normalize(dapi_proj, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                norm_foci = cv2.normalize(foci_proj, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Apply CLAHE if enabled (TRUE)
                if params.get("use_clahe", False):
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    norm_dapi = clahe.apply(norm_dapi)
                    norm_foci = clahe.apply(norm_foci)

                # Detect nuclei using watershed segmentation
                blurred = cv2.GaussianBlur(norm_dapi, (7, 7), 1)
                dapi_thresh = threshold_otsu(blurred)
                nuclei_mask = blurred > dapi_thresh
                nuclei_mask = remove_small_objects(nuclei_mask, min_size=100) # The smaller this is, the smaller nuclei can be detected too! 

                distance = ndi.distance_transform_edt(nuclei_mask)
                distance_blurred = cv2.GaussianBlur(distance, (5, 5), 0)  # ← new: smoother peaks
                local_max = peak_local_max(distance_blurred, labels=nuclei_mask, min_distance=3, footprint=np.ones((5, 5)), exclude_border=False)
                markers = np.zeros_like(nuclei_mask, dtype=int)
                for i, (y, x) in enumerate(local_max, 1):
                    markers[y, x] = i
                labels_ws = watershed(-distance_blurred, markers, mask=nuclei_mask)  
                cell_count = len(np.unique(labels_ws)) - 1

                # Save preview images for DAPI
                dapi_paths = preview_paths_template("dapi")
                for p in dapi_paths.values():
                    os.makedirs(p, exist_ok=True)
                save_previews(norm_dapi, norm_dapi, nuclei_mask, nuclei_mask, [], image_path, dapi_paths)

                # Save preview images for 53BP1
                preview_paths = preview_paths_template("53bp1")
                for p in preview_paths.values():
                    os.makedirs(p, exist_ok=True)
                regions, signal, thresholded, binary = run_analysis(norm_foci, nuclei_mask, params)

                # Retry with fallback parameters if no regions found with initial parameters
                rescued = False
                if not regions:
                    # Apply contrast/brightness boost for fallback
                    norm_foci = cv2.convertScaleAbs(norm_foci, alpha=1.5, beta=20)
                    regions, signal, thresholded, binary = run_analysis(norm_foci, nuclei_mask, fallback_params)
                    rescued = True
                if not regions:
                    failed_path = os.path.join(preview_paths['failed'], os.path.basename(image_path).replace('.nd2', '.png'))
                    imsave(failed_path, norm_foci, check_contrast=False)
                    foci_count = 0
                else:
                    save_previews(norm_foci, signal, thresholded, binary, regions, image_path, preview_paths)
                    foci_count = len(regions)

                # Append results for the current image
                channel_results.append({
                    "filename": os.path.basename(image_path),
                    "Cell Number": cell_count,
                    "53BP1 foci": foci_count,
                    "Retried": "yes" if rescued else "no"
                })

            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
        
        # Save results for the current folder
        if channel_results:
            df = pd.DataFrame(channel_results)
            csv_path = os.path.join(results_base, "Results.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved overall summary CSV: {csv_path}")
