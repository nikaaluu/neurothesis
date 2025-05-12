# ======================================================
# Author: Nika Lu Yang
# Date: April 2025
# Bachelor Thesis Research (BTR): "Unraveling the Impact of Aging on Parkinson’s Disease-derived Neuroepithelial Stem Cells"
# Maastricht Science Programme (MSP)
# ======================================================

import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from cellpose import models
from skimage.io import imsave
from skimage.measure import label, regionprops
from skimage.morphology import (
    closing,
    disk,
    remove_small_objects,
    white_tophat,
)
from skimage.segmentation import clear_border, find_boundaries
from tqdm import tqdm


# === Step 1: Define input and output directories ===
images_folder = "ICC_Cleaned"
save_path_main_directory = "53BP1_Results"

# Get script name
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Create timestamped results folder with script name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = os.path.join(save_path_main_directory, f'Analysis_{timestamp}_{script_name}')
os.makedirs(save_path, exist_ok=True)
output_root = save_path

# === Step 2: Define parameters for 53BP1 Foci Detection ===
# This can be changed to adjust the detection parameters
PERCENTILE         = 93.0 # Percentile-Schwellenwert (z.B. 97)
ABS_INTENSITY_MIN  = 85   # Absolute Hüllwert (0–255)
MIN_AREA           = 2  # minimal Region in px

params = {
    "min_size": 5,
    "area_min": 2,
    "area_max": 10,
    "ecc_max": 0.95,
    "intensity_threshold": 70,
    "use_clahe": False,
    "use_sum_projection": True,
    "use_tophat": True,
    "tophat_disk_small": 1,
    "tophat_disk_large": 2,
    "tophat_thresh_percentile": 90
}

# === Step 3: Helper Functions ===
# Save preview images for raw, processed, and detected regions
def save_previews(image, signal, thresholded, binary_mask, regions, image_path, preview_paths):
    base_name = os.path.basename(image_path).replace('.nd2', '.png')
    imsave(os.path.join(preview_paths['raw'], base_name), image, check_contrast=False)
    imsave(os.path.join(preview_paths['tophat'], base_name), signal.astype(np.uint8), check_contrast=False)
    imsave(os.path.join(preview_paths['threshold'], base_name), thresholded.astype(np.uint8) * 255, check_contrast=False)
    imsave(os.path.join(preview_paths['mask'], base_name), binary_mask.astype(np.uint8) * 255, check_contrast=False)
    base_name = os.path.basename(image_path).rsplit(".", 1)[0] + ".png"
    if regions:
        overlay = np.dstack([image] * 3)
        for r in regions:
            yy, xx = map(int, r.centroid)
            cv2.circle(overlay, (xx, yy), 3, (0, 255, 0), 1, cv2.LINE_AA)
        imsave(
            os.path.join(preview_paths["rp"], base_name.replace(".png", "_foci_overlay.png")),
            overlay.astype(np.uint8),
            check_contrast=False,
        )

def run_analysis(image_data: np.ndarray, cell_mask: np.ndarray, param_set: dict):

    # --- 1) optional white top‑hat to enhance spots ---
    if param_set.get("use_tophat", False):
        selem = disk(param_set.get("tophat_disk_small", 1))
        processed = white_tophat(image_data, selem)
    else:
        processed = image_data.copy()

    # --- 2) adaptive or fixed threshold ---
    if "tophat_thresh_percentile" in param_set:
        thr_val = np.percentile(processed, param_set["tophat_thresh_percentile"])
    else:
        thr_val = param_set["intensity_threshold"]

    thresholded = processed > thr_val

    # --- 3) restrict to nuclear mask ---
    binary = thresholded & cell_mask

    # --- 4) morphological cleanup ---
    binary = remove_small_objects(binary, min_size=param_set["min_size"])
    binary = closing(binary, disk(1))
    binary = clear_border(binary)

    # --- 5) label & region filtering ---
    lbl = label(binary)
    regions = [
        r
        for r in regionprops(lbl, intensity_image=processed)
        if param_set["area_min"] < r.area < param_set["area_max"] and r.eccentricity < param_set["ecc_max"]
    ]

    return regions, processed, thresholded, binary

# === Step 4: Foci Detection ===
def detect_foci_simple(foci8, mask):
    """
    1) White-Tophat (disk=5) um punktförmige Spitzen zu betonen
    2) Threshold = mean(mask) + 3*std(mask)  → sehr hoch, greift nur die hellsten
    3) Maske anwenden
    4) Label & Regionprops: min_area=1, keine Further Filters
    5) Return regions, processed, thresholded, binary
    """
    vals = foci8[mask]
    if vals.size == 0:
        return []
    thr_rel = np.percentile(vals, PERCENTILE)
    thr = max(thr_rel, ABS_INTENSITY_MIN)

    bw = (foci8 > thr) & mask
    lbl = label(bw)

    foci_list = []
    for prop in regionprops(lbl):
        if prop.area < MIN_AREA:
            continue
        
        ys, xs = prop.coords[:,0], prop.coords[:,1]
        max_int = int(foci8[ys, xs].max())
        y, x = map(int, prop.centroid)
        foci_list.append({'y': y, 'x': x, 'intensity': max_int})

    regions = get_regions(foci_list, foci8)
    binary = np.zeros_like(foci8, dtype=bool)
    proc = np.zeros_like(foci8, dtype=float)
    return regions, proc, thresholded, binary

def get_regions(foci_list, foci8):
    regions = []
    for foci in foci_list:
        y, x = foci['y'], foci['x']
        intensity = foci['intensity']
        region = {
            'label': len(regions) + 1,
            'coords': np.array([[y, x]]),
            'intensity': intensity,
            'area': 1,
            'eccentricity': 0.0,
            'centroid': (y, x)
        }
        regions.append(region)
    return regions

# === Step 5: Batch Processing ===
input_root = images_folder
cp_model = models.Cellpose(gpu=False, model_type="nuclei")
channel_results = []
for date_folder in os.listdir(input_root):
    if date_folder.lower() == "Bad":
        continue
    date_path = os.path.join(input_root, date_folder)
    if not os.path.isdir(date_path):
        continue

    for type_folder in os.listdir(date_path):
        if type_folder.lower() == "Bad":
            continue

        type_path = os.path.join(date_path, type_folder)
        if not os.path.isdir(type_path):
            continue

        print(f"\n--- Processing: {date_folder}/{type_folder} ---")
        results_base = os.path.join(output_root, date_folder, type_folder)
        os.makedirs(results_base, exist_ok=True)


        preview_paths_template = lambda channel: {
            'raw': os.path.join(results_base, channel, 'Raw'),
            'tophat': os.path.join(results_base, channel, 'Tophat'),
            'threshold': os.path.join(results_base, channel, 'Threshold'),
            'mask': os.path.join(results_base, channel, 'Mask'),
            'rp': os.path.join(results_base, channel, 'Rp'),
            'failed': os.path.join(results_base, channel, 'Failed')
        }

        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.nd2')

        all_files = []
        for root, dirs, files in os.walk(type_path):
            for file in files:
                if file.lower().endswith(valid_extensions) and not file.startswith("._"):
                    all_files.append(os.path.join(root, file))

        for image_path in tqdm(all_files, desc=f"Processing {date_folder}/{type_folder}", unit="image"):
            try:

                folder_name = os.path.basename(os.path.dirname(image_path))
                passage_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
                cell_line_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_path))))


                
                if folder_name.lower() != "bad":
                                    
                    img = AICSImage(image_path)
                    dapi_stack = img.get_image_data("ZYX", T=0, C=0)
                    foci1_stack = img.get_image_data("ZYX", T=0, C=1)
                    foci2_stack = img.get_image_data("ZYX", T=0, C=2)

                    proj_fn = np.sum if params.get("use_sum_projection", False) else np.max
                    dapi_proj = proj_fn(dapi_stack, axis=0)
                    foci_proj = proj_fn(foci1_stack, axis=0) 

                    norm_dapi = cv2.normalize(dapi_proj, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    
                    norm_foci = cv2.normalize(foci_proj, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    if params.get("use_clahe", False):
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        norm_dapi = clahe.apply(norm_dapi)
                        norm_foci = clahe.apply(norm_foci)

                    masks_cp, *_ = cp_model.eval(
                        norm_dapi,
                        channels=[0, 0],
                        diameter=None,           # auto
                        flow_threshold=-2,
                        cellprob_threshold=-2,
                        resample=True,
                    )
                    if masks_cp.max() == 0:
                        print(f"  No nuclei found in {image_path}.")
                        continue
                    # 2) Staub‑Filter (anpassen, falls nötig)
                    AREA_MIN  = 40
                    MEAN_MAX  = 100    
                    MAXI_MAX  = 200    

                    for r in regionprops(masks_cp, intensity_image=norm_dapi):
                        if (r.area < AREA_MIN
                            or r.mean_intensity > MEAN_MAX
                            or norm_dapi[r.coords[:, 0], r.coords[:, 1]].max() > MAXI_MAX):
                            masks_cp[masks_cp == r.label] = 0

                    nuclei_mask = masks_cp > 0          # binär fürs Foci‑Schritt
                    cell_count  = masks_cp.max()        # Anzahl Kerne nach Filter
                    dapi_paths = preview_paths_template("dapi")
                    for p in dapi_paths.values():
                        os.makedirs(p, exist_ok=True)
                    save_previews(norm_dapi, norm_dapi, nuclei_mask, nuclei_mask, [], image_path, dapi_paths)
                    # 3) Overlay‑PNG speichern (anstelle Watershed‑Overlay)
                    overlay_ws = np.dstack([norm_dapi]*3)
                    overlay_ws[find_boundaries(masks_cp)] = [255, 0, 0]
                    imsave(
                        os.path.join(
                            dapi_paths['rp'],
                            os.path.basename(image_path).replace('.nd2', '_cellpose_overlay.png')
                        ),
                        overlay_ws.astype(np.uint8),
                        check_contrast=False,
                    )



                    preview_paths = preview_paths_template("53bp1")
                    for p in preview_paths.values():
                        os.makedirs(p, exist_ok=True)

                    # 4) Foci‑Detection
                    preview_paths = preview_paths_template("53bp1")
                    for p in preview_paths.values():
                        os.makedirs(p, exist_ok=True)

                    regions, processed_signal, thresholded, binary_mask = run_analysis(
                        norm_foci, nuclei_mask, params
                    )
                    foci_count = len(regions)
                    rescued = 0

                    if foci_count == 0:
                        foci_count = 0
                        rescued = 0

                    save_previews(norm_foci, processed_signal, thresholded, binary_mask, regions, image_path, preview_paths)

                    channel_results.append({
                        "filename": os.path.basename(image_path),
                        "cell_count": cell_count,
                        "foci_count": foci_count,
                        "well": folder_name,
                        "passage": passage_name,
                        "cell_line": cell_line_name,
                        "bad": False,
                    })
                else:
                    print(f"  Skipping {image_path} due to 'Bad' folder.")
                    channel_results.append({
                        "filename": os.path.basename(image_path),
                        "cell_count": None,
                        "foci_count": None,
                        "well": folder_name,
                        "passage": passage_name,
                        "cell_line": cell_line_name,
                        "bad": True,
                    })
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")

if channel_results:
    df = pd.DataFrame(channel_results)
    csv_path = os.path.join(results_base, "Results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved overall summary CSV: {csv_path}")