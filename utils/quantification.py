import numpy as np
import pandas as pd
import cv2
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, disk
from skimage import filters
from skimage.filters.rank import entropy
from skimage.filters import threshold_multiotsu
from scipy.stats import kurtosis

# After testing
MAX_INDIVIDUAL_HER2_AREA = 80

# Deconvolution
def deconvolve_her2_cen17(image):
    def rgb_to_od(img):
        img = img.astype(np.float32) + 1
        return -np.log10(img / 255.0)

    stain_matrix = np.array([
        [0.533, 0.653, 0.538],  # HER2 (black)
        [0.273, 0.892, 0.360],  # CEN17 (pink)
        [0.0,   0.0,   0.0   ]  
    ])

    stain_matrix[2, :] = np.cross(stain_matrix[0, :], stain_matrix[1, :])
    
    stain_matrix /= np.linalg.norm(stain_matrix, axis=1)[:,np.newaxis]

    od = rgb_to_od(image)
    
    concentrations = np.dot(od.reshape(-1, 3), np.linalg.inv(stain_matrix))
    conc_img = concentrations.reshape(image.shape)

    return conc_img[:, :, 0], conc_img[:, :, 1] # HER2, CEN17

# CEN17 Detection
def detect_cen17(channel, image):
    norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # remove noise
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    remove_mask = ((hue >= 25) & (hue <= 130)) | (val < 50) | (sat < 120)
    mask[remove_mask] = 0

    labeled_mask = label(mask)
    cleaned_labeled_mask = remove_small_objects(labeled_mask, min_size=10)

    clean_mask = np.zeros_like(mask, dtype=np.uint8)
    for region in regionprops(cleaned_labeled_mask):
        if region.solidity > 0.5:
            for (row, col) in region.coords:
                clean_mask[row, col] = 255
                
    vis = image.copy()
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    return clean_mask, vis

# HER2 Detection
def detect_her2(channel, cen17_mask, image):
    norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    mean_val = np.mean(norm)
    foreground_ratio = (norm > filters.threshold_otsu(norm)).sum() / norm.size
    laplacian_var = cv2.Laplacian(norm, cv2.CV_64F).var()
    hist_kurtosis = kurtosis(np.histogram(norm, bins=256)[0])
    entropy_val = entropy(norm, disk(5)).mean()
    sd = np.std(norm)

    # mutli-Otsu thresholding 
    thresholds = threshold_multiotsu(norm, classes=5)
    if foreground_ratio < 0.02:
        threshold_used = thresholds[1]
    elif (mean_val < 75) or ((mean_val < 85) and (sd < 6) and (entropy_val < 2)):
        threshold_used = thresholds[2]
    else:
        threshold_used = thresholds[3]

    mask = (norm > threshold_used).astype(np.uint8) * 255

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue_mask = None

    # remove noise
    if entropy_val > 2.5 and foreground_ratio > 0.4:
        blue_mask = ((h >= 85) & (h <= 120)) & (v >= 150)
        
    elif sd < 5 and foreground_ratio > 0.2:
        blue_mask = ((h >= 85) & (h <= 125)) | (v >= 140)
        
    elif (entropy_val < 2.5 and foreground_ratio > 0.3) or (mean_val > 120 and foreground_ratio > 0.25):
        blue_mask = ((h >= 85) & (h <= 135)) & (v >= 135)
        
    elif entropy_val > 2 and foreground_ratio < 0.05:
        blue_mask = None
        
    elif mean_val > 90 and laplacian_var < 10 and foreground_ratio < 0.15:
        blue_mask = ((h >= 85) & (h <= 135)) & (v >= 160)
        
    else:
        blue_mask = (h >= 85) & (h <= 135)
        

    if blue_mask is not None:
        mask[blue_mask] = 0

    her2_labels = label(mask)
    cen17_labels = label(cen17_mask)
    clean_mask = np.zeros_like(mask, dtype=np.uint8)

    # remove small region and those overalapped with CEN17 mask 
    for region in regionprops(her2_labels):
        if region.area > 5:
            coords = tuple(zip(*region.coords))
            if foreground_ratio > 0.05:
                overlap_pixels = sum(cen17_labels[r, c] != 0 for r, c in region.coords)
                overlap_ratio = overlap_pixels / region.area
                if overlap_ratio <= 0.2:
                    clean_mask[coords] = 255
            else:
                clean_mask[coords] = 255

    vis = image.copy()
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    return clean_mask, vis

def split_clusters(mask, reference_size = MAX_INDIVIDUAL_HER2_AREA, area_threshold = MAX_INDIVIDUAL_HER2_AREA, image=None):
    labeled = label(mask)
    props = regionprops(labeled)

    cluster_props = []
    cluster_contours = []
    cluster_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    her2_vis = image.copy() if image is not None else None

    for region in props:
        region_mask = (labeled == region.label).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = contours[0]

        if region.area > area_threshold:
            # Cluster signal
            cluster_area = region.area
            estimated_count = max(1, int(round(cluster_area / reference_size)))
            cluster_props.append({
                "cx": int(region.centroid[1]),
                "cy": int(region.centroid[0]),
                "area": cluster_area,
                "count": estimated_count
            })
            cluster_contours.append(contour)
            cv2.drawContours(cluster_mask, [contour], -1, 255, -1)
            if her2_vis is not None:
                cv2.drawContours(her2_vis, [contour], -1, (0, 255, 0), 2) 
                cv2.putText(her2_vis, str(estimated_count),
                            (int(region.centroid[1]), int(region.centroid[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            # Non-cluster (individual HER2 signal)
            if her2_vis is not None:
                cv2.drawContours(her2_vis, [contour], -1, (0, 255, 0), 2)  

    return cluster_props, cluster_contours, cluster_mask, her2_vis

def count_signals_per_nucleus(img, her2_mask, cen17_mask, nuclei_mask, MAX_INDIVIDUAL_HER2_AREA):
    cluster_props, cluster_contours, _, her2_vis_with_labels = split_clusters(
        her2_mask, MAX_INDIVIDUAL_HER2_AREA, MAX_INDIVIDUAL_HER2_AREA, image=img
    )
    
    nucleus_labels = label(nuclei_mask)
    pink_labels = label(cen17_mask)
    pink_regions = regionprops(pink_labels)
    black_labels = label(her2_mask)
    black_regions = regionprops(black_labels)
    single_black_regions = [r for r in black_regions if r.area <= MAX_INDIVIDUAL_HER2_AREA]

    cluster_mask_global = np.zeros_like(nucleus_labels, dtype=np.uint8)
    for cnt in cluster_contours:
        cv2.drawContours(cluster_mask_global, [cnt], -1, 1, -1)

    filtered_single_black_regions = [
        r for r in single_black_regions
        if cluster_mask_global[int(r.centroid[0]), int(r.centroid[1])] == 0
    ]

    filtered_nucleus_ids = [
        region.label for region in regionprops(nucleus_labels)
        if region.solidity >= 0.9
    ]

    results = []
    for nucleus_id in filtered_nucleus_ids:
        nucleus_mask_bin = (nucleus_labels == nucleus_id).astype(np.uint8)
        cen17_count = sum(1 for r in pink_regions if nucleus_mask_bin[int(r.centroid[0]), int(r.centroid[1])])
        her2_count = sum(1 for r in filtered_single_black_regions if nucleus_mask_bin[int(r.centroid[0]), int(r.centroid[1])])
        results.append({
            "nucleus_id": nucleus_id,
            "her2_count": her2_count,
            "cen17_count": cen17_count
        })

    df = pd.DataFrame(results)
    df['differentiation'] = df['her2_count'] - df['cen17_count']

    return df, filtered_single_black_regions, cluster_props, cluster_contours, nucleus_labels, pink_labels, black_labels, pink_regions, her2_vis_with_labels

def update_her2_with_clusters(df, cluster_props, cluster_contours, nucleus_labels):
    for cluster, cnt in zip(cluster_props, cluster_contours):
        cluster_mask = np.zeros_like(nucleus_labels, dtype=np.uint8)
        cv2.drawContours(cluster_mask, [cnt], -1, 1, -1)
        overlapping_nuclei = np.unique(nucleus_labels[(cluster_mask == 1) & (nucleus_labels > 0)])
        for nid in overlapping_nuclei:
            df.loc[df['nucleus_id'] == nid, 'her2_count'] += cluster['count']
    df['differentiation'] = df['her2_count'] - df['cen17_count']
    return df

def compute_ratios_and_status(df):
    results_summary = []

    # 1. All Nuclei
    all_nuclei = df[(df['her2_count'] >= 1) & (df['cen17_count'] >= 1)]

    her2_all = all_nuclei['her2_count'].sum()
    cen17_all = all_nuclei['cen17_count'].sum()

    ratio_all = round(her2_all / cen17_all, 2)
    status_all = 'Amplified' if ratio_all >= 2.0 else 'Non-Amplified'

    results_summary.append({
        "Method": "All Nuclei",
        "Total_Nuclei": len(all_nuclei), 
        "HER2": her2_all,
        "CEN17": cen17_all, 
        "Ratio": ratio_all, 
        "Status": status_all
    })

    # 2. Selection Criteria
    count_2_2 = len(df[(df['her2_count'] >= 2) & (df['cen17_count'] >= 2)])
    count_2_1 = len(df[(df['her2_count'] >= 2) & (df['cen17_count'] >= 1)])

    if count_2_2 >= 20:
        filtered = df[(df['her2_count'] >= 2) & (df['cen17_count'] >= 2)]
        method_used = "Meet Criteria (2 HER2:2 CEN17)"
    elif count_2_1 >= 20:
        filtered = df[(df['her2_count'] >= 2) & (df['cen17_count'] >= 1)]
        method_used = "Meet Criteria (2 HER2:1 CEN17)"
    else:
        filtered = df[(df['her2_count'] >= 1) & (df['cen17_count'] >= 1)]
        method_used = "Meet Criteria (1 HER2:1 CEN17)"

    filtered = filtered.sort_values(by='differentiation', ascending=False)
    
    her2_filter, cen17_filter = filtered['her2_count'].sum(), filtered['cen17_count'].sum()

    ratio_filter = round(her2_filter / cen17_filter, 2) if cen17_filter else np.nan
    status_filter = 'Amplified' if ratio_filter >= 2.0 else 'Non-Amplified'
    
    results_summary.append({
        "Method": method_used,
        "Total_Nuclei": len(filtered), 
        "HER2": her2_filter,
        "CEN17": cen17_filter, 
        "Ratio": ratio_filter, 
        "Status": status_filter
    })

    # 3. Select 20 Nuclei
    selected_20 = filtered.head(20)
    ratio_20 = round(selected_20['her2_count'].sum() / selected_20['cen17_count'].sum(), 2) if selected_20['cen17_count'].sum() else np.nan
    if 1.8 <= ratio_20 <= 2.2 and len(filtered) >= 40:
        combined = pd.concat([selected_20, filtered.iloc[20:40]])
        method_20 = "20+20 Method (Extended to 40)"
    else:
        combined = selected_20
        method_20 = "20+20 Method (Top 20)"

    her2_20 = combined['her2_count'].sum()
    cen17_20 = combined['cen17_count'].sum()
    ratio_20_final = round(her2_20 / cen17_20, 2) if cen17_20 else np.nan
    status_20 = 'Amplified' if ratio_20_final >= 2.0 else 'Non-Amplified'
    results_summary.append({
        "Method": method_20,
        "Total_Nuclei": len(combined), 
        "HER2": her2_20,
        "CEN17": cen17_20, 
        "Ratio": ratio_20_final, 
        "Status": status_20
    })

    return results_summary, combined 



def run_quantification(img, nuclei_mask, MAX_INDIVIDUAL_HER2_AREA = 80):

    # Deconvolution
    her2_channel, cen17_channel = deconvolve_her2_cen17(img)

    # Detect signals
    cen17_mask, cen17_vis = detect_cen17(cen17_channel, img)
    her2_mask, her2_vis = detect_her2(her2_channel, cen17_mask, img)

    # Count signals
    df, filtered_black, cluster_props, cluster_contours, \
    nucleus_labels, pink_labels, black_labels, pink_regions, her2_vis_labels = count_signals_per_nucleus(
        img, her2_mask, cen17_mask, nuclei_mask, MAX_INDIVIDUAL_HER2_AREA
    )   

    df = update_her2_with_clusters(df, cluster_props, cluster_contours, nucleus_labels)

    # Compute simple ratio 
    results_summary, selected_nuclei_df = compute_ratios_and_status(df)

    # All nuclei 
    selected_nuclei_overlay = img.copy()
    for region in regionprops(label(nuclei_mask)):
        contours, _ = cv2.findContours((label(nuclei_mask) == region.label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(selected_nuclei_overlay, contours, -1, (255, 0, 0), 2)

    # All nuclei and signals
    contour_overlay = np.ones_like(img) * 255
    for region in regionprops(label(nuclei_mask)):
        contours, _ = cv2.findContours((label(nuclei_mask) == region.label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 2)
    for region in regionprops(label(cen17_mask)):
        contours, _ = cv2.findContours((label(cen17_mask) == region.label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_overlay, contours, -1, (255, 0, 255), -1)
    for region in regionprops(label(her2_mask)):
        contours, _ = cv2.findContours((label(her2_mask) == region.label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_overlay, contours, -1, (0, 0, 0), -1)
    for cnt in cluster_contours:
        cv2.drawContours(contour_overlay, [cnt], -1, (0, 0, 0), -1)

    # Quantified nuclei and signals
    selected_overlay = np.ones_like(img) * 255
    top_method = results_summary[-1]
    selected_ids = selected_nuclei_df['nucleus_id'].tolist()

    for nid in selected_ids:
        nucleus_mask = (nucleus_labels == nid).astype(np.uint8)
        contours, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(selected_overlay, contours, -1, (0, 255, 0), 2)
    for region in pink_regions:
        if nucleus_labels[int(region.centroid[0]), int(region.centroid[1])] in selected_ids:
            contours, _ = cv2.findContours((pink_labels == region.label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(selected_overlay, contours, -1, (255, 0, 255), -1)
    for region in filtered_black:
        if nucleus_labels[int(region.centroid[0]), int(region.centroid[1])] in selected_ids:
            contours, _ = cv2.findContours((black_labels == region.label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(selected_overlay, contours, -1, (0, 0, 0), -1)
    for cluster, cnt in zip(cluster_props, cluster_contours):
        cluster_mask = np.zeros_like(nucleus_labels, dtype=np.uint8)
        cv2.drawContours(cluster_mask, [cnt], -1, 1, -1)
        overlapping = np.unique(nucleus_labels[(cluster_mask == 1) & (nucleus_labels > 0)])
        if any(n in selected_ids for n in overlapping):
            cv2.drawContours(selected_overlay, [cnt], -1, (0, 0, 0), -1)

    final_result = next(
        r for r in results_summary
        if "20+20" in r["Method"]
    )


    return {
    "HER2": final_result["HER2"],
    "CEN17": final_result["CEN17"],
    "ratio": final_result["Ratio"],
    "status": final_result['Status'],
    "original": img,
    "detection_overlay": contour_overlay,
    "quantification_overlay": selected_overlay,
    }
