import numpy as np
import cv2
from cellpose import models
import torch

# Load BOTH models
def load_models():
    # custom_model = models.CellposeModel(
    #     gpu=False,
    #     pretrained_model="model/cellpose_HITL"
    # )

    custom_model = models.CellposeModel(gpu=False)
    weights_path = "model/cellpose_HITL_weights.pt"
    state_dict = torch.load(weights_path, map_location="cpu")
    custom_model.net.load_state_dict(state_dict)


    cyto3_model = models.Cellpose(
        gpu=False,
        model_type='cyto3'
    )

    return {
        "custom": custom_model,
        "cyto3": cyto3_model
    }

# Segmentation
def run_segmentation(model, img_np, channels=[[0, 3]]):
    result = model.eval([img_np], diameter=None, channels=channels)
    mask = result[0][0]
    return mask

def mask_to_display(mask):
    # normalize to 0–255
    # mask_norm = (mask.astype(np.float32) / mask.max()) * 255
    # return mask_norm.astype(np.uint8)

    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # remove background

    np.random.seed(42)  # consistent colors every run

    for label in unique_labels:
        color = np.random.randint(0, 255, size=3)
        colored_mask[mask == label] = color

    return colored_mask

# Draw contours
def draw_contours(image, mask):
    contour_img = image.copy()

    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]

    for label in unique_labels:
        binary = (mask == label).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)

    return contour_img, len(unique_labels)
