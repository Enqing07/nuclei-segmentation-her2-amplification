import streamlit as st
import numpy as np
from PIL import Image

from utils.segmentation import load_models, run_segmentation, draw_contours, mask_to_display
from utils.quantification import run_quantification


# Configuring the Page
st.set_page_config(
    page_title="Nuclei Segmentation and Quantification of HER2 Gene Amplification", 
    page_icon="🔬",
    layout="centered"
)

st.markdown("""
<style>
/* Remove rounded corners from images */
img {
    border-radius: 0px !important;
}
</style>
""", unsafe_allow_html=True)


# Header and Description
st.markdown(
    """
    <h1 style='text-align: center;'>🔬 Nuclei Segmentation and Quantification of HER2 Gene Amplification</h1>
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 25px;'>
        <p style='line-height: 1.6; text-align: justify; color: #31333F; margin: 0;'>
            This webapp performs automated nuclei segmentation and HER2 gene amplification quantification for pathology images.
            It integrates deep learning-based segmentation with signal detection to compute HER2/CEN17 ratios, supporting efficient breast cancer assessment.
            Use the sidebar to select the analysis mode and segmentation model.
        </p>
    </div>
    """, unsafe_allow_html=True
)

# Model Selection Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "**Select Mode:**",
        ("Single Model", "Model Comparison"),
    )
    st.divider()

    if mode == "Single Model":

        model_choice = st.radio(
            "**Select Segmentation Model:**",
            ["Cyto3 + HITL", "Cyto3"]
        )

        if "HITL" in model_choice:
            st.success("Best for HER2 pathology images.")
        else:
            st.info("General-purpose model.")

    else:
        st.info(
            "Compare segmentation results between:\n"
            "- Cyto3 (baseline)\n"
            "- Cyto3 + HITL (custom)\n\n"
            "Quantification is disabled in this mode."
        )

        model_choice = None

    # Reset session state if mode or model changed
    # Check if mode or model changed
    if "prev_mode" not in st.session_state:
        st.session_state["prev_mode"] = mode
    if "prev_model" not in st.session_state:
        st.session_state["prev_model"] = model_choice

    # Clear segmentation if mode or model changed
    if st.session_state["prev_mode"] != mode:
        for key in ["segmentation_result", "mask_display", "mask", "nuclei_count"]:
            st.session_state.pop(key, None)
        st.session_state["prev_mode"] = mode
    elif mode == "Single Model" and st.session_state["prev_model"] != model_choice:
        for key in ["segmentation_result", "mask_display", "mask", "nuclei_count"]:
            st.session_state.pop(key, None)
        st.session_state["prev_model"] = model_choice


# Load models
@st.cache_resource
def get_models():
    return load_models()

models_dict = get_models()
channels = [[0, 3]]

# Upload
uploaded_file = st.file_uploader(
    "Upload Pathology Image",
    type=["png", "jpg", "jpeg", "tif"]
)

# Analysis 
def run_analysis(img_np, mode, model_choice, models_dict, channels):
    status_placeholder = st.empty()
    status_placeholder.info("Analyzing...")

    st.header("1. Nuclei Segmentation")

    if mode == "Model Comparison":
        # comparison
        mask_custom = run_segmentation(models_dict["custom"], img_np, channels)
        contour_custom, count_custom = draw_contours(img_np, mask_custom)

        mask_cyto3 = run_segmentation(models_dict["cyto3"], img_np, channels)
        contour_cyto3, count_cyto3 = draw_contours(img_np, mask_cyto3)

        st.subheader("Segmentation Comparison")
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.image(contour_cyto3, caption="Cyto3", use_container_width=True)
            st.write(f"Nuclei Count: {count_cyto3}")

        with col2:
            st.image(contour_custom, caption="Cyto3 + HITL", use_container_width=True)
            st.write(f"Nuclei Count: {count_custom}")

        st.warning("Quantification is disabled in comparison mode.")

    else:
        # single model
        selected_model = models_dict["custom"] if "HITL" in model_choice else models_dict["cyto3"]

        st.caption(f"Model: {model_choice}")

        # Step 1: Segmentation
        mask = run_segmentation(selected_model, img_np, channels)
        contour_img, count = draw_contours(img_np, mask)
        mask_display = mask_to_display(mask)

        st.subheader("Result")
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.image(contour_img, caption="Contour", use_container_width=True)

        with col2:
            st.image(mask_display, caption="Mask", use_container_width=True)

        st.write(f"Nuclei Count: {count}")

        # Step 2: Quantification
        st.header("2. HER2 Quantification")
        st.caption("20 nuclei with the highest HER2–CEN17 signal differentiation values were selected to ensure robust and representative ratio computation." \
        "For each of these nuclei, the HER2/CEN17 ratio is calculated. A ratio greater than or equal to 2.0 indicates that HER2 is amplified, while a ratio below 2.0 indicates it is non-amplified.")
        result = run_quantification(img_np, mask)

        st.subheader("Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("HER2", result["HER2"])
        with col2:
            st.metric("CEN17", result["CEN17"])
        with col3:
            st.metric("Ratio", f"{result['ratio']:.2f}")

        st.write(f"### Status: {result['status']}")

        col1, col2, col3 = st.columns(3, gap="small")
        with col1:
            st.image(result["original"], caption="Original Image", use_container_width=True)
        with col2:
            st.image(result["detection_overlay"], caption="Detection Overview", use_container_width=True)
        with col3:
            st.image(result["quantification_overlay"], caption="Quantification", use_container_width=True)
        
    status_placeholder.success("Analysis Complete!")

# Initialize session state
for key in ["last_uploaded_file", "last_mode", "last_model"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Check if analysis needs to run
needs_analysis = False

if uploaded_file is not None:
    if (st.session_state["last_uploaded_file"] != uploaded_file or
        st.session_state["last_mode"] != mode or
        (mode == "Single Model" and st.session_state["last_model"] != model_choice)):

        # Update session state
        st.session_state["last_uploaded_file"] = uploaded_file
        st.session_state["last_mode"] = mode
        st.session_state["last_model"] = model_choice if mode == "Single Model" else None

        # Clear previous results
        for key in ["mask", "segmentation_result", "mask_display", "nuclei_count"]:
            st.session_state.pop(key, None)

        needs_analysis = True

    # Only run analysis once
    if needs_analysis:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        st.image(image, caption="Original Image", use_container_width=True)
        mc = model_choice if mode == "Single Model" else None
        run_analysis(img_np, mode, mc, models_dict, channels)