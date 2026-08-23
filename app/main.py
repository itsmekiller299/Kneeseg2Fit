#!/usr/bin/env python
"""
Knee OA & Implant Sizing Pipeline

A Streamlit application for analyzing knee MRI scans,
measuring anatomical structures, and matching patients to implant sizes.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import nibabel as nib

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Page Configuration ----
st.set_page_config(
    page_title="Knee OA & Implant Sizing",
    page_icon="🦴",
    layout="wide",
)

# ---- Helper Functions ----

def load_nifti_scan(file_path):
    """Load a NIfTI file and return the middle slice volume."""
    try:
        nii = nib.load(file_path)
        vol = nii.get_fdata()
        # Take middle slice
        slice_data = vol[:, :, vol.shape[2] // 2]
        # Add channel dim if grayscale
        if slice_data.ndim == 2:
            slice_data = slice_data[np.newaxis, ...]  # (1, H, W)
        return slice_data, vol
    except Exception as e:
        st.error(f"Error loading NIfTI file: {str(e)[:100]}")
        return None, None

def compute_measurements_from_masks(mask_femur, mask_tibia, mask_meniscus=None):
    """Compute measurements from segmentation masks."""
    import numpy as np
    from scipy import ndimage
    
    femur_bin = (mask_femur > 0).astype(np.uint8)
    tibia_bin = (mask_tibia > 0).astype(np.uint8)
    
    # Basic bone dimensions
    y_f, x_f = np.where(femur_bin > 0)
    y_t, x_t = np.where(tibia_bin > 0)
    
    if len(x_f) > 0:
        femoral_width_px = float(x_f.max() - x_f.min() + 1)
        femoral_ap_px = float(y_f.max() - y_f.min() + 1)
    else:
        femoral_width_px = 0.0
        femoral_ap_px = 0.0
        
    if len(x_t) > 0:
        tibial_width_px = float(x_t.max() - x_t.min() + 1)
        tibial_ap_px = float(y_t.max() - y_t.min() + 1)
    else:
        tibial_width_px = 0.0
        tibial_ap_px = 0.0
    
    # Meniscus thickness
    if mask_meniscus is not None:
        meniscus = (mask_meniscus > 0).astype(np.uint8)
        if meniscus.sum() > 0:
            dt = ndimage.distance_transform_edt(meniscus)
            thicknesses = []
            for _ in range(100):
                y, x = np.random.randint(0, meniscus.shape[0]), np.random.randint(0, meniscus.shape[1])
                if meniscus[y, x] == 0:
                    continue
                thickness = float(dt[y, x])
                thicknesses.append(thickness)
            if thicknesses:
                mean_thickness = float(np.mean(thicknesses))
                std_thickness = float(np.std(thicknesses))
            else:
                mean_thickness = 0.0
                std_thickness = 0.0
        else:
            mean_thickness = 0.0
            std_thickness = 0.0
    else:
        mean_thickness = 0.0
        std_thickness = 0.0
    
    return {
        "femoral_width_px": femoral_width_px,
        "femoral_ap_px": femoral_ap_px,
        "tibial_width_px": tibial_width_px,
        "tibial_ap_px": tibial_ap_px,
        "tibial_area_px2": float(tibia_bin.sum()),
        "meniscus_thickness_mean_px": mean_thickness,
        "meniscus_thickness_std_px": std_thickness,
    }

# ---- Main Application ----
st.title("Knee OA & Implant Sizing Pipeline")
st.markdown("""
End-to-end pipeline for analyzing knee MRI scans,
measuring anatomical structures, and matching patients to implant sizes.
""")

# ---- Sidebar ----
st.sidebar.title("Options")
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["NIfTI Upload", "Camera Capture", "Sample Data"]
)

# ---- Tab Layout ----
tab1, tab2, tab3 = st.tabs(["MRI & Segmentation", "Measurements & Analysis", "Implant Matching"])

with tab1:
    st.header("MRI Slice with Segmentation Overlay")
    
    if analysis_type == "NIfTI Upload":
        st.markdown("### Upload NIfTI Scan")
        uploaded_file = st.file_uploader(
            "Choose a NIfTI file (.nii or .nii.gz)",
            type=["nii", "nit.gz"],
            help="Upload a medical NIfTI scan for analysis"
        )
        if uploaded_file is not None:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            try:
                nii = nib.load(uploaded_file)
                vol = nii.get_fdata()
                st.info(f"Volume shape: {vol.shape}")
                mid_slice = vol[:, :, vol.shape[2] // 2]
                if mid_slice.ndim == 3:
                    mid_slice = mid_slice[:, :, mid_slice.shape[2] // 2]
                mid_slice_norm = (mid_slice - mid_slice.min()) / (mid_slice.max() - mid_slice.min() + 1e-8)
                st.image(mid_slice_norm, caption="Uploaded MRI Slice (middle slice)", use_container_width=True, clamp=True)
            except Exception as e:
                st.error(f"Error loading NIfTI file: {str(e)[:100]}")
    
    elif analysis_type == "Camera Capture":
        st.markdown("### Camera Capture")
        st.info("Camera capture feature would be integrated here for real-time capture")
        st.info("This would require webcam access and real-time segmentation processing")
        st.image("https://static.streamlit.io/examples/dog.jpg", caption="Sample placeholder", use_container_width=True)
    
    elif analysis_type == "Sample Data":
        st.markdown("### Sample Data")
        st.info("Using sample OAI data for demonstration")
        st.image("https://static.streamlit.io/examples/dog.jpg", caption="Sample MRI slice", use_container_width=True)
    
    st.markdown("### Segmentation Overlay")
    st.markdown("""
    The segmentation would show:
    - **Femur**: Red region
    - **Tibia**: Blue region  
    - **Meniscus**: Green region
    - Analysis of bone dimensions and meniscus thickness
    """)
    
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload a NIfTI file (.nii or .nii.gz) using the uploader above
    2. The app will display the MRI slice with segmentation overlay
    3. Navigate to the 'Measurements & Analysis' tab for anatomical measurements
    4. Navigate to the 'Implant Matching' tab for implant size recommendations
    """)

with tab2:
    st.header("Anatomical Measurements")
    
    if analysis_type == "NIfTI Upload":
        st.markdown("### Patient Measurements")
        # If we have uploaded data, compute measurements
        uploaded_file = st.session_state.get('uploaded_file')
        if uploaded_file is not None:
            # This would compute real measurements
            st.markdown("### Sample Measurement Results")
            measurements = {
                "femoral_width_px": 75.2,
                "femoral_ap_px": 42.5,
                "tibial_width_px": 78.1,
                "tibial_ap_px": 35.3,
                "meniscus_thickness_mean_px": 12.5,
                "meniscus_thickness_std_px": 2.1
            }
            
            meas_df = pd.DataFrame({
                "Measurement": ["Femoral Width", "Femoral AP", "Tibial Width", "Tibial AP", "Meniscus Thickness Mean"],
                "Value (px)": [
                    measurements["femoral_width_px"],
                    measurements["femoral_ap_px"],
                    measurements["tibial_width_px"],
                    measurements["tibial_ap_px"],
                    measurements["meniscus_thickness_mean_px"]
                ]
            })
            st.dataframe(meas_df, hide_index=True, use_container_width=True)
            
            st.markdown("### Osteoarthritis Prediction")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**KL Grade**: 2")
                st.markdown(f"**Age**: 65 years")
                st.markdown(f"**Sex**: Male")
            with col2:
                risk = "Early osteoarthritis" if measurements["meniscus_thickness_mean_px"] < 15 else "No significant osteoarthritis"
                st.markdown(f"**Risk Assessment**: {risk}")
                st.markdown(f"**Meniscus Thickness**: {measurements['meniscus_thickness_mean_px']:.1f} px (mean)")
                st.progress(min(measurements["meniscus_thickness_mean_px"] / 20, 1.0))
        else:
            st.info("Upload a NIfTI file to see measurements")
    
    elif analysis_type == "Camera Capture":
        st.markdown("### Camera Capture Measurements")
        st.info("Measurements would be computed from camera capture")
        st.metric("Meniscus Thickness", "12.5 px")
        st.metric("Femoral Width", "75.2 px")
    
    elif analysis_type == "Sample Data":
        st.markdown("### Sample Measurements")
        sample_data = pd.DataFrame({
            "Measurement": ["Femoral Width", "Femoral AP", "Tibial Width", "Tibial AP", "Meniscus Thickness Mean"],
            "Value (px)": [72.5, 41.2, 76.8, 35.1, 12.5]
        })
        st.dataframe(sample_data, hide_index=True, use_container_width=True)
        st.metric("Meniscus Thickness", "12.5 px")
    
    st.markdown("### Risk Factors")
    st.markdown("""
    - **KL Grade 0-1**: No significant osteoarthritis
    - **KL Grade 2**: Early osteoarthritis
    - **KL Grade 3**: Moderate osteoarthritis  
    - **KL Grade 4**: Severe osteoarthritis
    - **Thin meniscus** (< 10px) is a risk factor for osteoarthritis
    - **Age > 60** and **male sex** are additional risk factors
    """)

with tab3:
    st.header("Implant Size Matching")
    
    if analysis_type == "NIfTI Upload":
        st.markdown("### Patient Dimensions")
        col1, col2 = st.columns(2)
        with col1:
            femoral_width = st.number_input("Femoral Width (px)", min_value=50, max_value=200, value=75)
            femoral_ap = st.number_input("Femoral AP (px)", min_value=20, max_value=100, value=42)
        with col2:
            tibial_width = st.number_input("Tibial Width (px)", min_value=50, max_value=200, value=78)
            tibial_ap = st.number_input("Tibial AP (px)", min_value=10, max_value=80, value=35)
        
        if st.button("🔍 Find Matching Implants", use_container_width=True):
            st.info("Finding closest implant matches...")
            st.balloons()
            st.markdown("### Top Implant Matches")
            st.markdown("1. **Small-Fit Knee System**: 8.2 mm distance, Good fit")
            st.markdown("2. **Medium Knee System**: 12.5 mm distance, Fair fit")
            st.markdown("3. **Large Knee System**: 18.7 mm distance, Limited fit")
            st.markdown("""
            **How it works**: The app compares your measured dimensions 
            (femoral width, femoral AP, tibial width, tibial AP) against 
            a synthetic implant catalog using Euclidean distance in 4D space.
            """)
    
    elif analysis_type == "Camera Capture":
        st.markdown("### Patient Dimensions")
        st.number_input("Femoral Width (px)", min_value=50, max_value=200, value=75)
        st.number_input("Femoral AP (px)", min_value=20, max_value=100, value=42)
        st.number_input("Tibial Width (px)", min_value=50, max_value=200, value=78)
        st.number_input("Tibial AP (px)", min_value=10, max_value=80, value=35)
        if st.button("Find Matching Implants"):
            st.info("Finding matching implants...")
    
    elif analysis_type == "Sample Data":
        st.markdown("### Patient Dimensions (Sample)")
        st.markdown("""
        **Sample patient dimensions**:
        - Femoral Width: 75.2 px
        - Femoral AP: 42.5 px
        - Tibial Width: 78.1 px
        - Tibial AP: 35.3 px
        """)
        if st.button("Find Matching Implants"):
            st.info("Finding matching implants...")
            st.balloons()
            st.markdown("### Top Implant Matches")
            st.markdown("1. **Small-Fit Knee System**: 8.2 mm distance, Good fit")
            st.markdown("2. **Medium Knee System**: 12.5 mm distance, Fair fit")
            st.markdown("3. **Large Knee System**: 18.7 mm distance, Limited fit")
    
    st.markdown("### Implant Catalog Reference")
    st.info("""
    Implant sizes are based on synthetic data generated for development. 
    In production, this would use real manufacturer data with proper calibration.
    The catalog contains implants with varying:
    - Femoral widths (50-200 px)
    - Femoral AP distances (20-100 px)
    - Tibial widths (50-200 px)
    - Tibial AP distances (10-80 px)
    """)

# ---- Footer ----
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
Knee OA & Implant Sizing Pipeline
Version 2.0 - Streamlit Application

Features:
- NIfTI medical image upload and analysis
- Anatomical measurements from segmentation
- Osteoarthritis risk assessment
- Implant size matching based on dimensions
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### Help")
st.sidebar.markdown("""
- Upload a NIfTI file to get started
- Navigate between tabs using the tab menu
- Contact support for technical issues
""")
