import cv2
import numpy as np
import streamlit as st
from measureTEM_simple import simple_get_scale_from_user_streamlit, simple_custom_select_roi_streamlit
import os

st.title("Simplified Canvas Test")

# Load a test image
image_files = [f for f in os.listdir("output/image_files") if f.endswith('.png')]
if image_files:
    test_image_path = os.path.join("output/image_files", image_files[0])
    
    st.write(f"Testing with: {test_image_path}")
    
    # Load image with OpenCV as grayscale
    img_gray = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    
    if img_gray is not None:
        st.write(f"Image shape: {img_gray.shape}, dtype: {img_gray.dtype}")
        st.write(f"Min: {img_gray.min()}, Max: {img_gray.max()}")
        
        # Show original image
        st.subheader("Original Image")
        st.image(img_gray, caption="Loaded grayscale image", use_container_width=True)
        
        # Test simplified scale function
        st.subheader("Scale Measurement Test")
        scale_result = simple_get_scale_from_user_streamlit(img_gray, canvas_key="test_scale")
        if scale_result:
            st.success(f"Scale measured: {scale_result}")
        
        # Test simplified ROI function
        st.subheader("ROI Selection Test")
        roi_result = simple_custom_select_roi_streamlit(img_gray, canvas_key="test_roi")
        if roi_result:
            st.success(f"ROI selected: {roi_result}")
            
    else:
        st.error("Failed to load image")
else:
    st.error("No image files found")