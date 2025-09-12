import cv2
import numpy as np
import streamlit as st
from measureTEM import _to_pil
import os

st.title("Image Display Test")

# Test with an existing image from the output directory
image_files = [f for f in os.listdir("output/image_files") if f.endswith('.png')]

if image_files:
    test_image_path = os.path.join("output/image_files", image_files[0])
    st.write(f"Testing with: {test_image_path}")
    
    # Test 1: Direct streamlit display
    st.subheader("Direct Streamlit Display")
    st.image(test_image_path, caption="Direct display", use_container_width=True)
    
    # Test 2: Load with OpenCV and display
    st.subheader("OpenCV Load + Streamlit Display")
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        st.write(f"Image shape: {img.shape}, dtype: {img.dtype}")
        st.write(f"Min: {img.min()}, Max: {img.max()}")
        st.image(img, caption="OpenCV grayscale", use_container_width=True)
        
        # Test 3: Convert with our _to_pil function
        st.subheader("Using _to_pil Conversion")
        pil_img = _to_pil(img)
        if pil_img:
            st.write(f"PIL Image mode: {pil_img.mode}, size: {pil_img.size}")
            st.image(pil_img, caption="Converted with _to_pil", use_container_width=True)
    else:
        st.error("Failed to load image with OpenCV")
else:
    st.error("No image files found in output/image_files directory")