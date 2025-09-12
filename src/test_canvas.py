import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os

st.title("Canvas Test - Simplified")

# Load a test image
image_files = [f for f in os.listdir("output/image_files") if f.endswith('.png')]
if image_files:
    test_image_path = os.path.join("output/image_files", image_files[0])
    
    st.write(f"Testing with: {test_image_path}")
    
    # Load image directly with PIL
    pil_img = Image.open(test_image_path)
    st.write(f"PIL Image - Mode: {pil_img.mode}, Size: {pil_img.size}")
    
    # Test 1: Display image normally
    st.subheader("Normal Image Display")
    st.image(pil_img, caption="Direct PIL display", use_container_width=True)
    
    # Test 2: Canvas with PIL image directly
    st.subheader("Canvas with PIL Image")
    try:
        canvas = st_canvas(
            background_image=pil_img,
            width=400,
            height=300,
            drawing_mode="line",
            stroke_color="#ff0000",
            stroke_width=3,
            update_streamlit=True,
            key="test_canvas_1",
        )
        st.write("Canvas created successfully with PIL image")
    except Exception as e:
        st.error(f"Canvas with PIL failed: {e}")
    
    # Test 3: Convert to grayscale and test
    st.subheader("Grayscale Test")
    if pil_img.mode != 'L':
        pil_gray = pil_img.convert('L')
    else:
        pil_gray = pil_img
    
    st.write(f"Grayscale PIL - Mode: {pil_gray.mode}, Size: {pil_gray.size}")
    st.image(pil_gray, caption="Grayscale PIL", use_container_width=True)
    
    # Test 4: Canvas with grayscale image
    st.subheader("Canvas with Grayscale Image")
    try:
        canvas2 = st_canvas(
            background_image=pil_gray,
            width=400,
            height=300,
            drawing_mode="line",
            stroke_color="#ff0000",
            stroke_width=3,
            update_streamlit=True,
            key="test_canvas_2",
        )
        st.write("Canvas created successfully with grayscale image")
    except Exception as e:
        st.error(f"Canvas with grayscale failed: {e}")
    
    # Test 5: Convert grayscale back to RGB
    st.subheader("Grayscale to RGB Conversion Test")
    pil_rgb = pil_gray.convert('RGB')
    st.write(f"Converted RGB PIL - Mode: {pil_rgb.mode}, Size: {pil_rgb.size}")
    st.image(pil_rgb, caption="Grayscale converted to RGB", use_container_width=True)
    
    try:
        canvas3 = st_canvas(
            background_image=pil_rgb,
            width=400,
            height=300,
            drawing_mode="line",
            stroke_color="#ff0000",
            stroke_width=3,
            update_streamlit=True,
            key="test_canvas_3",
        )
        st.write("Canvas created successfully with converted RGB image")
    except Exception as e:
        st.error(f"Canvas with converted RGB failed: {e}")
        
else:
    st.error("No image files found")