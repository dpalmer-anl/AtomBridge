import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import base64

def simple_get_scale_from_user_streamlit(img_gray, canvas_key="scale_canvas"):
    """Simplified scale measurement without complex compatibility shims"""
    st.subheader("Measure Scale Bar")
    st.write("Draw a red line along the scale bar, then enter its length (nm) and click Confirm scale.")

    # Convert to PIL RGB directly
    if isinstance(img_gray, np.ndarray):
        # Ensure uint8
        if img_gray.dtype != np.uint8:
            img_gray = ((img_gray - img_gray.min()) * 255 / (img_gray.max() - img_gray.min())).astype(np.uint8)
        
        # Convert grayscale to RGB
        if img_gray.ndim == 2:
            pil_img = Image.fromarray(img_gray, mode='L').convert('RGB')
        else:
            pil_img = Image.fromarray(img_gray)
    else:
        pil_img = img_gray
    
    # Calculate display dimensions
    iw, ih = pil_img.size
    disp_w = max(1, min(900, iw))
    scale_disp_to_orig = iw / disp_w if disp_w > 0 else 1.0
    disp_h = int(round(ih / scale_disp_to_orig)) if scale_disp_to_orig > 0 else ih

    # Create canvas
    canvas = st_canvas(
        background_image=pil_img,
        width=disp_w,
        height=disp_h,
        drawing_mode="line",
        stroke_color="#ff0000",
        stroke_width=3,
        update_streamlit=True,
        key=canvas_key,
    )

    pixel_length = None
    if canvas.json_data and canvas.json_data.get("objects"):
        for obj in reversed(canvas.json_data["objects"]):
            if obj.get("type") == "line":
                x1, y1 = float(obj.get("x1", 0)), float(obj.get("y1", 0))
                x2, y2 = float(obj.get("x2", 0)), float(obj.get("y2", 0))
                disp_len = float(np.hypot(x2 - x1, y2 - y1))
                pixel_length = disp_len * scale_disp_to_orig
                st.write(f"Current line length: {pixel_length:.2f} px")
                break

    real_nm = st.number_input("Scale bar length (nm)", min_value=0.0, step=0.1, value=0.0, key=canvas_key + "_nm")
    if st.button("Confirm scale", key=canvas_key + "_confirm"):
        if pixel_length and pixel_length > 0 and real_nm > 0:
            return real_nm / pixel_length
        st.warning("Draw a line and enter a positive number.")
    return None

def simple_custom_select_roi_streamlit(img_gray, canvas_key="roi_canvas"):
    """Simplified ROI selection without complex compatibility shims"""
    st.subheader("Select ROI")
    st.write("Draw a green rectangle around the analysis region, then click Confirm ROI.")

    # Convert to PIL RGB directly
    if isinstance(img_gray, np.ndarray):
        # Ensure uint8
        if img_gray.dtype != np.uint8:
            img_gray = ((img_gray - img_gray.min()) * 255 / (img_gray.max() - img_gray.min())).astype(np.uint8)
        
        # Convert grayscale to RGB
        if img_gray.ndim == 2:
            pil_img = Image.fromarray(img_gray, mode='L').convert('RGB')
        else:
            pil_img = Image.fromarray(img_gray)
    else:
        pil_img = img_gray
    
    # Calculate display dimensions
    iw, ih = pil_img.size
    disp_w = max(1, min(900, iw))
    scale_disp_to_orig = iw / disp_w if disp_w > 0 else 1.0
    disp_h = int(round(ih / scale_disp_to_orig)) if scale_disp_to_orig > 0 else ih

    # Create canvas
    canvas = st_canvas(
        background_image=pil_img,
        width=disp_w,
        height=disp_h,
        drawing_mode="rect",
        stroke_color="#00ff00",
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        update_streamlit=True,
        key=canvas_key,
    )

    roi = None
    if canvas.json_data and canvas.json_data.get("objects"):
        for obj in reversed(canvas.json_data["objects"]):
            if obj.get("type") == "rect":
                left = float(obj.get("left", 0.0))
                top = float(obj.get("top", 0.0))
                width = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))
                height = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0))
                x = int(round(left * scale_disp_to_orig))
                y = int(round(top * scale_disp_to_orig))
                w = int(round(width * scale_disp_to_orig))
                h = int(round(height * scale_disp_to_orig))
                roi = (max(0, x), max(0, y), max(0, w), max(0, h))
                st.write(f"Current ROI (px): {roi}")
                break

    if st.button("Confirm ROI", key=canvas_key + "_confirm"):
        if roi and roi[2] > 0 and roi[3] > 0:
            return roi
        st.warning("Draw a non-empty rectangle.")
    return None