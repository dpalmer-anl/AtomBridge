import os
import shutil
import base64
import mimetypes
from pathlib import Path
import fitz  # PyMuPDF
import streamlit as st
import glob
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max

import extractfigures as legacy

# Import the interactive measurement utilities from measureTEM
from measureTEM import (
    # get_scale_from_user,
    # custom_select_roi,
    get_scale_from_user_streamlit,
    measure_atomic_spacing_realspace,
    custom_select_roi_streamlit,
)

# Helper to build an LLM-ready image payload (bytes + base64 + metadata)
def build_llm_image_payload(image_path: str, caption: str):
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/png" if image_path.lower().endswith(".png") else "application/octet-stream"
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return {
        "filename": os.path.basename(image_path),
        "path": image_path,
        "mime": mime,
        "bytes": data,
        "base64": b64,
        "caption": caption,
    }

# Streamlit app
st.image("../AtomBridge.jpg", use_container_width=True)

# Initialize session state
if "figure_data" not in st.session_state:
    st.session_state.figure_data = None
if "images" not in st.session_state:
    st.session_state.images = []
if "captions" not in st.session_state:
    st.session_state.captions = []
if "selected_image_path" not in st.session_state:
    st.session_state.selected_image_path = None
if "selected_caption" not in st.session_state:
    st.session_state.selected_caption = None
if "selected_image_payload" not in st.session_state:
    st.session_state.selected_image_payload = None
if "pixel_to_nm" not in st.session_state:
    st.session_state.pixel_to_nm = None
if "roi" not in st.session_state:
    st.session_state.roi = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
# NEW: UI mode flags
if "scale_mode" not in st.session_state:
    st.session_state.scale_mode = False
if "roi_mode" not in st.session_state:
    st.session_state.roi_mode = False

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Directory for the output
outputDirectory = "output"
text_output_directory = os.path.join(outputDirectory, "text_files")
image_output_directory = os.path.join(outputDirectory, "image_files")
os.makedirs(outputDirectory, exist_ok=True)
os.makedirs(text_output_directory, exist_ok=True)
os.makedirs(image_output_directory, exist_ok=True)

# Dropdown for model selection
llm_type = st.selectbox("Select the model type:", ("GPT-5", "Gemini-2.5-Pro", "Claude-3.5", "Claude-4"))

# Process PDFs
if uploaded_files and st.button("Process PDFs"):
    if os.path.isdir(image_output_directory):
        shutil.rmtree(image_output_directory)
    os.makedirs(image_output_directory, exist_ok=True)

    legacy.output_folder = Path(image_output_directory)

    if not hasattr(legacy, "process_pdf") or not callable(getattr(legacy, "process_pdf")):
        st.error("old_extractfigures.process_pdf is not available. Please ensure old_extractfigures.py contains a process_pdf(doc) function.")
        st.stop()

    dfs = []
    for uf in uploaded_files:
        file_path = os.path.join(outputDirectory, uf.name)
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())

        try:
            with fitz.open(file_path) as doc:
                records = legacy.process_pdf(doc)
        except Exception as e:
            st.exception(e)
            continue

        if records:
            dfs.append(pd.DataFrame(records))

    if dfs:
        figure_data = pd.concat(dfs, ignore_index=True)
        figure_data["figure_path_files"] = figure_data["image_file"].apply(
            lambda x: os.path.join(image_output_directory, x)
        )
        st.session_state.figure_data = figure_data
        st.session_state.images = figure_data["figure_path_files"].tolist()
        st.session_state.captions = figure_data["caption"].tolist()
        st.session_state.selected_image_path = None
        st.session_state.selected_caption = None
        st.session_state.selected_image_payload = None
        st.success(f"Processed {len(uploaded_files)} PDF(s).")
    else:
        st.session_state.figure_data = None
        st.session_state.images = []
        st.session_state.captions = []
        st.session_state.selected_image_path = None
        st.session_state.selected_caption = None
        st.session_state.selected_image_payload = None
        st.warning("No figures found in the uploaded PDF(s).")

# API key and prompt inputs (placeholder for later LLM integration)
api = st.text_input("Provide API key:", type="password")
prompt = st.text_area("Enter the prompt for the LLM model:")

# Display figures if available
if st.session_state.figure_data is not None and len(st.session_state.images) > 0:
    images = st.session_state.images
    captions = st.session_state.captions

    selected_index = st.slider("Select an image", 0, len(images) - 1, 0)
    st.image(images[selected_index], caption=captions[selected_index], use_container_width=True)
    current_image_path = images[selected_index]

    # Select current image for downstream LLM processing
    if st.button("Use this image for post-processing"):
        st.session_state.selected_image_path = current_image_path
        st.session_state.selected_caption = captions[selected_index]
        st.session_state.selected_image_payload = build_llm_image_payload(
            current_image_path, captions[selected_index]
        )
        st.success(f"Selected: {os.path.basename(current_image_path)} for LLM processing.")

    # Buttons toggle Streamlit-native interactive modes
    if st.button("Measure Scale Bar"):
        st.session_state.scale_mode = True
    if st.button("Select ROI"):
        st.session_state.roi_mode = True

    # Streamlit-native scale measurement flow
    if st.session_state.scale_mode:
        img_gray = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            st.error("Failed to load the selected image.")
        else:
            ratio = get_scale_from_user_streamlit(img_gray, canvas_key=f"scale_canvas_{selected_index}")
            if ratio is not None:
                st.session_state.pixel_to_nm = ratio
                st.session_state.scale_mode = False
                st.success(f"Calculated pixel-to-nm ratio: {ratio:.6f}")

    # Streamlit-native ROI selection flow
    if st.session_state.roi_mode:
        img_gray = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            st.error("Failed to load the selected image.")
        else:
            roi = custom_select_roi_streamlit(img_gray, canvas_key=f"roi_canvas_{selected_index}")
            if roi is not None:
                st.session_state.roi = roi
                st.session_state.roi_mode = False
                st.success(f"Selected ROI: {roi}")

    # Analyze the selected ROI when both scale and ROI are available
    if st.session_state.roi and st.session_state.pixel_to_nm:
        img_full_gray = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
        if img_full_gray is not None:
            x, y, w, h = st.session_state.roi
            img_roi_gray = img_full_gray[y:y+h, x:x+w]
            measure_atomic_spacing_realspace(img_roi_gray, st.session_state.pixel_to_nm)
        else:
            st.error("Failed to load the selected image.")

# # Preview the selected image payload (for downstream use)
# if st.session_state.selected_image_payload:
#     sel = st.session_state.selected_image_payload
#     st.subheader("Selected image for LLM (prepared)")
#     st.write(f"File: {sel['filename']}  |  MIME: {sel['mime']}")
#     st.write(f"Caption: {sel['caption']}")
#     st.image(sel["path"], caption="Selected", use_container_width=True)