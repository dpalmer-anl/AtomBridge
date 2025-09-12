import os
import shutil
import base64
import mimetypes

import streamlit as st
import glob
import pandas as pd

from figures_and_captions import process_pdf

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
        "bytes": data,          # for APIs that accept raw bytes
        "base64": b64,          # for APIs that require base64
        "caption": caption,
    }

# Streamlit app
st.title("⚛️ AtomBridge Graphical Interface")

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

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

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
    # Clean image output to avoid mixing old and new files
    if os.path.isdir(image_output_directory):
        shutil.rmtree(image_output_directory)
    os.makedirs(image_output_directory, exist_ok=True)

    dfs = []
    for uf in uploaded_files:
        file_path = os.path.join(outputDirectory, uf.name)
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())

        df = process_pdf(file_path, image_output_directory)
        if df is not None and not df.empty:
            dfs.append(df)

    if dfs:
        figure_data = pd.concat(dfs, ignore_index=True)
        figure_data["figure_path_files"] = figure_data["image_file"].apply(
            lambda x: os.path.join(image_output_directory, x)
        )
        st.session_state.figure_data = figure_data
        st.session_state.images = figure_data["figure_path_files"].tolist()
        st.session_state.captions = figure_data["caption"].tolist()
        # reset any previous selection
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

    # Select current image for downstream LLM processing
    if st.button("Use this image for LLM"):
        st.session_state.selected_image_path = images[selected_index]
        st.session_state.selected_caption = captions[selected_index]
        st.session_state.selected_image_payload = build_llm_image_payload(
            images[selected_index], captions[selected_index]
        )
        st.success(f"Selected: {os.path.basename(images[selected_index])} for LLM processing.")
else:
    st.info("Upload PDFs and click 'Process PDFs' to view and select extracted figures.")

# Display figures if available
# if st.session_state.figure_data is not None and len(st.session_state.images) > 0:
#     images = st.session_state.images
#     captions = st.session_state.captions

#     selected_index = st.slider("Select an image", 0, len(images) - 1, 0)
#     st.image(images[selected_index], caption=captions[selected_index], use_container_width=True)

#     # Download selected image
#     try:
#         with open(images[selected_index], "rb") as fh:
#             st.download_button(
#                 label="Download image",
#                 data=fh.read(),
#                 file_name=os.path.basename(images[selected_index]),
#             )
#     except FileNotFoundError:
#         st.warning("Selected image file not found on disk.")
# else:
#     st.info("Upload PDFs and click 'Process PDFs' to view extracted figures.")

# Preview the selected image payload (for downstream use)
if st.session_state.selected_image_payload:
    sel = st.session_state.selected_image_payload
    st.subheader("Selected image for LLM (prepared)")
    st.write(f"File: {sel['filename']}  |  MIME: {sel['mime']}")
    st.write(f"Caption: {sel['caption']}")
    # Optional: show again
    st.image(sel["path"], caption="Selected", use_container_width=True)

    # Example stub: you can call your LLM here with sel['bytes'] or sel['base64']
    # if st.button("Run LLM (stub)"):
    #     st.info("LLM call not implemented yet.")