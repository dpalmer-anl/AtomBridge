import os
import shutil

import streamlit as st
import glob
import pandas as pd


from figures_and_captions import process_pdf

# Streamlit app
st.title("⚛️ AtomBridge Graphical Interface")

# Initialize session state
if "figure_data" not in st.session_state:
    st.session_state.figure_data = None
if "images" not in st.session_state:
    st.session_state.images = []
if "captions" not in st.session_state:
    st.session_state.captions = []

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

# Process cleaned text files to get embeddings and tokens
# key_file_path = "/Users/riteshk/Library/CloudStorage/Box-Box/Research-postdoc/oxRSE-project/API_KEY"  # Replace with the actual path to your OpenAI key file

# Display uploaded files
if uploaded_files:
    # st.write("Uploaded PDF files:")
    uploaded_file_names = [
        uploaded_file.name for uploaded_file in uploaded_files
    ]
    # for uploaded_file in uploaded_file_names:
    # st.write(uploaded_file)

    # Total number of files uploaded
    # totalFiles = len(uploaded_files)

    # Button to process PDFs
    if st.button("Process PDFs"):
        if not os.path.exists(outputDirectory):
            os.makedirs(outputDirectory)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(outputDirectory, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            figure_data = process_pdf(file_path, text_output_directory)

    #     cropAllPdfs(uploaded_file_names, outputDirectory, totalFiles)

    #     for uploaded_file in uploaded_files:
    #         file_path = os.path.join(outputDirectory, uploaded_file.name)
    #         process_pdf(file_path, text_output_directory)

    #     st.success("PDFs processed successfully.")
    #     st.write(f"Processed {totalFiles} PDFs.")
    #     # process_all_pdfs(uploaded_files, outputDirectory)
    #     # Process text files
    #     process_text_files(text_output_directory)
    #     st.write("Text files have been cleaned.")

    #     # # Process cleaned text files to get embeddings and tokens
    #     # key_file_path = (
    #     #     "/Users/riteshk/Library/CloudStorage/Box-Box/Research-postdoc/oxRSE-project/API_KEY"  # Replace with the actual path to your OpenAI key file
    #     # )
    #     # llm_type = (
    #     #     "Ollama"  # Specify the LLM type (e.g., 'GPT', 'HF', 'Ollama')
    #     # )
    #     process_files(
    #         text_output_directory, outputDirectory, key_file_path, llm_type
    #     )
    #     st.write(
    #         "Text files have been processed to get embeddings and tokens."
    #     )

# Text input for question
# question = st.text_area("Provide API key:")
api = st.text_input("Provide API key:", type="password")

# Text input for prompt
prompt = st.text_area("Enter the prompt for the LLM model:")

# Button to submit the question and prompt
# if st.button("Ask"):
#     if question and prompt:
#         st.write("Question:")
#         st.write(question)
#         st.write("Prompt:")
#         st.write(prompt)

#         # Query the LLM
#         # llm_type = (
#         #     "Ollama"  # Specify the LLM type (e.g., 'GPT', 'HF', 'Ollama')
#         # )

#         response = query_llm(question, prompt, key_file_path, llm_type)

#         st.write("Response:")
#         st.write(response)
#     else:
#         st.write("Please enter both a question and a prompt.")
# # else:
# # st.write("Upload PDF files and enter a question and a prompt to proceed.")

# # Button to delete embedding files
# if st.button("Delete embedding files"):
#     npy_files_deleted = 0
#     embedding_directory = "./"
#     for root, dirs, files in os.walk(embedding_directory):
#         for file in files:
#             if file.endswith(".npy"):
#                 os.remove(os.path.join(root, file))
#                 npy_files_deleted += 1
#     st.success(f"Deleted {npy_files_deleted} embedding files.")

# # Button to delete output folder
# if st.button("Delete output folder"):
#     if os.path.exists(outputDirectory):
#         shutil.rmtree(outputDirectory)
#         st.success("Output folder deleted successfully.")
#     else:
#         st.warning("Output folder does not exist.")

# fig_path = "Output_TEM/ZnO_Output/page3_fig1.jpeg"
# fig_path = "Output_TEM/ZnO_Output/"
# fig_path_files = glob.glob(os.path.join(image_output_directory, "*jpeg"))
figure_data['figure_path_files'] = figure_data['image_file'].apply(lambda x: os.path.join(image_output_directory, x))
fig_path_files = figure_data['figure_path_files'].tolist()

## single image display
# Display an image from a URL
# st.image(fig_path, caption="Figure 4. TEM image (a), EFTEM image (b) with magniﬁed inset, and HRTEM image (c) for thermally annealed yolk-shell intermediate nano- particles. (d) FFT image of Figure 4c overlaid in Figure 2c (red: ZnO, green: ZnS, blue: the thermally annealed intermediate nanoparticle; see Supporting Information for original images). J. AM. CHEM. SOC. 9 VOL. 131, NO. 39, 2009 13945 C O M M U N I C A T I O N S", use_container_width=True)


## multiple image display with slider
# List of images and captions
# images = ["image1.jpg", "image2.jpg", "image3.jpg"]  # Replace with your image paths
images = fig_path_files  # Replace with your image paths
# captions = ["Caption for Image 1", "Caption for Image 2", "Caption for Image 3"]
# df_caption = pd.read_csv("Output_TEM/ZnO_Output/figures_and_captions.csv")
# captions = df_caption["caption"].tolist()
captions = figure_data['caption'].tolist()

# # Use select_slider for a more descriptive slider
# selected_caption = st.select_slider("Choose an image", options=captions)

# # Find the index of the selected caption
# selected_index = captions.index(selected_caption)

# # Display the corresponding image
# st.image(images[selected_index], caption=selected_caption, use_container_width=True)

# Create a slider to select the image index
selected_index = st.slider("Select an image", 0, len(images) - 1, 0)

# Display the selected image with its caption
st.image(images[selected_index], caption=captions[selected_index], use_container_width=True)

# Add download button
st.download_button(
label="Download CIF file",
data=fig_path,
file_name="sample_data.cif",
# mime="text/csv"
)
