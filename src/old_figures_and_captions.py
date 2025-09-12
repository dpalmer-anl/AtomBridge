import fitz  # PyMuPDF
import re
import os, sys
import pandas as pd
from pathlib import Path
from PIL import Image
import io

# Configuration
# pdf_path = Path("papers/park-et-al-2009-hetero-epitaxial-anion-exchange-yields-single-crystalline-hollow-nanoparticles.pdf")
# pdf_path = sys.argv[1]
# pdf_path = ""
# output_folder = Path("Output_TEM/ZnO_Output")
# output_folder.mkdir(parents=True, exist_ok=True)
# csv_path = output_folder / "figures_and_captions.csv"

# # Main function to process the PDF
# def process_pdf(doc, output_folder=output_folder):
#     """
#     Processes the PDF to extract and match figures with their captions.
#     Returns a list of dictionaries containing the extracted data.
#     """
#     figure_data = []

#     # Regex to find all potential figure captions
#     # It looks for "Figure X" or "Figure Y" and captures the full caption until the next figure or end of page.
#     caption_pattern = re.compile(
#         r"(Figure\s?\d+(?:[a-d])?(?:[.:]\s*.*?))(?=\n\nFigure\s?\d|Figures|References|\Z)",
#         re.DOTALL | re.IGNORECASE
#     )

#     for page_num, page in enumerate(doc, start=1):
#         text = page.get_text("text")
        
#         # Find all images on the page
#         images_on_page = page.get_images(full=True)
        
#         # Find all captions on the page
#         captions = caption_pattern.findall(text)
        
#         # Extract images and map them to captions based on proximity and naming
#         for i, img_info in enumerate(images_on_page):
#             xref = img_info[0]
#             try:
#                 # Extract the image bytes
#                 base_image = doc.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 image_ext = base_image["ext"]
                
#                 # Check for image type and convert if necessary (e.g., from JPX to PNG)
#                 if image_ext.lower() == "jpx":
#                     img = Image.open(io.BytesIO(image_bytes))
#                     image_bytes = io.BytesIO()
#                     img.save(image_bytes, format="PNG")
#                     image_ext = "png"
#                     image_bytes = image_bytes.getvalue()
                
#                 # Create a filename based on page number and figure index
#                 # This assumes figures are listed in order of appearance
#                 image_filename = f"page{page_num}_fig{i+1}.{image_ext}"
#                 with open(output_folder / image_filename, "wb") as f:
#                     f.write(image_bytes)
                
#                 # Simple heuristic to match the first found caption with the first set of images, etc.
#                 caption_text = captions[i] if i < len(captions) else "Caption not found."
                
#                 figure_data.append({
#                     "page": page_num,
#                     "image_file": image_filename,
#                     "caption": " ".join(caption_text.split())
#                 })
                
#             except Exception as e:
#                 print(f"Error processing image on page {page_num}: {e}")
#                 continue
                
#     return figure_data

# # Main execution
# try:
#     doc = fitz.open(str(pdf_path))
#     figure_data = process_pdf(doc)

#     # Save mapping to CSV
#     pd.DataFrame(figure_data).to_csv(csv_path, index=False)
#     print(f"Successfully extracted {len(figure_data)} images and captions to {csv_path}.")

# except FileNotFoundError:
#     print(f"Error: The file '{pdf_path}' was not found.")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")

# Regex to find potential figure captions
CAPTION_PATTERN = re.compile(
    r"(Figure\s?\d+(?:[a-d])?(?:[.:]\s*.*?))(?=\n\nFigure\s?\d|Figures|References|\Z)",
    re.DOTALL | re.IGNORECASE
)

def process_pdf(pdf_path, output_folder):
    """
    Process a PDF file to extract figures and their captions.
    Args:
        pdf_path (str | Path): Path to the input PDF.
        output_folder (str | Path): Directory where extracted images will be saved.
    Returns:
        pd.DataFrame: Columns = [page, image_file, caption]
    """
    pdf_path = str(pdf_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            captions = CAPTION_PATTERN.findall(text)
            images_on_page = page.get_images(full=True)

            for i, img_info in enumerate(images_on_page):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png").lower()

                    if image_ext == "jpx":
                        img = Image.open(io.BytesIO(image_bytes))
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        image_bytes = buf.getvalue()
                        image_ext = "png"

                    image_filename = f"page{page_num}_fig{i+1}.{image_ext}"
                    (output_dir / image_filename).write_bytes(image_bytes)

                    caption_text = captions[i] if i < len(captions) else "Caption not found."
                    records.append({
                        "page": page_num,
                        "image_file": image_filename,
                        "caption": " ".join(caption_text.split())
                    })
                except Exception as e:
                    print(f"Error processing image on page {page_num}: {e}")
                    continue

    return pd.DataFrame(records)

if __name__ == "__main__":
    # Optional CLI usage: python figures_and_captions.py input.pdf output_dir
    if len(sys.argv) < 3:
        print("Usage: python figures_and_captions.py <input.pdf> <output_dir>")
        sys.exit(1)
    in_pdf = sys.argv[1]
    out_dir = sys.argv[2]
    df = process_pdf(in_pdf, out_dir)
    csv_path = Path(out_dir) / "figures_and_captions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Extracted {len(df)} figures. CSV saved to {csv_path}")
