import fitz  # PyMuPDF
import re
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import io

# Configuration
pdf_path = Path("3D.pdf")
output_folder = Path("Output_TEM/3D_Output")
output_folder.mkdir(parents=True, exist_ok=True)
csv_path = output_folder / "figures_and_captions.csv"

# Main function to process the PDF
def process_pdf(doc):
    """
    Processes the PDF to extract and match figures with their captions based on spatial proximity.
    Returns a list of dictionaries containing the extracted data.
    """
    figure_data = []

    # Regex to find potential figure caption headers (e.g., "Figure 1.", "Figure 2. ")
    # This pattern is more flexible and doesn't rely on specific line breaks
    caption_header_pattern = re.compile(
        r"Figure\s*(\d+)",
        re.DOTALL | re.IGNORECASE
    )

    for page_num, page in enumerate(doc, start=1):
        # Extract text blocks with their bounding boxes
        text_blocks = page.get_text("blocks")
        
        # Find all images on the page with their bounding boxes
        images_on_page = page.get_images(full=True)
        img_rects = [page.get_image_rects(img_info[0]) for img_info in images_on_page]
        
        img_details = []
        for i, (img_info, rect_list) in enumerate(zip(images_on_page, img_rects)):
            if rect_list:
                img_details.append({
                    "xref": img_info[0],
                    "rect": rect_list[0],  # Get the first bounding box
                    "page": page_num,
                    "filename_prefix": f"page{page_num}_fig{i+1}"
                })

        # Find all captions with their bounding boxes
        captions_on_page = []
        for block in text_blocks:
            block_text = block[4]
            # Find the figure number match
            match = caption_header_pattern.search(block_text)
            if match:
                # Capture the full text of the caption
                full_caption_text = block_text.strip().replace('\n', ' ')
                # A simple check to avoid short, non-caption text
                if len(full_caption_text.split()) > 5:
                    captions_on_page.append({
                        "text": full_caption_text,
                        "rect": fitz.Rect(block[:4]),
                        "fig_number": int(match.group(1))
                    })
        
        # Sort both lists by vertical position to simplify matching
        img_details.sort(key=lambda x: x['rect'].y1)
        captions_on_page.sort(key=lambda x: x['rect'].y0)

        # Match images to captions based on proximity
        for img_d in img_details:
            best_match = None
            min_distance = float('inf')
            
            for caption_d in captions_on_page:
                # Check if the caption is below the image and is not too far away
                if caption_d['rect'].y0 > img_d['rect'].y1 and caption_d['rect'].y0 - img_d['rect'].y1 < min_distance:
                    min_distance = caption_d['rect'].y0 - img_d['rect'].y1
                    best_match = caption_d

            if best_match:
                # Process and save the image if a plausible caption is found
                try:
                    base_image = doc.extract_image(img_d['xref'])
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    if image_ext.lower() == "jpx":
                        img = Image.open(io.BytesIO(image_bytes))
                        image_bytes = io.BytesIO()
                        img.save(image_bytes, format="PNG")
                        image_ext = "png"
                        image_bytes = image_bytes.getvalue()

                    image_filename = f"{img_d['filename_prefix']}.{image_ext}"
                    with open(output_folder / image_filename, "wb") as f:
                        f.write(image_bytes)

                    # Add the found data to our list, ensuring no duplicates
                    if not any(entry['image_file'] == image_filename for entry in figure_data):
                        figure_data.append({
                            "page": page_num,
                            "image_file": image_filename,
                            "caption": best_match['text']
                        })

                except Exception as e:
                    print(f"Error processing image {img_d['filename_prefix']}: {e}")
                    continue
    
    # Optional: Filter out small images that are likely not figures
    filtered_data = []
    for entry in figure_data:
        image_path = output_folder / entry['image_file']
        if not image_path.exists():
            continue
        try:
            with Image.open(image_path) as img:
                # A heuristic: check if the image is reasonably large
                if img.width > 100 and img.height > 100:
                    filtered_data.append(entry)
                else:
                    os.remove(image_path) # Clean up small, irrelevant images
                    print(f"Skipping and removing small image: {entry['image_file']}")
        except Exception as e:
            print(f"Could not open image file {image_path}: {e}")
            
    return filtered_data

# Main execution
try:
    doc = fitz.open(str(pdf_path))
    figure_data = process_pdf(doc)

    # Save mapping to CSV
    pd.DataFrame(figure_data).to_csv(csv_path, index=False)
    print(f"Successfully extracted {len(figure_data)} images and captions to {csv_path}.")

except FileNotFoundError:
    print(f"Error: The file '{pdf_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
