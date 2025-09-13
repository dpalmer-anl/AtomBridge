import fitz  # PyMuPDF
import re
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import io
import cv2
import numpy as np

# --- TUNABLE PARAMETERS ---
SUBFIGURE_PADDING = 10 # Pixels to add around each detected subfigure.
PLOT_WHITESPACE_THRESHOLD = 0.75 # (75%) Figures with more white space than this will be treated as plots.

# --- Configuration ---
pdf_path = Path("papers/fang-et-al-2024-mechanistic-insights-into-potassium-assistant-thermal-catalytic-oxidation-of-soot-over-single.pdf")
output_folder = Path("OutputImages1D_new")
output_folder.mkdir(parents=True, exist_ok=True)
csv_path = output_folder / "figures_and_captions.csv"

# --- Helper Functions ---

def merge_close_bboxes(bboxes, threshold=15):
    """
    Merges bounding boxes that are close to each other, both horizontally and vertically.
    This is crucial for handling tight layouts like elemental map grids.
    """
    if not bboxes:
        return []
    
    rects_to_merge = [fitz.Rect(bbox) for bbox in bboxes]
    
    while True:
        merged_one = False
        new_rects = []
        
        # Start with the first rectangle as a new group
        current_group = rects_to_merge[0]
        
        # Iterate through the remaining rectangles
        for i in range(1, len(rects_to_merge)):
            rect = rects_to_merge[i]
            
            # Inflate the current group's bounding box to check for intersection
            inflated_current = current_group + (-threshold, -threshold, threshold, threshold)
            
            if rect.intersects(inflated_current):
                current_group.include_rect(rect)
                merged_one = True
            else:
                new_rects.append(current_group)
                current_group = rect
        
        new_rects.append(current_group)
        
        # If no merges occurred in a full pass, we are done
        if not merged_one:
            break
            
        rects_to_merge = new_rects
        
    return rects_to_merge


def segment_figure(image_bytes):
    """
    Intelligently segments a figure image into subfigures using a single pipeline.
    It's designed to handle both sparse plots and dense, gridded images.
    """
    try:
        img_cv = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        img_h, img_w = img_cv.shape

        # Adaptive thresholding to handle lighting variations
        blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
        thresh_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 25, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and get bounding boxes
        min_subfig_area = img_w * img_h * 0.01
        raw_bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_subfig_area]
        
        # Merge close bounding boxes to group related elements
        merged_bboxes = merge_close_bboxes(raw_bboxes)

        # Final check and filtering
        subfigures = []
        min_area_final = img_w * img_h * 0.02
        max_subfig_area = img_w * img_h * 0.95
        
        for bbox in merged_bboxes:
            x, y, w, h = bbox
            if min_area_final < w * h < max_subfig_area:
                # Add padding
                x_pad = max(0, x - SUBFIGURE_PADDING)
                y_pad = max(0, y - SUBFIGURE_PADDING)
                w_pad = min(img_w - x_pad, w + (2 * SUBFIGURE_PADDING))
                h_pad = min(img_h - y_pad, h + (2 * SUBFIGURE_PADDING))
                
                subfigures.append((x_pad, y_pad, w_pad, h_pad))

        # Sort subfigures from top-to-bottom and left-to-right
        subfigures.sort(key=lambda b: (b[1], b[0]))
        return subfigures
        
    except Exception as e:
        print(f"Error during contour segmentation: {e}")
        return []

# --- Main PDF Processing Function ---
def process_pdf(doc):
    """Main PDF processing loop."""
    figure_data = []
    caption_pattern = re.compile(r"^figure\s?\d+", re.IGNORECASE)
    for page_num, page in enumerate(doc, start=1):
        mid_x = page.rect.width / 2
        page_elements = []
        
        raw_image_bboxes = [page.get_image_bbox(img) for img in page.get_images(full=True) if page.get_image_bbox(img).is_valid]
        figure_areas = merge_close_bboxes(raw_image_bboxes)
        
        for i, area in enumerate(figure_areas):
            if area.width > 50 and area.height > 50:
                page_elements.append({'type': 'figure_area', 'bbox': area, 'id': f'fig_{page_num}_{i}'})

        for block in page.get_text("blocks", flags=16):
            block_text = block[4].strip()
            if caption_pattern.match(block_text):
                block_bbox = fitz.Rect(block[:4])
                page_elements.append({'type': 'caption', 'bbox': block_bbox, 'text': " ".join(block_text.split())})

        page_elements.sort(key=lambda el: (0 if el['bbox'].x0 < mid_x else 1, el['bbox'].y0))
        
        processed_figure_ids = set()

        # First pass: Process all figures with matching captions
        for i, element in enumerate(page_elements):
            if element['type'] == 'figure_area' and i + 1 < len(page_elements) and page_elements[i+1]['type'] == 'caption':
                figure_area, caption_elem = element, page_elements[i+1]
                if (figure_area['bbox'].x0 < mid_x) != (caption_elem['bbox'].x0 < mid_x): continue

                processed_figure_ids.add(figure_area['id'])
                try:
                    pix = page.get_pixmap(clip=figure_area['bbox'], dpi=300)
                    image_bytes = pix.tobytes("png")
                    caption_text = caption_elem['text']
                    caption_text = caption_text.replace('\u2212', '-').replace('\u2013', '-')
                    figure_title_match = re.match(r"(Figure\s*\d+)", caption_text, re.IGNORECASE)
                    figure_title = figure_title_match.group(1) if figure_title_match else "Figure"
                    subfigure_bboxes = segment_figure(image_bytes)
                    if subfigure_bboxes and len(subfigure_bboxes) > 1:
                        img = Image.open(io.BytesIO(image_bytes))
                        for j, bbox in enumerate(subfigure_bboxes):
                            sub_label = chr(ord('a') + j)
                            cropped_img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                            image_filename = f"Page{page_num}_{figure_title.replace(' ', '_')}_{sub_label}.png"
                            cropped_img.save(output_folder / image_filename)
                            figure_data.append({"page": page_num, "figure_title": figure_title, "subfigure_label": sub_label, "image_file": image_filename, "caption": caption_text})
                    else:
                        image_filename = f"Page{page_num}_{figure_title.replace(' ', '_')}.png"
                        with open(output_folder / image_filename, "wb") as f: f.write(image_bytes)
                        figure_data.append({"page": page_num, "figure_title": figure_title, "subfigure_label": "N/A", "image_file": image_filename, "caption": caption_text})
                except Exception as e:
                    print(f"Error processing captioned figure on page {page_num}: {e}")

        # Second pass: Process any "orphaned" figures like TOC images
        for element in page_elements:
            if element['type'] == 'figure_area' and element['id'] not in processed_figure_ids:
                figure_area = element
                print(f"INFO: Processing orphaned image (likely TOC graphic) on page {page_num}...")
                try:
                    pix = page.get_pixmap(clip=figure_area['bbox'], dpi=300)
                    image_bytes = pix.tobytes("png")
                    figure_title = f"Page{page_num}_Graphical_Abstract"
                    caption_text = "Graphical Abstract"
                    
                    # Also segment TOC graphics in case they have parts
                    subfigure_bboxes = segment_figure(image_bytes)
                    if subfigure_bboxes and len(subfigure_bboxes) > 1:
                        img = Image.open(io.BytesIO(image_bytes))
                        for j, bbox in enumerate(subfigure_bboxes):
                            sub_label = chr(ord('a') + j)
                            cropped_img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                            image_filename = f"{figure_title}_{sub_label}.png"
                            cropped_img.save(output_folder / image_filename)
                            figure_data.append({"page": page_num, "figure_title": figure_title, "subfigure_label": sub_label, "image_file": image_filename, "caption": caption_text})
                    else:
                        image_filename = f"{figure_title}.png"
                        with open(output_folder / image_filename, "wb") as f: f.write(image_bytes)
                        figure_data.append({"page": page_num, "figure_title": figure_title, "subfigure_label": "N/A", "image_file": image_filename, "caption": caption_text})
                except Exception as e:
                    print(f"Error processing orphaned figure on page {page_num}: {e}")

    return figure_data

# --- Main Execution ---
try:
    doc = fitz.open(str(pdf_path))
    figure_data = process_pdf(doc)
    if figure_data:
        df = pd.DataFrame(figure_data)
        # Sort final output by page and then by figure title/label for consistency
        df = df.sort_values(by=['page', 'figure_title', 'subfigure_label'])
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Γ£à Successfully extracted {len(df)} figures/subfigures to {csv_path}.")
    else:
        print("ΓÜá∩╕Å No figures were extracted. Check the PDF path and content.")
except FileNotFoundError:
    print(f"Γ¥î Error: The file '{pdf_path}' was not found.")
except Exception as e:
    print(f"Γ¥î An unexpected error occurred: {e}")
