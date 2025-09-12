import fitz  # PyMuPDF
import re
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import io
import cv2
import numpy as np

# Tunable parameters
SUBFIGURE_PADDING = 10  # pixels
PLOT_WHITESPACE_THRESHOLD = 0.75  # unused but kept for compatibility

def _to_rect(b):
    # Normalize a bbox into a fitz.Rect
    if isinstance(b, fitz.Rect):
        return fitz.Rect(b)
    if isinstance(b, (tuple, list)) and len(b) == 4:
        x, y, w, h = b
        # If tuple looks like (x, y, w, h) convert to (x0, y0, x1, y1)
        if w > x and h > y:
            return fitz.Rect(x, y, w, h)
        else:
            return fitz.Rect(x, y, x + w, y + h)
    raise ValueError("Unsupported bbox format")

def merge_close_bboxes(bboxes, threshold=15):
    """
    Merge rectangles that are close to each other (both axes).
    Accepts fitz.Rect or (x,y,w,h)/(x0,y0,x1,y1).
    """
    rects_to_merge = [_to_rect(b) for b in bboxes]
    if not rects_to_merge:
        return []

    while True:
        if len(rects_to_merge) == 1:
            return rects_to_merge
        merged_one = False
        new_rects = []

        current_group = rects_to_merge[0]
        for i in range(1, len(rects_to_merge)):
            rect = rects_to_merge[i]
            inflated = current_group + (-threshold, -threshold, threshold, threshold)
            if rect.intersects(inflated):
                current_group.include_rect(rect)
                merged_one = True
            else:
                new_rects.append(current_group)
                current_group = rect
        new_rects.append(current_group)

        if not merged_one:
            return new_rects
        rects_to_merge = new_rects

def segment_figure(image_bytes):
    """
    Segment a figure image into subfigures.
    Returns a list of (x, y, w, h) in image pixel coords, sorted top-to-bottom then left-to-right.
    """
    try:
        img_cv = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            return []
        img_h, img_w = img_cv.shape

        blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_subfig_area = img_w * img_h * 0.01
        raw_bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_subfig_area]

        merged = merge_close_bboxes(raw_bboxes)

        subfigures = []
        min_area_final = img_w * img_h * 0.02
        max_area_final = img_w * img_h * 0.95

        for r in merged:
            # r may be fitz.Rect if produced by merge; convert to (x,y,w,h)
            x0, y0, x1, y1 = int(r.x0), int(r.y0), int(r.x1), int(r.y1)
            w, h = max(0, x1 - x0), max(0, y1 - y0)
            if min_area_final < (w * h) < max_area_final:
                x_pad = max(0, x0 - SUBFIGURE_PADDING)
                y_pad = max(0, y0 - SUBFIGURE_PADDING)
                w_pad = min(img_w - x_pad, w + 2 * SUBFIGURE_PADDING)
                h_pad = min(img_h - y_pad, h + 2 * SUBFIGURE_PADDING)
                subfigures.append((x_pad, y_pad, w_pad, h_pad))

        subfigures.sort(key=lambda b: (b[1], b[0]))
        return subfigures
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return []

def process_pdf(pdf_path, output_folder):
    """
    Extract figures and captions (including subpanels) from a PDF.
    Saves images under output_folder and returns a DataFrame with:
    [page, figure_title, subfigure_label, image_file, caption]
    """
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_data = []
    caption_pattern = re.compile(r"^figure\s?\d+", re.IGNORECASE)

    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            mid_x = page.rect.width / 2

            # Collect image display rects
            raw_image_bboxes = []
            for img in page.get_images(full=True) or []:
                xref = img[0]
                rects = page.get_image_rects(xref) or []
                for r in rects:
                    if r.is_valid:
                        raw_image_bboxes.append(r)

            figure_areas = merge_close_bboxes(raw_image_bboxes)

            page_elements = []
            for i, area in enumerate(figure_areas):
                if area.width > 50 and area.height > 50:
                    page_elements.append({
                        "type": "figure_area",
                        "bbox": area,
                        "id": f"fig_{page_num}_{i}"
                    })

            for block in page.get_text("blocks", flags=16) or []:
                block_text = (block[4] or "").strip()
                if not block_text:
                    continue
                if caption_pattern.match(block_text):
                    block_bbox = fitz.Rect(block[:4])
                    page_elements.append({
                        "type": "caption",
                        "bbox": block_bbox,
                        "text": " ".join(block_text.split())
                    })

            # Sort by column (left/right) then top-to-bottom
            page_elements.sort(key=lambda el: (0 if el["bbox"].x0 < mid_x else 1, el["bbox"].y0))

            processed_figure_ids = set()

            # Pass 1: Figures immediately followed by a caption in the same column
            for i, element in enumerate(page_elements):
                if element["type"] == "figure_area" and i + 1 < len(page_elements) and page_elements[i + 1]["type"] == "caption":
                    figure_area, caption_elem = element, page_elements[i + 1]
                    if (figure_area["bbox"].x0 < mid_x) != (caption_elem["bbox"].x0 < mid_x):
                        continue

                    processed_figure_ids.add(figure_area["id"])
                    try:
                        pix = page.get_pixmap(clip=figure_area["bbox"], dpi=300)
                        image_bytes = pix.tobytes("png")
                        caption_text = caption_elem["text"]
                        caption_text = caption_text.replace("\u2212", "-").replace("\u2013", "-")
                        figure_title_match = re.match(r"(Figure\s*\d+)", caption_text, re.IGNORECASE)
                        figure_title = figure_title_match.group(1) if figure_title_match else f"Figure_page{page_num}"

                        # Segment into subfigures
                        subfigure_bboxes = segment_figure(image_bytes)
                        if subfigure_bboxes and len(subfigure_bboxes) > 1:
                            img = Image.open(io.BytesIO(image_bytes))
                            for j, bbox in enumerate(subfigure_bboxes):
                                sub_label = chr(ord("a") + j)
                                cropped = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                                image_filename = f"Page{page_num}_{figure_title.replace(' ', '_')}_{sub_label}.png"
                                cropped.save(output_dir / image_filename)
                                figure_data.append({
                                    "page": page_num,
                                    "figure_title": figure_title,
                                    "subfigure_label": sub_label,
                                    "image_file": image_filename,
                                    "caption": caption_text
                                })
                        else:
                            image_filename = f"Page{page_num}_{figure_title.replace(' ', '_')}.png"
                            (output_dir / image_filename).write_bytes(image_bytes)
                            figure_data.append({
                                "page": page_num,
                                "figure_title": figure_title,
                                "subfigure_label": "N/A",
                                "image_file": image_filename,
                                "caption": caption_text
                            })
                    except Exception as e:
                        print(f"Error processing captioned figure on page {page_num}: {e}")

            # Pass 2: Orphaned figures (e.g., TOC graphics)
            for element in page_elements:
                if element["type"] == "figure_area" and element["id"] not in processed_figure_ids:
                    try:
                        pix = page.get_pixmap(clip=element["bbox"], dpi=300)
                        image_bytes = pix.tobytes("png")
                        figure_title = f"Page{page_num}_Graphical_Abstract"
                        caption_text = "Graphical Abstract"

                        subfigure_bboxes = segment_figure(image_bytes)
                        if subfigure_bboxes and len(subfigure_bboxes) > 1:
                            img = Image.open(io.BytesIO(image_bytes))
                            for j, bbox in enumerate(subfigure_bboxes):
                                sub_label = chr(ord("a") + j)
                                cropped = img.crop((bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]))
                                image_filename = f"{figure_title}_{sub_label}.png"
                                cropped.save(output_dir / image_filename)
                                figure_data.append({
                                    "page": page_num,
                                    "figure_title": figure_title,
                                    "subfigure_label": sub_label,
                                    "image_file": image_filename,
                                    "caption": caption_text
                                })
                        else:
                            image_filename = f"{figure_title}.png"
                            (output_dir / image_filename).write_bytes(image_bytes)
                            figure_data.append({
                                "page": page_num,
                                "figure_title": figure_title,
                                "subfigure_label": "N/A",
                                "image_file": image_filename,
                                "caption": caption_text
                            })
                    except Exception as e:
                        print(f"Error processing orphaned figure on page {page_num}: {e}")

    # Return as DataFrame
    if figure_data:
        df = pd.DataFrame(figure_data)
        df = df.sort_values(by=["page", "figure_title", "subfigure_label"], kind="stable").reset_index(drop=True)
        return df
    return pd.DataFrame(columns=["page", "figure_title", "subfigure_label", "image_file", "caption"])