import fitz  # PyMuPDF
import re
from pathlib import Path
from PIL import Image
import pandas as pd
import io
import os

# Regex to find potential figure caption headers (e.g., "Figure 1", "Figure 2")
CAPTION_HEADER_PATTERN = re.compile(r"Figure\s*(\d+)", re.IGNORECASE)

# Heuristic size threshold to drop small, likely non-figure images
MIN_W, MIN_H = 100, 100

def process_pdf(pdf_path, output_folder):
    """
    Process a PDF to extract figures and their captions using spatial proximity.
    Only keeps images that have a plausible caption below them and are larger than MIN_W x MIN_H.
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
            # 1) Text blocks with rectangles
            text_blocks = page.get_text("blocks") or []

            # 2) Images on page with rectangles
            images_on_page = page.get_images(full=True) or []
            img_rects = [page.get_image_rects(img_info[0]) for img_info in images_on_page]

            img_details = []
            for i, (img_info, rect_list) in enumerate(zip(images_on_page, img_rects)):
                if rect_list:
                    img_details.append({
                        "xref": img_info[0],
                        "rect": rect_list[0],  # first display rect
                        "page": page_num,
                        "idx_on_page": i + 1,  # 1-based for filename
                    })

            # 3) Find caption-like blocks
            captions_on_page = []
            for block in text_blocks:
                if len(block) < 5:
                    continue
                block_text = (block[4] or "")
                match = CAPTION_HEADER_PATTERN.search(block_text)
                if match:
                    full_caption_text = block_text.strip().replace("\n", " ")
                    # avoid very short lines
                    if len(full_caption_text.split()) > 5:
                        captions_on_page.append({
                            "text": full_caption_text,
                            "rect": fitz.Rect(block[:4]),
                            "fig_number": int(match.group(1)),
                        })

            # Sort top->bottom
            img_details.sort(key=lambda x: x["rect"].y1)
            captions_on_page.sort(key=lambda x: x["rect"].y0)

            # 4) Match images to nearest caption below; skip images without a plausible caption
            for img_d in img_details:
                best_match = None
                min_distance = float("inf")
                for caption_d in captions_on_page:
                    # caption must be below the image; choose smallest vertical gap
                    if caption_d["rect"].y0 > img_d["rect"].y1:
                        gap = caption_d["rect"].y0 - img_d["rect"].y1
                        if gap < min_distance:
                            min_distance = gap
                            best_match = caption_d

                # Only keep images that have a plausible caption
                if not best_match:
                    continue

                # 5) Extract, save, and size-filter the image
                try:
                    base_image = doc.extract_image(img_d["xref"])
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png").lower()

                    # Normalize JPX to PNG for compatibility
                    if image_ext == "jpx":
                        pil = Image.open(io.BytesIO(image_bytes))
                        buf = io.BytesIO()
                        pil.save(buf, format="PNG")
                        image_bytes = buf.getvalue()
                        image_ext = "png"

                    image_filename = f"page{page_num}_fig{img_d['idx_on_page']}.{image_ext}"
                    out_path = output_dir / image_filename
                    out_path.write_bytes(image_bytes)

                    # Drop small images
                    keep = True
                    try:
                        with Image.open(out_path) as pil_img:
                            if pil_img.width <= MIN_W or pil_img.height <= MIN_H:
                                keep = False
                    except Exception:
                        keep = False

                    if keep:
                        records.append({
                            "page": page_num,
                            "image_file": image_filename,
                            "caption": best_match["text"],
                        })
                    else:
                        try:
                            os.remove(out_path)
                        except OSError:
                            pass

                except Exception as e:
                    print(f"Error processing image on page {page_num}: {e}")
                    continue

    return pd.DataFrame(records)