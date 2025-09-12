from __future__ import annotations
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # Optional; handled in code


@dataclass
class Figure:
    page_index: int
    image_path: str
    bbox: Tuple[float, float, float, float]
    caption: str


def extract_figures(pdf_path: str, out_dir: str = "_uploads/fig_extract") -> List[Figure]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    figures: List[Figure] = []
    doc = fitz.open(pdf_path)
    for pi, page in enumerate(doc):
        data = page.get_text("dict")
        blocks = data.get("blocks", [])
        # Build list of text blocks for caption proximity
        text_blocks = [b for b in blocks if b.get("type") == 0]
        for bi, b in enumerate(blocks):
            if b.get("type") != 1:
                continue  # not image
            bbox = tuple(b.get("bbox"))  # x0,y0,x1,y1
            # Extract image from xref if present
            xref = b.get("image") or b.get("xref")
            if not xref:
                # Fallback: rasterize bbox region
                pix = page.get_pixmap(clip=bbox)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            else:
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                except Exception:
                    pix = page.get_pixmap(clip=bbox)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            fname = f"fig_p{pi+1}_{bi}.png"
            fpath = str(Path(out_dir) / fname)
            img.save(fpath)

            # Heuristic caption: nearest text block below the image bbox
            cap = ""
            ix0, iy0, ix1, iy1 = bbox
            best_dist = 1e9
            for tb in text_blocks:
                tbbox = tb.get("bbox")
                if not tbbox:
                    continue
                tx0, ty0, tx1, ty1 = tbbox
                # consider text that starts below image bottom and horizontally overlaps
                if ty0 >= iy1 and not (tx1 < ix0 or tx0 > ix1):
                    d = ty0 - iy1
                    if d < best_dist:
                        best_dist = d
                        cap = "".join(span.get("text", "") for line in tb.get("lines", []) for span in line.get("spans", []))

            figures.append(Figure(page_index=pi, image_path=fpath, bbox=(ix0, iy0, ix1, iy1), caption=cap.strip()))
    return figures


def hough_circles_detect(image_path: str) -> List[Tuple[float, float, float]]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not installed. Please install opencv-python.")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img_blur = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                               param1=100, param2=30, minRadius=2, maxRadius=40)
    out: List[Tuple[float, float, float]] = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            out.append((float(x), float(y), float(r)))
    return out


def run_tem_to_atom_coords(image_path: str) -> List[Tuple[float, float]]:
    """Try external circle_detection.TEMtoAtomCoordinates first; fallback to HoughCircles.
    Returns list of (x,y) coordinates in pixel units.
    """
    # Attempt to import external module if present in repo
    try:
        from circle_detection.TEMtoAtomCoordinates import process_image as tem_process  # type: ignore
        coords = tem_process(image_path)  # expected to return Nx2
        return [(float(x), float(y)) for x, y in coords]
    except Exception:
        circles = hough_circles_detect(image_path)
        return [(x, y) for (x, y, r) in circles]

