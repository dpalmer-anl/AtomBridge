from __future__ import annotations
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from PIL import ImageDraw

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
    page_text: str = ""
    is_tem: bool = False
    tem_score: int = 0


def _score_tem_relevance(text: str) -> Tuple[int, int]:
    """Return (tem_hits, non_tem_hits) based on caption/page text heuristics."""
    t = (text or "").lower()
    tem_kw = [
        "tem", "stem", "haadf", "hrtem", "electron microscopy", "micrograph",
        "lattice", "atomic resolution", "diffraction", "fft", "zone axis", "angstrom", " nm"
    ]
    non_kw = [
        "spectrum", "spectra", "cv ", "cyclic volt", "electrochem", "xrd", "raman",
        "xps", "graph", "plot", "eels spectrum"
    ]
    tem_hits = sum(1 for k in tem_kw if k in t)
    non_hits = sum(1 for k in non_kw if k in t)
    return tem_hits, non_hits


def extract_figures(pdf_path: str, out_dir: str = "_uploads/fig_extract") -> List[Figure]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    figures: List[Figure] = []
    doc = fitz.open(pdf_path)
    for pi, page in enumerate(doc):
        data = page.get_text("dict")
        page_txt = page.get_text("text") or ""
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

            fig = Figure(page_index=pi, image_path=fpath, bbox=(ix0, iy0, ix1, iy1), caption=cap.strip(), page_text=page_txt)
            tem_hits, non_hits = _score_tem_relevance(fig.caption + "\n" + fig.page_text)
            fig.is_tem = tem_hits > 0 and tem_hits >= non_hits
            fig.tem_score = tem_hits - non_hits
            figures.append(fig)
    return figures


def crop_image(image_path: str, box: Tuple[int, int, int, int], out_dir: str = "_uploads/crops") -> str:
    """Crop the image to box=(x, y, w, h) and save to out_dir; return path."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    x, y, w, h = box
    x2, y2 = max(0, x), max(0, y)
    crop = img.crop((x2, y2, x2 + max(1, w), y2 + max(1, h)))
    name = Path(image_path).stem + f"_crop_{x2}_{y2}_{w}_{h}.png"
    out_path = str(Path(out_dir) / name)
    crop.save(out_path)
    return out_path


def split_into_grid(image_path: str, rows: int, cols: int, out_dir: str = "_uploads/subfigs") -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Split the image into rows x cols tiles; return list of (path, box)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    tiles: List[Tuple[str, Tuple[int, int, int, int]]] = []
    tw, th = W // max(1, cols), H // max(1, rows)
    for r in range(rows):
        for c in range(cols):
            x, y = c * tw, r * th
            w, h = (tw if c < cols - 1 else (W - x)), (th if r < rows - 1 else (H - y))
            path = crop_image(image_path, (x, y, w, h), out_dir=out_dir)
            tiles.append((path, (x, y, w, h)))
    return tiles


def parse_subfigure_labels(caption: str) -> List[str]:
    """Extract subfigure labels like (a), (b), (c) from caption."""
    import re
    labs = re.findall(r"\(([a-zA-Z])\)", caption or "")
    return [l.lower() for l in labs]


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


def overlay_points(image_path: str, coords: List[Tuple[float, float]], out_dir: str = "_uploads/overlays") -> str:
    """Draw small red circles at coord locations on the image and save."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for (x, y) in coords:
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=2)
    out_path = str(Path(out_dir) / (Path(image_path).stem + "_points.png"))
    img.save(out_path)
    return out_path


def heatmap_overlay(image_path: str, coords: List[Tuple[float, float]], out_dir: str = "_uploads/overlays") -> Optional[str]:
    """Create a simple heatmap overlay (if cv2 available); else return points overlay."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return overlay_points(image_path, coords, out_dir=out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    base = cv2.imread(image_path)
    if base is None:
        return overlay_points(image_path, coords, out_dir=out_dir)
    h, w = base.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)
    for (x, y) in coords:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w:
            acc[yi, xi] += 1.0
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=5, sigmaY=5)
    acc = acc / (acc.max() + 1e-6)
    heat = (acc * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.6, heat_color, 0.4, 0)
    out_path = str(Path(out_dir) / (Path(image_path).stem + "_heatmap.png"))
    cv2.imwrite(out_path, overlay)
    return out_path
