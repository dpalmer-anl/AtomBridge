import io
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image


def _to_pil(img):
    if img is None:
        return None
    img = np.copy(img)
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            img = (img * 255).astype(np.uint8)
        else:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = ((img - img_min) * 255 / (img_max - img_min)).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=2)
        return Image.fromarray(img_rgb)
    elif img.ndim == 3 and img.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif img.ndim == 3 and img.shape[2] == 1:
        img_2d = img[..., 0]
        img_rgb = np.stack([img_2d, img_2d, img_2d], axis=2)
        return Image.fromarray(img_rgb)
    else:
        img_2d = img[..., 0] if img.ndim > 2 else img
        img_rgb = np.stack([img_2d, img_2d, img_2d], axis=2)
        return Image.fromarray(img_rgb)


def get_scale_from_user_streamlit(img_gray, canvas_key="scale_canvas"):
    st.subheader("Measure Scale Bar")
    st.write("Draw a red line along the scale bar, then enter its length (nm) and click Confirm scale.")

    pil_img = _to_pil(img_gray)
    iw, ih = pil_img.size
    disp_w = max(1, min(900, iw))
    scale_disp_to_orig = iw / disp_w if disp_w > 0 else 1.0
    disp_h = int(round(ih / scale_disp_to_orig)) if scale_disp_to_orig > 0 else ih

    canvas = st_canvas(
        background_image=pil_img,
        width=disp_w,
        height=disp_h,
        drawing_mode="line",
        stroke_color="#ff0000",
        stroke_width=3,
        update_streamlit=True,
        key=canvas_key,
    )

    pixel_length = None
    if canvas.json_data and canvas.json_data.get("objects"):
        for obj in reversed(canvas.json_data["objects"]):
            if obj.get("type") == "line":
                x1, y1 = float(obj.get("x1", 0)), float(obj.get("y1", 0))
                x2, y2 = float(obj.get("x2", 0)), float(obj.get("y2", 0))
                disp_len = float(np.hypot(x2 - x1, y2 - y1))
                pixel_length = disp_len * scale_disp_to_orig
                st.write(f"Current line length: {pixel_length:.2f} px")
                break

    real_nm = st.number_input("Scale bar length (nm)", min_value=0.0, step=0.1, value=0.0, key=canvas_key + "_nm")
    if st.button("Confirm scale", key=canvas_key + "_confirm"):
        if pixel_length and pixel_length > 0 and real_nm > 0:
            return real_nm / pixel_length
        st.warning("Draw a line and enter a positive number.")
    return None


def custom_select_roi_streamlit(img_gray, canvas_key="roi_canvas"):
    st.subheader("Select ROI")
    st.write("Draw a green rectangle around the analysis region, then click Confirm ROI.")

    pil_img = _to_pil(img_gray)
    iw, ih = pil_img.size
    disp_w = max(1, min(900, iw))
    scale_disp_to_orig = iw / disp_w if disp_w > 0 else 1.0
    disp_h = int(round(ih / scale_disp_to_orig)) if scale_disp_to_orig > 0 else ih

    canvas = st_canvas(
        background_image=pil_img,
        width=disp_w,
        height=disp_h,
        drawing_mode="rect",
        stroke_color="#00ff00",
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        update_streamlit=True,
        key=canvas_key,
    )

    roi = None
    if canvas.json_data and canvas.json_data.get("objects"):
        for obj in reversed(canvas.json_data["objects"]):
            if obj.get("type") == "rect":
                left = float(obj.get("left", 0.0))
                top = float(obj.get("top", 0.0))
                width = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))
                height = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0))
                x = int(round(left * scale_disp_to_orig))
                y = int(round(top * scale_disp_to_orig))
                w = int(round(width * scale_disp_to_orig))
                h = int(round(height * scale_disp_to_orig))
                roi = (max(0, x), max(0, y), max(0, w), max(0, h))
                st.write(f"Current ROI (px): {roi}")
                break

    if st.button("Confirm ROI", key=canvas_key + "_confirm"):
        if roi and roi[2] > 0 and roi[3] > 0:
            return roi
        st.warning("Draw a non-empty rectangle.")
    return None


def measure_atomic_spacing_realspace(img, pixel_to_nm_ratio):
    # Lazy imports for scientific stack to avoid module import errors at app import time
    try:
        from scipy.spatial import KDTree  # type: ignore
    except Exception as e:
        st.error(f"SciPy required for analysis (KDTree not available): {e}")
        return None
    try:
        from skimage.feature import peak_local_max  # type: ignore
    except Exception as e:
        st.error(f"scikit-image required for analysis (peak_local_max not available): {e}")
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)

    min_dist = 5
    coordinates = peak_local_max(img_enhanced, min_distance=min_dist, threshold_rel=0.6, exclude_border=True)

    if len(coordinates) < 10:
        st.write(f"Detection failed. Found only {len(coordinates)} atoms. Try a larger ROI or check image contrast.")
        return None

    coords = coordinates[:, ::-1]
    st.write(f"Detected {len(coords)} atoms.")

    tree = KDTree(coords)
    distances, indices = tree.query(coords, k=min(7, len(coords)))

    neighbor_vectors = []
    all_nearest_neighbor_distances = distances[:, 1]

    for i, point_indices in enumerate(indices):
        for neighbor_index in point_indices[1:]:
            vector = coords[neighbor_index] - coords[i]
            neighbor_vectors.append(vector)

    neighbor_vectors = np.array(neighbor_vectors)

    median_nn_distance = np.median(all_nearest_neighbor_distances)
    eps = median_nn_distance * 0.4
    min_samples = max(5, int(len(coords) * 0.25))

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(neighbor_vectors)
    labels = db.labels_

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    cluster_centers = []
    for k in unique_labels:
        class_members = labels == k
        mean_vector = np.mean(neighbor_vectors[class_members], axis=0)
        cluster_centers.append(mean_vector)

    if len(cluster_centers) < 2:
        st.write(f"Could not find at least 2 primary lattice directions. Found {len(cluster_centers)}.")
        return None

    st.write(f"Discovered {len(cluster_centers)} primary vector directions.")

    cluster_centers = sorted(cluster_centers, key=np.linalg.norm)
    vec_a = cluster_centers[0]
    vec_b = None

    for i in range(1, len(cluster_centers)):
        cos_sim = np.dot(vec_a, cluster_centers[i]) / (np.linalg.norm(vec_a) * np.linalg.norm(cluster_centers[i]))
        if abs(cos_sim) < 0.95:
            vec_b = cluster_centers[i]
            break

    if vec_b is None:
        st.write("Could not determine two distinct lattice vectors. The structure may be 1D.")
        return None

    len_a_pixels = np.linalg.norm(vec_a)
    len_b_pixels = np.linalg.norm(vec_b)

    len_a_nm = len_a_pixels * pixel_to_nm_ratio
    len_b_nm = len_b_pixels * pixel_to_nm_ratio

    dot_product = np.dot(vec_a, vec_b)
    angle_rad = np.arccos(np.clip(dot_product / (len_a_pixels * len_b_pixels), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    st.write("\n--- Results ---")
    st.write(f"Lattice Vector a: {len_a_nm:.4f} nm")
    st.write(f"Lattice Vector b: {len_b_nm:.4f} nm")
    st.write(f"Angle between vectors: {angle_deg:.2f} degrees")

    img_with_detections = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y in coords:
        cv2.circle(img_with_detections, (int(x), int(y)), 3, (255, 0, 0), 1, lineType=cv2.LINE_AA)

    center_atom_idx = np.argmin(np.linalg.norm(coords - np.mean(coords, axis=0), axis=1))
    start_point = tuple(coords[center_atom_idx].astype(int))
    end_point_a = tuple((coords[center_atom_idx] + vec_a).astype(int))
    end_point_b = tuple((coords[center_atom_idx] + vec_b).astype(int))
    cv2.arrowedLine(img_with_detections, start_point, end_point_a, (0, 255, 0), 2, line_type=cv2.LINE_AA)
    cv2.arrowedLine(img_with_detections, start_point, end_point_b, (0, 255, 255), 2, line_type=cv2.LINE_AA)

    fig = plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image ROI')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_enhanced, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB))
    plt.title('Lattice Vectors Detected')
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return {"a_nm": float(len_a_nm), "b_nm": float(len_b_nm), "gamma_deg": float(angle_deg)}
