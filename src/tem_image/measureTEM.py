import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max
import os

# --- Global variables for mouse callback ---
ref_points = []
drawing = False
roi_rect = None
drawing_roi = False
resizing_handle = None
HANDLE_SIZE = 8

def draw_line_callback(event, x, y, flags, param):
    """Mouse callback function to draw a line on the image."""
    global ref_points, drawing
    image = param['image']

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Create a copy to draw the temporary line in real-time
            temp_image = image.copy()
            cv2.line(temp_image, ref_points[0], (x, y), (0, 0, 255), 2, lineType=cv2.LINE_AA) # Red line
            cv2.imshow("Draw Scale Bar", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_points.append((x, y))
        drawing = False
        # Draw the final line on the original image clone
        cv2.line(image, ref_points[0], ref_points[1], (0, 0, 255), 2, lineType=cv2.LINE_AA) # Red line
        cv2.imshow("Draw Scale Bar", image)

def get_scale_from_user(image):
    """
    Displays the image and lets the user draw a line on the scale bar
    to calculate the pixel-to-nm ratio.
    """
    global ref_points
    ref_points = []
    clone = image.copy()
    window_name = "Draw Scale Bar"
    cv2.namedWindow(window_name)
    
    param_dict = {'image': clone}
    cv2.setMouseCallback(window_name, draw_line_callback, param_dict)

    print("INSTRUCTIONS (SCALE BAR):")
    print("1. A window will appear with your image.")
    print("2. Click and drag your mouse to draw a red line along the scale bar.")
    print("3. Once the line is drawn, press 'Enter' to confirm.")
    print("4. Press 'r' to reset and draw again if you make a mistake.")

    while True:
        cv2.imshow(window_name, clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            clone = image.copy()
            param_dict['image'] = clone
            ref_points = []
            print("Reset. Please draw the line again.")
            
        elif key == 13:
            if len(ref_points) == 2:
                break
            else:
                print("Please draw a line before pressing Enter.")

    cv2.destroyAllWindows()

    if len(ref_points) == 2:
        pixel_length = np.sqrt((ref_points[1][0] - ref_points[0][0])**2 + 
                               (ref_points[1][1] - ref_points[0][1])**2)
        
        real_world_length = None
        while real_world_length is None:
            try:
                value = input("\nIn the terminal, please enter the scale bar length (in nm) and press Enter: ")
                real_world_length = float(value)
                if real_world_length <= 0:
                    print("Error: Please enter a positive number.")
                    real_world_length = None
            except ValueError:
                print("Error: Invalid input. Please enter a number (e.g., 5 or 2.5).")
        
        if pixel_length > 0:
            return real_world_length / pixel_length
    
    return None

def roi_interaction_callback(event, x, y, flags, param):
    """Mouse callback for drawing and resizing the ROI rectangle."""
    global roi_rect, drawing_roi, resizing_handle

    if event == cv2.EVENT_LBUTTONDOWN:
        if roi_rect is not None:
            (rx, ry, rw, rh) = roi_rect
            handles = {
                'top-left': (rx, ry), 'top-right': (rx + rw, ry),
                'bottom-left': (rx, ry + rh), 'bottom-right': (rx + rw, ry + rh)
            }
            for name, (hx, hy) in handles.items():
                if abs(x - hx) < HANDLE_SIZE and abs(y - hy) < HANDLE_SIZE:
                    resizing_handle = name
                    return
        drawing_roi = True
        roi_rect = [x, y, 0, 0]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_roi:
            roi_rect[2] = x - roi_rect[0]
            roi_rect[3] = y - roi_rect[1]
        elif resizing_handle:
            (rx, ry, rw, rh) = roi_rect
            if resizing_handle == 'top-left':
                roi_rect = [x, y, rw + (rx - x), rh + (ry - y)]
            elif resizing_handle == 'top-right':
                roi_rect = [rx, y, x - rx, rh + (ry - y)]
            elif resizing_handle == 'bottom-left':
                roi_rect = [x, ry, rw + (rx - x), y - ry]
            elif resizing_handle == 'bottom-right':
                roi_rect = [rx, ry, x - rx, y - ry]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_roi = False
        resizing_handle = None
        if roi_rect is not None:
            # Ensure width and height are positive
            roi_rect[0] = min(roi_rect[0], roi_rect[0] + roi_rect[2])
            roi_rect[1] = min(roi_rect[1], roi_rect[1] + roi_rect[3])
            roi_rect[2] = abs(roi_rect[2])
            roi_rect[3] = abs(roi_rect[3])

def custom_select_roi(image):
    """A custom ROI selector that allows resizing."""
    global roi_rect
    roi_rect = None
    clone = image.copy()
    window_name = "Select Analysis Region (ROI)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, roi_interaction_callback)

    print("\nINSTRUCTIONS (ROI SELECTION):")
    print("1. Draw a rectangle around the atoms.")
    print("2. Drag the corners to resize the box.")
    print("3. Press 'Enter' to confirm, or 'r' to reset.")

    while True:
        temp_image = clone.copy()
        if roi_rect is not None and roi_rect[2] > 0 and roi_rect[3] > 0:
            (x, y, w, h) = [int(v) for v in roi_rect]
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            handles = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
            for hx, hy in handles:
                cv2.circle(temp_image, (hx, hy), HANDLE_SIZE, (0, 0, 255), -1)
        
        cv2.imshow(window_name, temp_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            roi_rect = None
        elif key == 13: # Enter
            break
            
    cv2.destroyAllWindows()
    if roi_rect and roi_rect[2] > 0 and roi_rect[3] > 0:
        return tuple(int(v) for v in roi_rect)
    return None


def measure_atomic_spacing_realspace(img, pixel_to_nm_ratio, original_filename):
    """
    Measures atomic lattice vectors from a TEM image using a universal method
    that adapts to different lattice types. Uses peak finding for atom detection.
    """
    # 1. Preprocess the Image - CLAHE is great for enhancing local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)
    
    # 2. Find Atom Centers using Local Maxima Peak Finding
    min_dist = 5
    # --- CHANGE: Increased threshold to be more selective for bright atoms ---
    # This helps ignore dimmer atoms in complex structures like SrTiO3
    coordinates = peak_local_max(img_enhanced, min_distance=min_dist, threshold_rel=0.6, exclude_border=True)

    if len(coordinates) < 10:
        print(f"Detection failed. Found only {len(coordinates)} atoms. Try a larger ROI or check image contrast.")
        return

    coords = coordinates[:, ::-1]
    print(f"Detected {len(coords)} atoms.")

    # --- UNIVERSAL LATTICE VECTOR FINDER ---

    # 3. Find nearest-neighbor vectors for each atom
    tree = KDTree(coords)
    distances, indices = tree.query(coords, k=min(7, len(coords)))
    
    neighbor_vectors = []
    all_nearest_neighbor_distances = distances[:, 1]
    
    for i, point_indices in enumerate(indices):
        for neighbor_index in point_indices[1:]:
            vector = coords[neighbor_index] - coords[i]
            neighbor_vectors.append(vector)
    
    neighbor_vectors = np.array(neighbor_vectors)

    # 4. Use DBSCAN to find clusters of vectors, with adaptive parameters
    median_nn_distance = np.median(all_nearest_neighbor_distances)
    eps = median_nn_distance * 0.4
    min_samples = max(5, int(len(coords) * 0.25))

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(neighbor_vectors)
    labels = db.labels_
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    # 5. Calculate the mean vector for each discovered cluster
    cluster_centers = []
    for k in unique_labels:
        class_members = labels == k
        mean_vector = np.mean(neighbor_vectors[class_members], axis=0)
        cluster_centers.append(mean_vector)

    if len(cluster_centers) < 2:
        print(f"Could not find at least 2 primary lattice directions. Found {len(cluster_centers)}.")
        return
        
    print(f"Discovered {len(cluster_centers)} primary vector directions.")
    
    # 6. Select the two basis vectors from the cluster centers
    cluster_centers = sorted(cluster_centers, key=np.linalg.norm)
    vec_a = cluster_centers[0]
    vec_b = None

    for i in range(1, len(cluster_centers)):
        cos_sim = np.dot(vec_a, cluster_centers[i]) / (np.linalg.norm(vec_a) * np.linalg.norm(cluster_centers[i]))
        if abs(cos_sim) < 0.95:
            vec_b = cluster_centers[i]
            break
    
    if vec_b is None:
        print("Could not determine two distinct lattice vectors. The structure may be 1D.")
        return
        
    # 7. Calculate lengths and angle
    len_a_pixels = np.linalg.norm(vec_a)
    len_b_pixels = np.linalg.norm(vec_b)
    
    len_a_nm = len_a_pixels * pixel_to_nm_ratio
    len_b_nm = len_b_pixels * pixel_to_nm_ratio

    dot_product = np.dot(vec_a, vec_b)
    angle_rad = np.arccos(np.clip(dot_product / (len_a_pixels * len_b_pixels), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    print("\n--- Results ---")
    print(f"Lattice Vector a: {len_a_nm:.4f} nm")
    print(f"Lattice Vector b: {len_b_nm:.4f} nm")
    print(f"Angle between vectors: {angle_deg:.2f} degrees")

    # 8. Visualization
    img_with_detections = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y in coords:
        # --- Set circle color to blue (BGR format) ---
        cv2.circle(img_with_detections, (int(x), int(y)), 3, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    
    center_atom_idx = np.argmin(np.linalg.norm(coords - np.mean(coords, axis=0), axis=1))
    start_point = tuple(coords[center_atom_idx].astype(int))
    end_point_a = tuple((coords[center_atom_idx] + vec_a).astype(int))
    end_point_b = tuple((coords[center_atom_idx] + vec_b).astype(int))
    cv2.arrowedLine(img_with_detections, start_point, end_point_a, (0, 255, 0), 2, line_type=cv2.LINE_AA)
    cv2.arrowedLine(img_with_detections, start_point, end_point_b, (0, 255, 255), 2, line_type=cv2.LINE_AA)

    # --- NEW: Save the final analysis image ---
    base, ext = os.path.splitext(original_filename)
    output_filename = f"{base}_analysis.png"
    cv2.imwrite(output_filename, img_with_detections)
    print(f"\nSaved analysis image to: {output_filename}")

    # 9. Display the plot
    plt.figure(figsize=(18, 6))
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
    plt.title(f'Lattice Vectors Detected')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    image_path = 'OutputImages1D_new/Page3_Figure_2_l.png' 

    try:
        # Load the original image in grayscale for analysis
        original_img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img_gray is None:
            raise FileNotFoundError

        # Create a color version for UI operations
        original_img_bgr = cv2.cvtColor(original_img_gray, cv2.COLOR_GRAY2BGR)

        # Pass the BGR image to the UI function
        pixel_to_nm = get_scale_from_user(original_img_bgr)

        if pixel_to_nm:
            print(f"\nCalculated pixel-to-nm ratio: {pixel_to_nm:.6f}")
            
            # Pass the BGR image to the ROI selector for correct color drawing
            roi = custom_select_roi(original_img_bgr)
            
            if roi is None or roi[2] == 0 or roi[3] == 0:
                 print("\nNo ROI selected or invalid ROI. Exiting.")
            else:
                # Crop the ROI from the original GRAYSCALE image for analysis
                img_roi_gray = original_img_gray[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                # Pass the original filename to the analysis function for saving
                measure_atomic_spacing_realspace(img_roi_gray, pixel_to_nm, image_path)
        else:
            print("\nScale measurement cancelled or failed. Exiting.")

    except NameError:
        print("\nINFO: Please set the 'image_path' variable to run the analysis.")
    except FileNotFoundError:
        print(f"\nERROR: The file '{image_path}' was not found. Please provide a valid path.")
    except ImportError:
        print("\nERROR: This script requires scikit-learn and scikit-image. Please install them:\n'pip install scikit-learn scikit-image'")

