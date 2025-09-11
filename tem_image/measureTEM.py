import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

# --- Global variables for mouse callback ---
ref_points = []
drawing = False

def draw_line_callback(event, x, y, flags, param):
    """Mouse callback function to draw a line on the image."""
    global ref_points, drawing
    image = param['image']

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Create a copy to draw the temporary line
            temp_image = image.copy()
            cv2.line(temp_image, ref_points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Draw Scale Bar", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_points.append((x, y))
        drawing = False
        # Draw the final line on the original image clone
        cv2.line(image, ref_points[0], ref_points[1], (0, 255, 0), 2)
        cv2.imshow("Draw Scale Bar", image)

def get_scale_from_user(image):
    """
    Displays the image and lets the user draw a line on the scale bar
    to calculate the pixel-to-nm ratio.
    """
    global ref_points
    # Reset ref_points at the beginning of the function
    ref_points = []
    clone = image.copy()
    window_name = "Draw Scale Bar"
    cv2.namedWindow(window_name)
    
    # Pass the image clone as a parameter to the callback
    param_dict = {'image': clone}
    cv2.setMouseCallback(window_name, draw_line_callback, param_dict)

    print("INSTRUCTIONS:")
    print("1. A window will appear with your image.")
    print("2. Click and drag your mouse to draw a line along the scale bar.")
    print("3. Once the line is drawn, press 'Enter' to confirm.")
    print("4. Press 'r' to reset and draw again if you make a mistake.")

    while True:
        cv2.imshow(window_name, clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):  # Reset if the user makes a mistake
            clone = image.copy()
            param_dict['image'] = clone # update image in dict
            ref_points = []
            print("Reset. Please draw the line again.")
            
        elif key == 13:  # 13 is the Enter key
            if len(ref_points) == 2:
                break
            else:
                print("Please draw a line before pressing Enter.")

    cv2.destroyAllWindows()

    if len(ref_points) == 2:
        # Calculate the pixel length of the drawn line
        pixel_length = np.sqrt((ref_points[1][0] - ref_points[0][0])**2 + 
                               (ref_points[1][1] - ref_points[0][1])**2)
        
        # Use a robust command-line prompt
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

def _estimate_atom_properties(img_processed):
    """
    Analyzes the image to automatically estimate the average area of atoms
    using adaptive thresholding.
    """
    # Increase the block size to better handle blurry/fuzzy spots
    binary_img = cv2.adaptiveThreshold(img_processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 2)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: return None
    areas = [cv2.contourArea(c) for c in contours]
    plausible_areas = [area for area in areas if 5 < area < 500]

    if not plausible_areas: return None
    return np.median(plausible_areas)

def measure_atomic_spacing_realspace(img, pixel_to_nm_ratio):
    """
    Measures atomic lattice vectors from a TEM image using a universal method
    that adapts to different lattice types.
    NOTE: Requires scikit-learn (`pip install scikit-learn`)
    """
    # 1. Preprocess the Image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)
    
    img_blurred = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
    
    # 2. Automatically estimate atom area
    estimated_area = _estimate_atom_properties(img_blurred)
    if estimated_area is None:
        print("Could not automatically estimate atom size. Please check image quality.")
        return
    
    print(f"Automatically estimated median atom area: {estimated_area:.2f} pixels^2")

    # 3. Set up Blob Detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = estimated_area * 0.4
    params.maxArea = estimated_area * 1.6
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByConvexity = True
    params.minConvexity = 0.75
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    detector = cv2.SimpleBlobDetector_create(params)

    # 4. Detect Atoms
    keypoints = detector.detect(img_blurred)
    if not keypoints:
        print("No atoms detected. The automatic estimation might have failed.")
        return
    print(f"Detected {len(keypoints)} atoms.")
    coords = np.array([kp.pt for kp in keypoints])

    # --- UNIVERSAL LATTICE VECTOR FINDER ---
    if len(coords) < 10:
        print("Not enough atoms detected for lattice vector analysis.")
        return

    # 5. Find nearest-neighbor vectors for each atom
    tree = KDTree(coords)
    # Find up to 7 neighbors for robustness
    distances, indices = tree.query(coords, k=min(7, len(coords)))
    
    neighbor_vectors = []
    for i, point_indices in enumerate(indices):
        for neighbor_index in point_indices[1:]:
            vector = coords[neighbor_index] - coords[i]
            neighbor_vectors.append(vector)
    
    neighbor_vectors = np.array(neighbor_vectors)

    # 6. Use DBSCAN to find clusters of vectors without specifying the number of clusters
    # eps: The search radius for neighbors. Related to atom spacing.
    # min_samples: How many vectors must be in a cluster to be considered valid.
    eps = np.sqrt(estimated_area) * 0.5 
    min_samples = max(5, int(len(coords) * 0.5)) # Require at least 5 atoms or 50% to confirm a direction

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(neighbor_vectors)
    labels = db.labels_
    
    unique_labels = set(labels)
    # Remove the noise label (-1)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    # 7. Calculate the mean vector for each discovered cluster
    cluster_centers = []
    for k in unique_labels:
        class_members = labels == k
        mean_vector = np.mean(neighbor_vectors[class_members], axis=0)
        cluster_centers.append(mean_vector)

    if len(cluster_centers) < 2:
        print(f"Could not find at least 2 primary lattice directions. Found {len(cluster_centers)}.")
        return
        
    print(f"Discovered {len(cluster_centers)} primary vector directions.")
    
    # 8. Select the two basis vectors from the cluster centers
    # Sort the vectors by length
    cluster_centers = sorted(cluster_centers, key=np.linalg.norm)
    vec_a = cluster_centers[0]
    vec_b = None

    # Find the next shortest vector that is not parallel to vec_a
    for i in range(1, len(cluster_centers)):
        cos_sim = np.dot(vec_a, cluster_centers[i]) / (np.linalg.norm(vec_a) * np.linalg.norm(cluster_centers[i]))
        # if angle is > 18 degrees (cos_sim < 0.95), it's a distinct vector
        if abs(cos_sim) < 0.95: 
            vec_b = cluster_centers[i]
            break
    
    if vec_b is None:
        print("Could not determine two distinct lattice vectors. The structure may be 1D.")
        return
        
    # 9. Calculate lengths and angle
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

    # 10. Visualization
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    center_atom_idx = np.argmin(np.linalg.norm(coords - np.mean(coords, axis=0), axis=1))
    start_point = tuple(coords[center_atom_idx].astype(int))
    end_point_a = tuple((coords[center_atom_idx] + vec_a).astype(int))
    end_point_b = tuple((coords[center_atom_idx] + vec_b).astype(int))
    cv2.arrowedLine(img_with_keypoints, start_point, end_point_a, (0, 255, 0), 2)
    cv2.arrowedLine(img_with_keypoints, start_point, end_point_b, (0, 255, 255), 2)


    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image ROI')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_blurred, cmap='gray')
    plt.title('Processed (Blurred) Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_with_keypoints)
    plt.title(f'Lattice Vectors Detected')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # You must provide the path to your own image.
    image_path = 'OutputImagesLee/Page2_Figure_1_a.png' 

    try:
        # Load the image once
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            raise FileNotFoundError

        # Step 1: Get scale from user interactively
        pixel_to_nm = get_scale_from_user(original_img)

        # Step 2: If scale is valid, ask for ROI and then run the analysis
        if pixel_to_nm:
            print(f"\nCalculated pixel-to-nm ratio: {pixel_to_nm:.6f}")
            
            print("\nINSTRUCTIONS FOR ROI SELECTION:")
            print("1. A window will appear with your image.")
            print("2. Click and drag to draw a rectangle around the ATOMS you want to analyze.")
            print("3. Press 'Enter' or 'Space' to confirm the ROI.")
            
            roi = cv2.selectROI("Select Analysis Region (ROI)", original_img, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Analysis Region (ROI)")
            
            img_roi = original_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

            if img_roi.size == 0:
                 print("\nNo ROI selected. Exiting.")
            else:
                measure_atomic_spacing_realspace(img_roi, pixel_to_nm)
        else:
            print("\nScale measurement cancelled or failed. Exiting.")

    except NameError:
        print("\nINFO: Please set the 'image_path' variable to run the analysis.")
    except FileNotFoundError:
        print(f"\nERROR: The file '{image_path}' was not found. Please provide a valid path.")
    except ImportError:
        print("\nERROR: This script requires scikit-learn. Please install it using 'pip install scikit-learn'")

