import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Test image loading and display without Streamlit
def test_image_loading():
    print("=== Debugging Image Display Issues ===")
    
    # Get first available image
    image_dir = "output/image_files"
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} does not exist")
        return
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    if not image_files:
        print(f"Error: No PNG files found in {image_dir}")
        return
    
    test_image_path = os.path.join(image_dir, image_files[0])
    print(f"Testing with: {test_image_path}")
    
    # Test 1: Load with OpenCV
    print("\n1. Loading with OpenCV...")
    img_bgr = cv2.imread(test_image_path)
    img_gray = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    
    if img_bgr is None:
        print("Error: cv2.imread failed to load image")
        return
    
    print(f"   BGR shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")
    print(f"   BGR min/max: {img_bgr.min()}/{img_bgr.max()}")
    print(f"   Gray shape: {img_gray.shape}, dtype: {img_gray.dtype}")
    print(f"   Gray min/max: {img_gray.min()}/{img_gray.max()}")
    
    # Test 2: Load with PIL
    print("\n2. Loading with PIL...")
    pil_img = Image.open(test_image_path)
    print(f"   PIL mode: {pil_img.mode}, size: {pil_img.size}")
    pil_array = np.array(pil_img)
    print(f"   PIL array shape: {pil_array.shape}, dtype: {pil_array.dtype}")
    print(f"   PIL array min/max: {pil_array.min()}/{pil_array.max()}")
    
    # Test 3: Convert grayscale to RGB manually
    print("\n3. Converting grayscale to RGB...")
    if img_gray is not None:
        # Method 1: Using numpy stack
        img_rgb_stack = np.stack([img_gray, img_gray, img_gray], axis=2)
        print(f"   Stack method - shape: {img_rgb_stack.shape}, min/max: {img_rgb_stack.min()}/{img_rgb_stack.max()}")
        
        # Method 2: Using cv2.cvtColor
        img_rgb_cv2 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        print(f"   CV2 method - shape: {img_rgb_cv2.shape}, min/max: {img_rgb_cv2.min()}/{img_rgb_cv2.max()}")
        
        # Check if they're the same
        print(f"   Methods produce same result: {np.array_equal(img_rgb_stack, img_rgb_cv2)}")
    
    # Test 4: Save test images
    print("\n4. Saving test images...")
    try:
        # Save original grayscale
        cv2.imwrite("test_gray.png", img_gray)
        print("   Saved test_gray.png")
        
        # Save RGB conversion using stack method
        img_rgb_bgr = cv2.cvtColor(img_rgb_stack, cv2.COLOR_RGB2BGR)
        cv2.imwrite("test_rgb_stack.png", img_rgb_bgr)
        print("   Saved test_rgb_stack.png")
        
        # Save RGB conversion using cv2 method
        img_rgb_bgr2 = cv2.cvtColor(img_rgb_cv2, cv2.COLOR_RGB2BGR)
        cv2.imwrite("test_rgb_cv2.png", img_rgb_bgr2)
        print("   Saved test_rgb_cv2.png")
        
    except Exception as e:
        print(f"   Error saving images: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_image_loading()