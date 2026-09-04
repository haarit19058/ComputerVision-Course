import os
import cv2
import glob
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_image_as_grayscale(image_path: str) -> np.ndarray:
    """Load image and normalize to [0, 1]"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    image = image.astype(np.float32) / 255.0  
    return image

if __name__ == "__main__":
    # ==========================================
    # 1. Process the QR Code Watermark
    # ==========================================
    watermark_path = os.getenv("watermark_path")
    
    if not watermark_path:
        print("Error: 'watermark_path' not found in .env file.")
    else:
        print(f"--- QR Code Watermark: {os.path.basename(watermark_path)} ---")
        qr_image = load_image_as_grayscale(watermark_path)
        
        # Resize to k x k (k = 321 to match the cover images)
        k = 321
        qr_resized = cv2.resize(qr_image, (k, k))
        
        _, S_qr, _ = np.linalg.svd(qr_resized, full_matrices=False)
        
        print(f"Top 20 singular values (Total: {len(S_qr)}):")
        for i in range(50):
            print(f"  {i+1:2d}: {S_qr[i]:.4f}")
        print("\n")

    # ==========================================
    # 2. Process the Cover Images in imgs/ folder
    # ==========================================
    image_paths = sorted(glob.glob("imgs/*.jpg"))
    
    if not image_paths:
        print("No images found in imgs/ directory.")
    else:
        for path in image_paths:
            print(f"--- Cover Image: {os.path.basename(path)} ---")
            cover_image = load_image_as_grayscale(path)
            
            # Perform SVD on the cover image
            # For a 321x481 image, this will return 321 singular values
            _, S_cover, _ = np.linalg.svd(cover_image, full_matrices=False)
            
            print(f"Top 20 singular values (Total: {len(S_cover)}):")
            for i in range(50):
                print(f"  {i+1:2d}: {S_cover[i]:.4f}")
            print("\n")