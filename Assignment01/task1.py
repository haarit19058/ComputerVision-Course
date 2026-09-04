# %%
# Imports
# ================================================================
# Don't update these imports. 
import os
import cv2
import glob
import getpass
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()
# %%
# Helper Functions
# ================================================================
# This are helper functions that you can use in your implementation.
# Don't update these functions

os.makedirs("plots", exist_ok=True)

def load_image_as_grayscale(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image

def load_image_as_rgb(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image

# ================================================================
# %%
# Task 1
# ======


def svd_decomposition(M: np.ndarray) -> tuple:
    """Perform SVD decomposition on an image
    
            Equivalent to np.linalg.svd(M, full_matrices=False), computed from the
            eigendecomposition of the smaller Gram matrix.
    
            Parameters
            ----------
            M : np.ndarray
                Input matrix of shape (H, W)
    
            Returns
            -------
            tuple
                U (H, k), sigma (k,), Vt (k, W) with k = min(H, W)
            """
    H, W = M.shape
    
    # Compute using the smaller Gram matrix to save operations
    if H >= W:
        # M^T M is (W, W)
        eigenvalues, V = np.linalg.eigh(M.T @ M)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        V = V[:, idx]
        
        sigma = np.sqrt(np.maximum(eigenvalues, 0))
        
        # U = M V Sigma^-1
        # Avoid division by zero for small singular values
        sigma_inv = np.divide(1.0, sigma, out=np.zeros_like(sigma), where=sigma > 1e-10)
        U = M @ V * sigma_inv
        Vt = V.T
    else:
        # M M^T is (H, H)
        eigenvalues, U = np.linalg.eigh(M @ M.T)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        U = U[:, idx]
        
        sigma = np.sqrt(np.maximum(eigenvalues, 0))
        
        # Vt = Sigma^-1 U^T M
        sigma_inv = np.divide(1.0, sigma, out=np.zeros_like(sigma), where=sigma > 1e-10)
        Vt = (sigma_inv[:, np.newaxis] * U.T) @ M
        
    return U, sigma, Vt

def add_watermark_single_channel(cover_image: np.ndarray, watermark: np.ndarray, alpha: float) -> np.ndarray:
    """Add watermark to a given channel of an image"""
    
    U_c, S_c, Vt_c = svd_decomposition(cover_image)
    _, S_w, _ = svd_decomposition(watermark)
    
    # Embed the watermark's singular values scaled by alpha
    S_watermarked = S_c + (alpha * S_w)
    
    # Rebuild using the cover's spatial structure
    # Broadcasting (U_c * S_watermarked) multiplies each column by the corresponding singular value
    watermarked_image = (U_c * S_watermarked) @ Vt_c
    
    return np.clip(watermarked_image, 0.0, 1.0)


def recover_watermark(original_image: np.ndarray, watermarked_image: np.ndarray, watermark: np.ndarray, alpha: float) -> np.ndarray:
    """Recover watermark from a given channel of an image"""
    
    _, S_orig, _ = svd_decomposition(original_image)
    _, S_watermarked, _ = svd_decomposition(watermarked_image)
    
    # We need the watermark's original structure (U, V^T) as our private key
    U_w, _, Vt_w = svd_decomposition(watermark)
    
    # Extract the hidden singular values
    S_recovered = (S_watermarked - S_orig) / alpha
    
    # Reconstruct the QR code
    recovered_watermark = (U_w * S_recovered) @ Vt_w
    
    return np.clip(recovered_watermark, 0.0, 1.0)


def add_watermark_rgb(cover_image: np.ndarray, watermark: np.ndarray, alpha: float) -> np.ndarray:
    """Add watermark to a given RGB image"""

    watermarked_image = np.zeros_like(cover_image)
    for channel in range(3):
        watermarked_image[..., channel] = add_watermark_single_channel(
            cover_image[..., channel], 
            watermark[..., channel], 
            alpha
        )
    return watermarked_image

def recover_watermark_rgb(original_image: np.ndarray, watermarked_image: np.ndarray, watermark: np.ndarray, alpha: float) -> np.ndarray:
    """Recover watermark from a given RGB image"""
    
    recovered_watermark = np.zeros_like(watermark)
    for channel in range(3):
        recovered_watermark[..., channel] = recover_watermark(
            original_image[..., channel], 
            watermarked_image[..., channel], 
            watermark[..., channel], 
            alpha
        )
    return recovered_watermark


if __name__ == "__main__":

    image_paths = sorted(glob.glob("imgs/*.jpg"))
    images_rgb = [load_image_as_rgb(path) for path in image_paths]
    watermark_rgb = load_image_as_rgb(os.getenv("watermark_path"))
    
    # NOTE: All 3 provided cover images in imgs/ are (321, 481) i.e. same size,
    # so k = min(H, W) = 321 for every image. As described in questions.md, the
    # watermark is resized to (k, k) once here in a hard-coded manner rather than
    # dynamically per image.     
    watermark_rgb = cv2.resize(watermark_rgb, (321, 321))

    watermarked_images_rgb = []
    recovered_watermarks_rgb = []
    
    # Set a reasonable alpha for the strength of the watermark
    alpha_value = 0.1 

    for image in images_rgb:
        # Apply watermarking to the image and recover the watermark from the watermarked image
        watermarked_image_rgb = add_watermark_rgb(image, watermark_rgb, alpha=alpha_value)
        recovered_rgb = recover_watermark_rgb(image, watermarked_image_rgb, watermark_rgb, alpha=alpha_value)

        watermarked_images_rgb.append(watermarked_image_rgb)
        recovered_watermarks_rgb.append(recovered_rgb)

    # 3x3 plots; first row: original images, second row: watermarked images, third row: recovered watermarks
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        axes[0, i].imshow(images_rgb[i])
        axes[0, i].set_title("Original Image")
        axes[0, i].axis("off")

        axes[1, i].imshow(watermarked_images_rgb[i])
        axes[1, i].set_title("Watermarked Image")
        axes[1, i].axis("off")

        axes[2, i].imshow(recovered_watermarks_rgb[i])
        axes[2, i].set_title("Recovered Watermark")
        axes[2, i].axis("off")
    plt.tight_layout()
    plt.savefig("plots/task1_results.png", metadata={"Author": getpass.getuser()})