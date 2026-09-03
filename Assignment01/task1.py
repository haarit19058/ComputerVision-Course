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

    return ...,...,... # comment this line and write your code for the function

def add_watermark_single_channel(cover_image: np.ndarray, watermark: np.ndarray, alpha: float) -> np.ndarray:
    """Add watermark to a given channel of an image
    
        Parameters
        ----------
        cover_image : np.ndarray
            Image where watermark needs to be applied
        watermark : np.ndarray
            Watermark
        alpha : float
            Alpha to control the strength of watermark
    
        Returns
        -------
        np.ndarray
            Watermarked Image
        """
    
    watermarked_image = np.zeros_like(cover_image) # comment this line and write your code for the function
    return watermarked_image


def recover_watermark(original_image: np.ndarray, watermarked_image: np.ndarray, watermark: np.ndarray, alpha: float) -> np.ndarray:
    """Recover watermark from a given channel of an image
    
        Parameters
        ----------
        original_image : np.ndarray
            Original image
        watermarked_image : np.ndarray
            Watermarked image
        watermark : np.ndarray
            Watermark
        alpha : float
            Alpha used during watermarking
    
        Returns
        -------
        np.ndarray
            Recovered watermark
        """

    
    recovered_watermark = np.zeros_like(watermark) # comment this line and write your code for the function
    return recovered_watermark


def add_watermark_rgb(cover_image: np.ndarray, watermark: np.ndarray, alpha: float) -> np.ndarray:
    """Add watermark to a given RGB image
        Parameters
        ----------
        cover_image : np.ndarray
            Image where watermark needs to be applied
        watermark : np.ndarray
            Watermark
        alpha : float
            Alpha to control the strength of watermark
    
        Returns
        -------
        np.ndarray
            Watermarked Image
        """

    watermarked_image = np.zeros_like(cover_image) # comment this line and write your code for the function
    return watermarked_image

def recover_watermark_rgb(original_image: np.ndarray, watermarked_image: np.ndarray, watermark: np.ndarray, alpha: float) -> np.ndarray:
    """Recover watermark from a given RGB image
        Parameters
        ----------
        original_image : np.ndarray
            Original image
        watermarked_image : np.ndarray
            Watermarked image
        watermark : np.ndarray
            Watermark
        alpha : float
            Alpha used during watermarking
    
        Returns
        -------
        np.ndarray
            Recovered watermark
        """

    
    recovered_watermark = np.zeros_like(watermark) # comment this line and write your code for the function
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

    for image in images_rgb:
        # Apply watermarking to the image and recover the watermark from the watermarked image
        # ###############################################
        # Comment these lines and write your code here
        watermarked_image_rgb = ...
        recovered_rgb = ...
        # ###############################################

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
