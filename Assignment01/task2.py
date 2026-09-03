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


from task1 import *

from dotenv import load_dotenv
load_dotenv()
# %%
# Task 2
# ======

def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Compute PSNR between two images
        Parameters
        ----------
        original : np.ndarray
            Original image
        modified : np.ndarray
            Modified image
        Returns
        -------
        float
            PSNR value
        """

    psnr = 0.0 # comment this line and write your code for the function
    return psnr


if __name__ == "__main__":

    image_paths = sorted(glob.glob("imgs/*.jpg"))
    images_rgb = [load_image_as_rgb(path) for path in image_paths]
    watermark_rgb = load_image_as_rgb(os.getenv("watermark_path"))
    # NOTE: All 3 provided cover images in imgs/ are (321, 481) i.e. same size,
    # so k = min(H, W) = 321 for every image. As described in questions.md, the
    # watermark is resized to (k, k) once here in a hard-coded manner rather than
    # dynamically per image.     
    watermark_rgb = cv2.resize(watermark_rgb, (321, 321))

    alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    psnr_watermarked = []
    psnr_recovered = []

    for alpha in alphas:
        print("Processing alpha:", alpha)
        psnr_watermarked_alpha = []
        psnr_recovered_alpha = []
        for i in range(3):
            watermarked_image_rgb = add_watermark_rgb(images_rgb[i], watermark_rgb, alpha=alpha)
            recovered_rgb = recover_watermark_rgb(images_rgb[i], watermarked_image_rgb, watermark_rgb, alpha=alpha)

            # Compute PSNR between (a) original and watermarked images; (b)original watermark and recovered watermark
            # ###############################################
            # Write your code here
            
            # ###############################################

        psnr_watermarked.append(psnr_watermarked_alpha)
        psnr_recovered.append(psnr_recovered_alpha)

    # plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i in range(3):
        ax1.plot(alphas, [psnr_watermarked[j][i] for j in range(len(alphas))], label=f"Image {i+1}")
        ax2.plot(alphas, [psnr_recovered[j][i] for j in range(len(alphas))], label=f"Image {i+1}")  

    ax1.set_title("PSNR between Original and Watermarked Images")
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_xticks(alphas)
    ax1.legend()

    ax2.set_title("PSNR between Watermark and Recovered Watermark")
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_xticks(alphas)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("plots/task2_psnr_results.png", metadata={"Author": getpass.getuser()})


