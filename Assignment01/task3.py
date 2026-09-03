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
from task2 import *
from dotenv import load_dotenv
load_dotenv()

# %%
# Task 3
# ======

def add_gaussian_noise(image: np.ndarray, stddev: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to an image
        Parameters
        ----------
        image : np.ndarray
            Input image
        stddev : float
            Standard deviation of the Gaussian noise
    
        Returns
        -------
        np.ndarray
            Noisy image
        """

    noisy_image = np.zeros_like(image) # comment this line and write your code for the function
    return noisy_image

STDDEV = [0.01, 0.05, 0.1, 0.15]  # Standard deviations for Gaussian noise
ALPHAS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

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

    for stddev in STDDEV:
        watermarked_images_per_stdev = []
        recovered_watermarks_per_stdev = []
        for alpha in ALPHAS:
            print(f"Processing stddev={stddev}, alpha={alpha}")
            watermarked_images_per_alpha = []
            recovered_watermarks_per_alpha = []
            for image in images_rgb:
                # Apply watermarking, add Gaussian noise, and recover the watermark from the noisy image
                # ###############################################
                # Comment below line and write your code here
                pass
                
                # ###############################################

            watermarked_images_per_stdev.append(watermarked_images_per_alpha)
            recovered_watermarks_per_stdev.append(recovered_watermarks_per_alpha)

        watermarked_images_rgb.append(watermarked_images_per_stdev)
        recovered_watermarks_rgb.append(recovered_watermarks_per_stdev)


    # 5x6 plots; for each stddev in STDDEV
    # each row corresponds to a different alpha value, show the watermarked noisy image and the recovered watermark side by side for each of the 3 images
    
    ##############################
    # Increase the number of alpha values to display in the plots if recovered QR code is not can't be decoded by QR scanner.
    NUMBER_OF_ALPHA_TO_DISPLAY = 5
    ##############################  

    for stddev_idx, stddev in enumerate(STDDEV):
        fig, axes = plt.subplots(len(ALPHAS[:NUMBER_OF_ALPHA_TO_DISPLAY]), 6, figsize=(18, 3 * len(ALPHAS[:NUMBER_OF_ALPHA_TO_DISPLAY])))
        for alpha_idx, alpha in enumerate(ALPHAS[:NUMBER_OF_ALPHA_TO_DISPLAY]):
            for img_idx in range(3):
                axes[alpha_idx, img_idx * 2].imshow(watermarked_images_rgb[stddev_idx][alpha_idx][img_idx])
                axes[alpha_idx, img_idx * 2].set_title(f"Watermarked (α={alpha})")
                axes[alpha_idx, img_idx * 2].axis("off")

                axes[alpha_idx, img_idx * 2 + 1].imshow(recovered_watermarks_rgb[stddev_idx][alpha_idx][img_idx])
                axes[alpha_idx, img_idx * 2 + 1].set_title(f"Recovered (α={alpha})")
                axes[alpha_idx, img_idx * 2 + 1].axis("off")
        # set title for whole figure
        fig.suptitle(f"Watermarked and Recovered Images with Gaussian Noise (stddev={stddev})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle

        # plt.tight_layout()
        plt.savefig(f"plots/task3_watermark_results_stddev_{stddev}.png", metadata={"Author": getpass.getuser()})
    

    

    noisy_watermarked_psnr = []
    recovered_watermark_psnr = []

    for stddev_idx, stddev in enumerate(STDDEV):
        noisy_watermarked_psnr_per_stddev = []
        recovered_watermark_psnr_per_stddev = []
        for image_idx, image in enumerate(images_rgb):
            noisy_watermarked_psnr_per_image = []
            recovered_watermark_psnr_per_image = []
            for alpha_idx, alpha in enumerate(ALPHAS):
                # Compute PSNR between (a) original and watermarked noisy images; (b) original watermark and recovered watermark from the noisy image
                # ###############################################
                # Comment below line and write your code here
                pass
                
                # ###############################################

            noisy_watermarked_psnr_per_stddev.append(noisy_watermarked_psnr_per_image)
            recovered_watermark_psnr_per_stddev.append(recovered_watermark_psnr_per_image)

        noisy_watermarked_psnr.append(noisy_watermarked_psnr_per_stddev)
        recovered_watermark_psnr.append(recovered_watermark_psnr_per_stddev)


    # 2x3 plots; the column represents the images;
    # 1st row plots psnr vs alpha for the noisy watermarked images; the legend specifies the stddev values
    # 2nd row plots psnr vs alpha for the recovered watermarks; the legend specifies the stddev values

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for image_idx in range(3):
        for stddev_idx, stddev in enumerate(STDDEV):
            axes[0, image_idx].plot(ALPHAS, noisy_watermarked_psnr[stddev_idx][image_idx], label=f"stddev={stddev}")
            axes[1, image_idx].plot(ALPHAS, recovered_watermark_psnr[stddev_idx][image_idx], label=f"stddev={stddev}")

        axes[0, image_idx].set_title(f"Noisy Watermarked PSNR vs Alpha (Image {image_idx + 1})")
        axes[0, image_idx].set_xlabel("Alpha")
        axes[0, image_idx].set_ylabel("PSNR (dB)")
        axes[0, image_idx].legend()
        axes[0, image_idx].grid()

        axes[1, image_idx].set_title(f"Recovered Watermark PSNR vs Alpha (Image {image_idx + 1})")
        axes[1, image_idx].set_xlabel("Alpha")
        axes[1, image_idx].set_ylabel("PSNR (dB)")
        axes[1, image_idx].legend()
        axes[1, image_idx].grid()

    plt.tight_layout()
    plt.savefig("plots/task3_psnr_vs_alpha.png", metadata={"Author": getpass.getuser()})



# %%
