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
# Helper Functions
# ================================================================
# This are helper functions that you can use in your implementation.
# Don't update these functions
def jpg_compression(image: np.ndarray, quality: int = 90) -> np.ndarray:
    # Convert to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Encode image to JPEG format in memory
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image_uint8, encode_param)
    
    # Decode the JPEG image back to numpy array
    decimg = cv2.imdecode(encimg, 1)
    
    # Convert back to float32 and normalize to [0, 1]
    decimg = decimg.astype(np.float32) / 255.0
    return decimg

# ================================================================

QUALITY = [90, 70, 50, 30]  # JPEG quality levels
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

    for quality in QUALITY:
        watermarked_images_per_quality = []
        recovered_watermarks_per_quality = []
        for alpha in ALPHAS:
            print(f"Processing quality={quality}, alpha={alpha}")
            watermarked_images_per_alpha = []
            recovered_watermarks_per_alpha = []
            for image in images_rgb:
                # Apply watermarking, apply JPEG compression, and recover the watermark from the compressed image
                # ###############################################
                
                # 1. Add the watermark to the clean image
                clean_watermarked = add_watermark_rgb(image, watermark_rgb, alpha)
                
                # 2. Simulate internet re-encoding by applying JPEG compression
                compressed_watermarked = jpg_compression(clean_watermarked, quality)
                
                # 3. Attempt to extract the QR code from the compressed image
                recovered_wm = recover_watermark_rgb(image, compressed_watermarked, watermark_rgb, alpha)
                
                watermarked_images_per_alpha.append(compressed_watermarked)
                recovered_watermarks_per_alpha.append(recovered_wm)
                
                # ###############################################

            watermarked_images_per_quality.append(watermarked_images_per_alpha)
            recovered_watermarks_per_quality.append(recovered_watermarks_per_alpha)

        watermarked_images_rgb.append(watermarked_images_per_quality)
        recovered_watermarks_rgb.append(recovered_watermarks_per_quality)


    # 5x6 plots; for each quality in QUALITY
    # each row corresponds to a different alpha value, show the watermarked noisy image and the recovered watermark side by side for each of the 3 images
    
    ##############################
    # Increase the number of alpha values to display in the plots if recovered QR code is not can't be decoded by QR scanner.
    NUMBER_OF_ALPHA_TO_DISPLAY = 5  
    ##############################

    for quality_idx, quality in enumerate(QUALITY):
        fig, axes = plt.subplots(len(ALPHAS[:NUMBER_OF_ALPHA_TO_DISPLAY]), 6, figsize=(18, 3 * len(ALPHAS[:NUMBER_OF_ALPHA_TO_DISPLAY])))
        for alpha_idx, alpha in enumerate(ALPHAS[:NUMBER_OF_ALPHA_TO_DISPLAY]):
            for img_idx in range(3):
                axes[alpha_idx, img_idx * 2].imshow(watermarked_images_rgb[quality_idx][alpha_idx][img_idx])
                axes[alpha_idx, img_idx * 2].set_title(f"Watermarked (α={alpha})")
                axes[alpha_idx, img_idx * 2].axis("off")

                axes[alpha_idx, img_idx * 2 + 1].imshow(recovered_watermarks_rgb[quality_idx][alpha_idx][img_idx])
                axes[alpha_idx, img_idx * 2 + 1].set_title(f"Recovered (α={alpha})")
                axes[alpha_idx, img_idx * 2 + 1].axis("off")
        # set title for whole figure
        fig.suptitle(f"Watermarked and Recovered Images with JPG Compression (quality={quality})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle

        # plt.tight_layout()
        plt.savefig(f"plots/task4_watermark_results_quality_{quality}.png", metadata={"Author": getpass.getuser()})
    

    

    noisy_watermarked_psnr = []
    recovered_watermark_psnr = []

    for quality_idx, quality in enumerate(QUALITY):
        noisy_watermarked_psnr_per_quality = []
        recovered_watermark_psnr_per_quality = []
        for image_idx, image in enumerate(images_rgb):
            noisy_watermarked_psnr_per_image = []
            recovered_watermark_psnr_per_image = []
            for alpha_idx, alpha in enumerate(ALPHAS):
                # Compute PSNR between (a) original and watermarked compressed images; (b) original watermark and recovered watermark from the compressed image
                # ###############################################
                
                # Fetch images from our previously built lists
                compressed_img = watermarked_images_rgb[quality_idx][alpha_idx][image_idx]
                recovered_img = recovered_watermarks_rgb[quality_idx][alpha_idx][image_idx]
                
                # (a) Compare pristine original cover against the compressed, watermarked version
                psnr_compressed_cover = compute_psnr(image, compressed_img)
                
                # (b) Compare original pristine QR code against the extracted one
                psnr_recovered_qr = compute_psnr(watermark_rgb, recovered_img)
                
                noisy_watermarked_psnr_per_image.append(psnr_compressed_cover)
                recovered_watermark_psnr_per_image.append(psnr_recovered_qr)
                
                # ###############################################

            noisy_watermarked_psnr_per_quality.append(noisy_watermarked_psnr_per_image)
            recovered_watermark_psnr_per_quality.append(recovered_watermark_psnr_per_image)

        noisy_watermarked_psnr.append(noisy_watermarked_psnr_per_quality)
        recovered_watermark_psnr.append(recovered_watermark_psnr_per_quality)


    # 2x3 plots; the column represents the images;
    # 1st row plots psnr vs alpha for the compressed watermarked images; the legend specifies the quality values
    # 2nd row plots psnr vs alpha for the recovered watermarks; the legend specifies the quality values

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for image_idx in range(3):
        for quality_idx, quality in enumerate(QUALITY):
            axes[0, image_idx].plot(ALPHAS, noisy_watermarked_psnr[quality_idx][image_idx], label=f"quality={quality}")
            axes[1, image_idx].plot(ALPHAS, recovered_watermark_psnr[quality_idx][image_idx], label=f"quality={quality}")

        axes[0, image_idx].set_title(f"Compressed Watermarked PSNR vs Alpha (Image {image_idx + 1})")
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
    plt.savefig("plots/task4_psnr_vs_alpha.png", metadata={"Author": getpass.getuser()})

# %%