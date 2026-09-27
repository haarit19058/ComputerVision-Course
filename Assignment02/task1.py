# %%
# Imports
# ================================================================
# Don't update these imports.
import os
import cv2
import getpass
import numpy as np
import matplotlib.pyplot as plt

# %%
# Helper Functions
# ================================================================
# These are helper functions that you can use in your implementation.
# Don't update these functions.

os.makedirs("plots", exist_ok=True)


def load_image_as_rgb(image_path: str) -> np.ndarray:
    """Load an image as RGB floats in [0, 1], shape (H, W, 3).

    This is the normal way to load an image in this assignment. Waldo is
    defined by his RED and WHITE stripes, and throwing that away costs you
    real information -- one of the questions asks you to measure how much.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR
    return image.astype(np.float64) / 255.0




def load_image_as_grayscale(image_path: str) -> np.ndarray:
    """Load straight to luminance, shape (H, W). Used by Task 3."""
    return to_grayscale(load_image_as_rgb(image_path))


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.06,
                       seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise to every channel, then clip to [0, 1]."""
    rng = np.random.default_rng(seed)
    return np.clip(image + rng.normal(0.0, sigma, image.shape), 0.0, 1.0)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.04,
                          seed: int = 0) -> np.ndarray:
    """Knock a fraction of PIXELS to pure black or pure white.

    Note it is whole pixels, not individual channel samples: a dead sensor
    pixel loses all three channels at once. Corrupting channels separately
    would give coloured confetti, which is not what this noise model is.
    """
    rng = np.random.default_rng(seed)
    out = image.copy()
    hit = rng.random(image.shape[:2]) < amount
    values = rng.integers(0, 2, int(hit.sum())).astype(np.float64)
    if image.ndim == 3:
        out[hit] = values[:, None]          # same value in all channels
    else:
        out[hit] = values
    return out




# ================================================================
# %%
# Task 1
# ======


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, done as two 1-D passes instead of one 2-D convolution.

    Provided because Tasks 3 and 4 call it hundreds of times and need it to be
    fast. A 2-D Gaussian is separable, so blurring rows and then columns costs
    O(H*W*k) instead of O(H*W*k^2) -- worth understanding, and it is one of
    the report questions.

    Works on (H, W) and on (H, W, 3): a linear filter acts on each colour
    channel independently, so only the two spatial axes are padded.
    """
    #TODO: Write your Code for Gaussian Blur

    return image

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse an RGB image to one luminance channel, shape (H, W).

    The Rec.601 weights below are not arbitrary: the eye is far more
    sensitive to green than to blue, so a plain (R+G+B)/3 makes blues look
    lighter than they are.

    Gradient-based code (Sobel, Harris, SIFT-like descriptors in Task 3)
    needs a SCALAR field to differentiate -- "the direction of steepest
    increase" has no meaning for a 3-vector without first choosing how to
    combine the channels. So Task 3 works on this, while Tasks 1 and 2 work
    in full colour. A 2-D array is passed through unchanged.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return image

    #TODO: Write the code for Grayscale
    return image

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an image with a kernel, keeping the same output size.

    Pad the image by half the kernel on each side (reflect the edge pixels),
    flip the kernel on both axes, then slide it over every pixel.

    COLOUR: the input may be (H, W) or (H, W, 3). A linear filter treats the
    channels independently, so the same kernel is applied to each one and
    nothing mixes between them. You do not need a separate code path for
    this -- pad only the two SPATIAL axes and the arithmetic broadcasts.
    Getting that padding wrong (padding the channel axis as well) is the
    single most common bug here.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W) or (H, W, 3), values in [0, 1]
    kernel : np.ndarray
        Kernel of shape (kh, kw), both odd

    Returns
    -------
    np.ndarray
        Filtered image, same shape as the input
    """
    return np.zeros_like(image)  # comment this line and write your code for the function


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Build a (size, size) Gaussian kernel that sums to 1.

    Sample exp(-(x^2 + y^2) / (2 * sigma^2)) on a grid centred on zero, then
    divide by the sum of the samples (not by the analytic constant).

    Parameters
    ----------
    size : int
        Side length, must be odd
    sigma : float
        Standard deviation in pixels

    Returns
    -------
    np.ndarray
        Kernel of shape (size, size), summing to 1
    """
    return np.zeros((size, size))  # comment this line and write your code for the function


def median_filter(image: np.ndarray, size: int) -> np.ndarray:
    """Replace every pixel by the median of its (size, size) neighbourhood.

    This one cannot be written as a convolution -- explain why in your answers.

    COLOUR: take the median of each channel separately. Be aware that this
    is a choice, not the only option: the output colour at a pixel may be one
    that appears nowhere in the input window, because the red median and the
    blue median can come from different neighbours. A "vector median" that
    returns an actual input pixel avoids that. You are asked about this.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W) or (H, W, 3)
    size : int
        Side length of the window, must be odd

    Returns
    -------
    np.ndarray
        Filtered image, same shape as the input
    """
    return np.zeros_like(image)  # comment this line and write your code for the function


def sobel_edges(image: np.ndarray) -> tuple:
    """Image gradients from the Sobel kernels, and their polar form.


    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W) or (H, W, 3)

    Returns
    -------
    tuple
        gx (H, W), gy (H, W), magnitude (H, W), orientation (H, W) in degrees
        wrapped into [0, 360)
    """
    return (np.zeros(image.shape[:2]), np.zeros(image.shape[:2]),
            np.zeros(image.shape[:2]), np.zeros(image.shape[:2]))  # comment this line and write your code for the function


def psnr(image: np.ndarray, reference: np.ndarray) -> float:
    """Peak signal-to-noise ratio, in decibels. Higher is better.

        MSE  = mean((image - reference) ** 2)
        PSNR = 10 * log10(1.0 / MSE)          because our images are in [0, 1]


    Parameters
    ----------
    image : np.ndarray
        The image being judged, shape (H, W) or (H, W, 3)
    reference : np.ndarray
        The clean image to compare against, same shape

    Returns
    -------
    float
        PSNR in dB, or float("inf") when the two images are identical
    """
    return 0.0  # comment this line and write your code for the function


def ssim(image: np.ndarray, reference: np.ndarray) -> tuple:
    """Structural similarity: compare local statistics, not just pixel error.

    Inside a Gaussian window of sigma 1.5, with local means mu, variances var
    and covariance cov, and C1 = 0.01 ** 2, C2 = 0.03 ** 2:

        SSIM = ((2*mu_a*mu_b + C1) * (2*cov + C2))
               / ((mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2))

    Every one of those local statistics is a Gaussian-weighted average, so the
    provided gaussian_blur computes all five of them.


    Parameters
    ----------
    image : np.ndarray
        The image being judged, shape (H, W) or (H, W, 3)
    reference : np.ndarray
        The clean image to compare against, same shape

    Returns
    -------
    tuple
        mean_ssim (float, 1.0 means identical) and ssim_map, the same shape
        as the input. The map is worth plotting: it shows you WHERE a filter
        did damage.
    """
    return 0.0, np.zeros_like(image)  # comment this line and write your code for the function


if __name__ == "__main__":

    clean = load_image_as_rgb("imgs/scene_easy.png")
    noisy = add_gaussian_noise(clean, sigma=0.06)
    salted = add_salt_pepper_noise(clean, amount=0.04)

    # Denoise each kind of noise with each kind of filter, then score them.
    # #############################################
    # Comment these lines and write your code here
    box_on_gaussian = np.zeros_like(noisy)
    gauss_on_gaussian = np.zeros_like(noisy)
    median_on_salt = np.zeros_like(salted)
    gauss_on_salt = np.zeros_like(salted)
    # #############################################

    results = [("box 9x9 on gaussian", noisy, box_on_gaussian),
               ("gaussian s=2 on gaussian", noisy, gauss_on_gaussian),
               ("median 5x5 on salt+pepper", salted, median_on_salt),
               ("gaussian s=1.5 on salt+pepper", salted, gauss_on_salt)]

    print("Colour images, {} channels".format(clean.shape[2]))
    print("{:<32}{:>10}{:>10}{:>10}{:>10}".format(
        "filter", "PSNR in", "PSNR out", "SSIM in", "SSIM out"))
    for label, before, after in results:
        print("{:<32}{:>10.2f}{:>10.2f}{:>10.4f}{:>10.4f}".format(
            label, psnr(before, clean), psnr(after, clean),
            ssim(before, clean)[0], ssim(after, clean)[0]))

    _gx, _gy, magnitude, _orientation = sobel_edges(clean)
    _mean_ssim, ssim_map = ssim(median_on_salt, clean)

    # 2x3 plots: noisy inputs and their best filter, then the SSIM map
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [(clean, "Clean", None), (noisy, "Gaussian noise", None),
              (gauss_on_gaussian, "Gaussian filtered", None),
              (salted, "Salt & pepper", None),
              (median_on_salt, "Median filtered", None),
              (ssim_map.mean(axis=2), "SSIM map (dark = damaged)", "gray")]
    for ax, (img, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(np.clip(img, 0.0, 1.0), cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("plots/task1_results.png", metadata={"Author": getpass.getuser()})
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(clean)
    axes[0].set_title("Input (RGB)")
    axes[1].imshow(magnitude, cmap="gray")
    axes[1].set_title("Sobel gradient magnitude (on luminance)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("plots/task1_edges.png", metadata={"Author": getpass.getuser()})
    plt.close()
    print("\nwrote plots/task1_results.png and plots/task1_edges.png")
