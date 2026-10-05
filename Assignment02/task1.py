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
    if sigma <= 0:
        return image.copy()

    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()

    # Pass 1: Horizontal blur along columns
    row_blur = convolve2d(image, kernel_1d[None, :])
    # Pass 2: Vertical blur along rows
    return convolve2d(row_blur, kernel_1d[:, None])


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

    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    return np.dot(image[..., :3], weights)


# Modified: Added `pad_mode` parameter with a default of "reflect" for Q1.10
def convolve2d(image: np.ndarray, kernel: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
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
    pad_mode : str
        Padding mode used by np.pad. Default is "reflect".

    Returns
    -------
    np.ndarray
        Filtered image, same shape as the input
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    pad_width = [(pad_h, pad_h), (pad_w, pad_w)]
    if image.ndim == 3:
        pad_width.append((0, 0))

    # Modified: Using the pad_mode parameter instead of hardcoding "reflect"
    padded = np.pad(image, pad_width, mode=pad_mode)
    k_flipped = kernel[::-1, ::-1]

    h_out, w_out = image.shape[:2]
    out = np.zeros_like(image, dtype=np.float64)

    for i in range(kh):
        for j in range(kw):
            window = padded[i : i + h_out, j : j + w_out]
            out += window * k_flipped[i, j]

    return out


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
    radius = size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(coords, coords)

    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


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
    radius = size // 2
    pad_width = [(radius, radius), (radius, radius)]
    if image.ndim == 3:
        pad_width.append((0, 0))

    padded = np.pad(image, pad_width, mode="reflect")

    if image.ndim == 2:
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
        return np.median(windows, axis=(-2, -1))
    else:
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size), axis=(0, 1))
        return np.median(windows, axis=(-2, -1))


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
    gray = to_grayscale(image)

    kx = np.array([[-1.0, 0.0, 1.0],
                   [-2.0, 0.0, 2.0],
                   [-1.0, 0.0, 1.0]], dtype=np.float64)

    ky = np.array([[-1.0, -2.0, -1.0],
                   [ 0.0,  0.0,  0.0],
                   [ 1.0,  2.0,  1.0]], dtype=np.float64)

    gx = convolve2d(gray, kx)
    gy = convolve2d(gray, ky)

    # Polar form: magnitude and orientation (in degrees, wrapped to [0, 360))
    magnitude = np.hypot(gx, gy)
    orientation = np.degrees(np.arctan2(gy, gx)) % 360.0

    return gx, gy, magnitude, orientation


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
    mse = np.mean((image - reference) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


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
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_a = gaussian_blur(image, 1.5)
    mu_b = gaussian_blur(reference, 1.5)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    var_a = gaussian_blur(image ** 2, 1.5) - mu_a_sq
    var_b = gaussian_blur(reference ** 2, 1.5) - mu_b_sq
    cov_ab = gaussian_blur(image * reference, 1.5) - mu_ab

    numerator = (2.0 * mu_ab + c1) * (2.0 * cov_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (var_a + var_b + c2)

    ssim_map = numerator / denominator
    mean_ssim = float(np.mean(ssim_map))

    return mean_ssim, ssim_map


if __name__ == "__main__":

    clean = load_image_as_rgb("imgs/scene_easy.png")
    noisy = add_gaussian_noise(clean, sigma=0.06)
    salted = add_salt_pepper_noise(clean, amount=0.04)

    # ---------------------------------------------------------
    # ORIGINAL RGB RUN
    # ---------------------------------------------------------
    # Denoise each kind of noise with each kind of filter, then score them.
    box_9x9 = np.ones((9, 9), dtype=np.float64) / 81.0
    box_on_gaussian = convolve2d(noisy, box_9x9)
    gauss_on_gaussian = gaussian_blur(noisy, sigma=2.0)
    median_on_salt = median_filter(salted, size=5)
    gauss_on_salt = gaussian_blur(salted, sigma=1.5)

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
            
    # ---------------------------------------------------------
    # NEW: GRAYSCALE RUN (For Question 1.9)
    # ---------------------------------------------------------
    print("\n--- Running Grayscale Table for Q 1.9 ---")
    clean_gray = to_grayscale(clean)
    noisy_gray = add_gaussian_noise(clean_gray, sigma=0.06)
    salted_gray = add_salt_pepper_noise(clean_gray, amount=0.04)

    box_on_gaussian_g = convolve2d(noisy_gray, box_9x9)
    gauss_on_gaussian_g = gaussian_blur(noisy_gray, sigma=2.0)
    median_on_salt_g = median_filter(salted_gray, size=5)
    gauss_on_salt_g = gaussian_blur(salted_gray, sigma=1.5)

    results_gray = [
        ("box 9x9 on gaussian", noisy_gray, box_on_gaussian_g),
        ("gaussian s=2 on gaussian", noisy_gray, gauss_on_gaussian_g),
        ("median 5x5 on salt+pepper", salted_gray, median_on_salt_g),
        ("gaussian s=1.5 on salt+pepper", salted_gray, gauss_on_salt_g)
    ]

    print("Grayscale images, 1 channel")
    print("{:<32}{:>10}{:>10}{:>10}{:>10}".format(
        "filter", "PSNR in", "PSNR out", "SSIM in", "SSIM out"))
    for label, before, after in results_gray:
        print("{:<32}{:>10.2f}{:>10.2f}{:>10.4f}{:>10.4f}".format(
            label, psnr(before, clean_gray), psnr(after, clean_gray),
            ssim(before, clean_gray)[0], ssim(after, clean_gray)[0]))

    # ---------------------------------------------------------
    # NEW: PADDING (For Question 1.10)
    # ---------------------------------------------------------
    print("\n--- Generating Padding Artefact for Q 1.10 ---")
    # Use a large kernel so the edge artifact is visually obvious
    large_kernel = gaussian_kernel(31, sigma=5.0) 
    
    # Run with reflect padding (standard behavior)
    blurred_reflect = convolve2d(clean, large_kernel, pad_mode="reflect")
    # Run with zero padding to see the artifact
    blurred_zero = convolve2d(clean, large_kernel, pad_mode="constant")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.clip(blurred_reflect, 0.0, 1.0))
    axes[0].set_title("Large Blur (Reflect Padding)")
    axes[1].imshow(np.clip(blurred_zero, 0.0, 1.0))
    axes[1].set_title("Large Blur (Zero Padding) - Note borders")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("plots/task1_padding_artefact.png", metadata={"Author": getpass.getuser()})
    plt.close()

    # ---------------------------------------------------------
    # ORIGINAL PLOTTING CODE
    # ---------------------------------------------------------
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
    
    # Updated print statement to reflect the new plot
    print("\nwrote plots/task1_results.png, plots/task1_edges.png, and plots/task1_padding_artefact.png")