# %%
# Imports
# ================================================================
# Don't update these imports.
import os
import cv2
import getpass
import numpy as np
import matplotlib.pyplot as plt
from task1 import *
from task2 import *

# %%
# Helper Functions
# ================================================================
# These are helper functions that you can use in your implementation.
# Don't update these functions.

os.makedirs("plots", exist_ok=True)


TEMPLATE_KEYPOINTS = 100 
SCENE_KEYPOINTS = 500    


def sample_bilinear(image: np.ndarray, ys: np.ndarray,
                    xs: np.ndarray) -> np.ndarray:
    """Read the image at fractional coordinates, clamping outside the border."""
    h, w = image.shape
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = ys - y0, xs - x0
    return ((1 - fy) * (1 - fx) * image[y0, x0]
            + (1 - fy) * fx * image[y0, x1]
            + fy * (1 - fx) * image[y1, x0]
            + fy * fx * image[y1, x1])


def orientation_histogram(magnitude: np.ndarray, orientation: np.ndarray,
                          n_bins: int, weights: np.ndarray = None) -> np.ndarray:
    """Histogram of gradient directions over [0, 360), weighted by magnitude.

    Each sample goes entirely into one bin (hard assignment). Bin b covers
    [b * 360 / n_bins, (b + 1) * 360 / n_bins).
    """
    mag = np.asarray(magnitude).ravel()
    ori = np.asarray(orientation).ravel()
    w = mag if weights is None else mag * np.asarray(weights).ravel()
    index = np.clip(np.floor(ori / (360.0 / n_bins)).astype(int), 0, n_bins - 1)
    return np.bincount(index, weights=w, minlength=n_bins).astype(np.float64)


# ================================================================
# %%
# Task 3
# ======
#

def harris_response(image: np.ndarray, sigma_d: float = 1.0,
                    sigma_i: float = 2.0, k: float = 0.04) -> np.ndarray:
    """Harris cornerness: how much the image changes in EVERY direction at once.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    sigma_d : float
        Blur applied before differentiating
    sigma_i : float
        Blur applied to the gradient products
    k : float
        Harris constant, usually 0.04 to 0.06

    Returns
    -------
    np.ndarray
        Cornerness map of shape (H, W)
    """
    blurred_img = gaussian_blur(image, sigma_d)
    gx, gy, _, _ = sobel_edges(blurred_img)
    
    ix2 = gaussian_blur(gx * gx, sigma_i)
    iy2 = gaussian_blur(gy * gy, sigma_i)
    ixy = gaussian_blur(gx * gy, sigma_i)
    
    det = (ix2 * iy2) - (ixy ** 2)
    trace = ix2 + iy2
    
    return det - k * (trace ** 2)


def detect_corners(image: np.ndarray, threshold_rel: float = 0.02,
                   min_distance: int = 3, max_corners: int = 120,
                   border: int = 4) -> np.ndarray:
    """Turn the Harris map into a list of corner positions.


    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    threshold_rel : float
        Fraction of the maximum response a corner must reach
    min_distance : int
        Minimum separation between corners
    max_corners : int
        Keep at most this many
    border : int
        Width of the ignored frame

    Returns
    -------
    np.ndarray
        Shape (N, 2), columns (y, x), strongest first. (0, 2) if none.
    """
    if image.size == 0:
        return np.zeros((0, 2))
        
    threshold = threshold_rel * np.max(image)
    ys, xs = np.nonzero(image >= threshold)
    scores = image[ys, xs]
    
    # Filter out points too close to the border
    h, w = image.shape
    valid_mask = (ys >= border) & (ys < h - border) & (xs >= border) & (xs < w - border)
    ys, xs, scores = ys[valid_mask], xs[valid_mask], scores[valid_mask]
    
    if len(scores) == 0:
        return np.zeros((0, 2))
        
    order = np.argsort(scores)[::-1]
    ys, xs, scores = ys[order], xs[order], scores[order]
    
    kept = []
    for y, x in zip(ys, xs):
        if len(kept) >= max_corners:
            break
            
        conflict = False
        for ky, kx in kept:
            if max(abs(y - ky), abs(x - kx)) < min_distance:
                conflict = True
                break
                
        if not conflict:
            kept.append((y, x))
            
    if not kept:
        return np.zeros((0, 2))
        
    return np.array(kept, dtype=np.float64)


def dog_keypoints(image: np.ndarray, sigma0: float = 0.8, n_scales: int = 8,
                  k: float = 1.25, threshold: float = 0.005,
                  border: int = 4, max_keypoints: int = 800) -> np.ndarray:
    """Find blob-like keypoints, and the SIZE each one lives at.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    sigma0, n_scales, k : float, int, float
        The scale ladder
    threshold : float
        Minimum absolute response
    border : int
        Ignored frame width
    max_keypoints : int
        Keep at most this many

    Returns
    -------
    np.ndarray
        Shape (N, 3), columns (y, x, sigma), strongest first. (0, 3) if none.
    """
    sigmas = [sigma0 * (k ** i) for i in range(n_scales + 2)]
    ladder = [gaussian_blur(image, s) for s in sigmas]
    dogs = np.stack([ladder[i + 1] - ladder[i] for i in range(n_scales + 1)], axis=0)
    
    # Identify local extrema in the 3x3x3 space-scale cube
    is_max = np.ones_like(dogs, dtype=bool)
    is_min = np.ones_like(dogs, dtype=bool)
    
    for ds in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if ds == 0 and dy == 0 and dx == 0:
                    continue
                shifted = np.roll(dogs, shift=(ds, dy, dx), axis=(0, 1, 2))
                is_max &= (dogs > shifted)
                is_min &= (dogs < shifted)
                
    extrema_mask = (is_max | is_min) & (np.abs(dogs) >= threshold)
    
    # Exclude extrema on the scale boundaries and spatial borders
    extrema_mask[0, :, :] = False
    extrema_mask[-1, :, :] = False
    extrema_mask[:, :border, :] = False
    extrema_mask[:, -border:, :] = False
    extrema_mask[:, :, :border] = False
    extrema_mask[:, :, -border:] = False
    
    ss, ys, xs = np.nonzero(extrema_mask)
    scores = np.abs(dogs[ss, ys, xs])
    
    if len(scores) == 0:
        return np.zeros((0, 3))
        
    order = np.argsort(scores)[::-1][:max_keypoints]
    ss, ys, xs = ss[order], ys[order], xs[order]
    
    keypoint_sigmas = np.array([sigmas[s] for s in ss])
    
    return np.column_stack((ys, xs, keypoint_sigmas)).astype(np.float64)


def dominant_orientation(image: np.ndarray, y: float, x: float,
                         sigma: float = 1.6, n_bins: int = 36) -> float:
    """The direction a keypoint's neighbourhood mostly points in.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    y, x : float
        Keypoint position
    sigma : float
        Scale the keypoint was found at
    n_bins : int
        Number of orientation bins

    Returns
    -------
    float
        Angle in degrees in [0, 360), or 0.0 if the window has no gradient
    """
    radius = int(np.ceil(3.0 * sigma))
    
    h, w = image.shape
    y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
    x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
    
    if y1 <= y0 or x1 <= x0:
        return 0.0
        
    patch = image[y0:y1, x0:x1]
    gx, gy, magnitude, orientation = sobel_edges(patch)
    
    # Create spatial gaussian weights for the patch
    yy, xx = np.mgrid[y0:y1, x0:x1]
    weights = np.exp(-((xx - x)**2 + (yy - y)**2) / (2.0 * sigma**2))
    
    hist = orientation_histogram(magnitude, orientation, n_bins, weights)
    
    if np.sum(hist) == 0:
        return 0.0
        
    peak_bin = np.argmax(hist)
    return float(peak_bin * (360.0 / n_bins))


def describe_keypoints(image: np.ndarray, keypoints: np.ndarray,
                       orientations: np.ndarray = None) -> np.ndarray:
    """Describe each keypoint by a grid of gradient-orientation histograms.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    keypoints : np.ndarray
        Shape (N, 3), columns (y, x, sigma)
    orientations : np.ndarray, optional
        Shape (N,) in degrees; None means upright descriptors

    Returns
    -------
    np.ndarray
        Shape (N, 128), one row per keypoint
    """
    N = len(keypoints)
    if N == 0:
        return np.zeros((0, 128))
        
    descriptors = np.zeros((N, 128), dtype=np.float64)
    
    # Compute image gradients once
    gx, gy, mag_full, ori_full = sobel_edges(image)
    
    for i in range(N):
        y, x, sigma = keypoints[i]
        angle = orientations[i] if orientations is not None else 0.0
        
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # 4x4 grid of cells, each cell is 3*sigma wide
        cell_w = 3.0 * sigma
        
        # We sample a 16x16 grid of points (4 points per cell) to build histograms
        coords = np.linspace(-1.5 * cell_w, 1.5 * cell_w, 16)
        xx_grid, yy_grid = np.meshgrid(coords, coords)
        
        # Rotate grid by -theta to align with keypoint orientation
        xx_rot = xx_grid * cos_t - yy_grid * sin_t
        yy_rot = xx_grid * sin_t + yy_grid * cos_t
        
        sample_x = x + xx_rot
        sample_y = y + yy_rot
        
        # Sample magnitudes and orientations at fractional coordinates
        sampled_mag = sample_bilinear(mag_full, sample_y, sample_x)
        sampled_ori = sample_bilinear(ori_full, sample_y, sample_x)
        
        # Adjust sampled orientations relative to the keypoint's dominant angle
        sampled_ori = (sampled_ori - angle) % 360.0
        
        # Apply global Gaussian weighting window (sigma = half the descriptor window)
        global_weights = np.exp(-(xx_grid**2 + yy_grid**2) / (2.0 * (8.0 * sigma)**2))
        weighted_mag = sampled_mag * global_weights
        
        # Accumulate the 128-d descriptor (16 cells x 8 bins)
        desc = np.zeros(128, dtype=np.float64)
        
        for row in range(4):
            for col in range(4):
                cell_mag = weighted_mag[row*4:(row+1)*4, col*4:(col+1)*4]
                cell_ori = sampled_ori[row*4:(row+1)*4, col*4:(col+1)*4]
                
                hist = orientation_histogram(cell_mag, cell_ori, n_bins=8)
                desc[(row * 4 + col) * 8 : (row * 4 + col + 1) * 8] = hist
                
        # Normalize to unit length
        norm = np.linalg.norm(desc)
        if norm > 1e-8:
            desc /= norm
            
        descriptors[i] = desc
        
    return descriptors


if __name__ == "__main__":

    template = load_image_as_grayscale("imgs/waldo_template.png")
    scene = load_image_as_grayscale("imgs/scene_easy.png")

    response = harris_response(template)
    corners = detect_corners(response, max_corners=TEMPLATE_KEYPOINTS)
    
    keypoints = dog_keypoints(template, max_keypoints=TEMPLATE_KEYPOINTS)
    
    angles = np.array([dominant_orientation(template, y, x, s) for y, x, s in keypoints])
    
    descriptors = describe_keypoints(template, keypoints, angles)

    print("Harris corners on the template :", len(corners))
    print("DoG keypoints on the template  :", len(keypoints))
    print("sigmas found                   :",
          np.round(np.unique(np.round(keypoints[:, 2], 2)), 2) if len(keypoints) else [])
    norms = np.linalg.norm(descriptors, axis=1) if len(descriptors) else np.zeros(1)
    print("descriptors                    :", descriptors.shape,
          "row norms in [{:.3f}, {:.3f}]".format(norms.min(), norms.max()) if len(descriptors) else "no descriptors")

    scene_keypoints = dog_keypoints(scene, max_keypoints=SCENE_KEYPOINTS)
    print("DoG keypoints on scene_easy    :", len(scene_keypoints))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(template, cmap="gray")
    axes[0].set_title("Template")
    axes[1].imshow(response, cmap="gray")
    axes[1].set_title("Harris response R")
    axes[2].imshow(template, cmap="gray")
    if len(corners):
        axes[2].scatter(corners[:, 1], corners[:, 0], s=18,
                        facecolors="none", edgecolors="lime")
    axes[2].set_title("Harris corners")
    axes[3].imshow(template, cmap="gray")
    if len(keypoints):
        axes[3].scatter(keypoints[:, 1], keypoints[:, 0], s=keypoints[:, 2] * 12,
                        facecolors="none", edgecolors="red")
    axes[3].set_title("DoG keypoints (size = sigma)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("plots/task3_results.png", metadata={"Author": getpass.getuser()})
    print("\nwrote plots/task3_results.png")

    print("\n--- 3.7 Ladder Trade Test ---")
    configs = [
        (0.8, 8, 1.25),
        (0.5, 10, 1.2),
        (1.5, 5, 1.5)
    ]

    for sigma0, n_scales, k in configs:
        kps = dog_keypoints(template, sigma0=sigma0, n_scales=n_scales, k=k, max_keypoints=TEMPLATE_KEYPOINTS)
        if len(kps) > 0:
            sig_min, sig_max = kps[:, 2].min(), kps[:, 2].max()
            print(f"sigma0={sigma0}, n_scales={n_scales}, k={k} -> Keypoints: {len(kps)}, σ range: [{sig_min:.2f}, {sig_max:.2f}]")
        else:
            print(f"sigma0={sigma0}, n_scales={n_scales}, k={k} -> Keypoints: 0, σ range: [N/A]")

