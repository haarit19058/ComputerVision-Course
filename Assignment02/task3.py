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


TEMPLATE_KEYPOINTS = 0 #TODO: Make your own Choice
SCENE_KEYPOINTS = 0    #TODO: Make your own chouce


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
    return np.zeros_like(image)  # comment this line and write your code for the function


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
    return np.zeros((0, 2))  # comment this line and write your code for the function


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
    return np.zeros((0, 3))  # comment this line and write your code for the function


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
    return 0.0  # comment this line and write your code for the function


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
    return np.zeros((len(keypoints), 128))  # comment this line and write your code for the function


if __name__ == "__main__":

    template = load_image_as_grayscale("imgs/waldo_template.png")
    scene = load_image_as_grayscale("imgs/scene_easy.png")

    # #############################################
    # Comment these lines and write your code here
    response = np.zeros_like(template)
    corners = np.zeros((0, 2))
    keypoints = np.zeros((0, 3))
    angles = np.zeros(0)
    descriptors = np.zeros((0, 128))
    # #############################################

    print("Harris corners on the template :", len(corners))
    print("DoG keypoints on the template  :", len(keypoints))
    print("sigmas found                   :",
          np.round(np.unique(np.round(keypoints[:, 2], 2)), 2))
    norms = np.linalg.norm(descriptors, axis=1) if len(descriptors) else np.zeros(1)
    print("descriptors                    :", descriptors.shape,
          "row norms in [{:.3f}, {:.3f}]".format(norms.min(), norms.max()))

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
