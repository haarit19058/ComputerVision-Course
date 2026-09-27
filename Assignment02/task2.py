# %%
# Imports
# ================================================================
# Don't update these imports.
import os
import cv2
import json
import getpass
import numpy as np
import matplotlib.pyplot as plt
from task1 import *

# %%
# Helper Functions
# ================================================================
# These are helper functions that you can use in your implementation.
# Don't update these functions.

os.makedirs("plots", exist_ok=True)

# Template sizes to search over, as multiples of the supplied template.
SCALES = (0.6, 0.72, 0.85, 1.0, 1.15, 1.35, 1.5, 1.7)


def load_ground_truth(scene_name: str) -> tuple:
    """Waldo's true box (x, y, w, h) for a scene. Use it to SCORE, never to find."""
    with open("imgs/annotations.json") as fh:
        annotations = json.load(fh)
    for scene in annotations["scenes"]:
        if scene["name"] == scene_name:
            box = scene["waldo"]
            return (box["x"], box["y"], box["w"], box["h"])
    raise KeyError("no scene named {!r}".format(scene_name))


def resize_template(template: np.ndarray, scale: float) -> np.ndarray:
    """Scale a template up or down. Resampling is not what this task teaches.

    Handles (h, w) and (h, w, 3) alike -- cv2.resize keeps the channel axis.
    """
    h = max(4, int(round(template.shape[0] * scale)))
    w = max(4, int(round(template.shape[1] * scale)))
    return cv2.resize(template, (w, h), interpolation=cv2.INTER_LINEAR)


def sliding_windows(image: np.ndarray, win_h: int, win_w: int) -> np.ndarray:
    """Every (win_h, win_w) window of ONE channel, as a read-only view.

    Shape (H - win_h + 1, W - win_w + 1, win_h, win_w). It is a VIEW, so it
    costs no memory -- but reshaping it to 2-D would force an 8 GB copy on a
    full scene. Contract it with np.einsum instead.

    Pass a single 2-D channel, not a colour image: three cheap views cost far
    less than one view three times the size, and it keeps the einsum
    subscripts simple.
    """
    if image.ndim != 2:
        raise ValueError("pass one 2-D channel, got shape {}".format(image.shape))
    return np.lib.stride_tricks.sliding_window_view(image, (win_h, win_w))


def iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def draw_box(image: np.ndarray, box: tuple, colour=(0, 1, 0)) -> np.ndarray:
    """Draw a rectangle on a copy of an RGB image."""
    out = image.copy()
    x, y, w, h = [int(round(v)) for v in box]
    cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
    return out


# ================================================================
# %%
# Task 2
# ======


def integral_image(image: np.ndarray) -> np.ndarray:
    """Summed-area table, so any box sum becomes four lookups.

    ii[y, x] must equal the sum of image[:y, :x], with a zero first row and
    first column. That zero border is what lets box_sum below stay
    branch-free, so do not leave it out. Two cumulative sums.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)

    Returns
    -------
    np.ndarray
        Table of shape (H + 1, W + 1)
    """
    return np.zeros((image.shape[0] + 1, image.shape[1] + 1))  # comment this line and write your code for the function


def box_sum(ii: np.ndarray, y0: np.ndarray, x0: np.ndarray,
            h: int, w: int) -> np.ndarray:
    """Sum of every (h, w) box with top-left (y0, x0), from a summed-area table.

    y0 and x0 may be arrays, so this does every window at once. Four lookups,
    whatever the box size -- that is the whole point of the table.

    Parameters
    ----------
    ii : np.ndarray
        Output of integral_image
    y0, x0 : np.ndarray
        Top-left corners, any common shape
    h, w : int
        Box height and width

    Returns
    -------
    np.ndarray
        Sums, broadcast to the shape of y0 and x0
    """
    return np.zeros_like(y0, dtype=np.float64)  # comment this line and write your code for the function


def ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-mean normalised cross-correlation at every position, in COLOUR.

    For a single channel, with window I of the image and template T:

        NCC = sum((I - mean I) * (T - mean T))
              / (||I - mean I||  *  ||T - mean T||)

    Higher is better, and it lies in [-1, 1]. Unlike a plain sum of squared
    differences it is unchanged by I -> a*I + b for a > 0, which is why it
    survives the uneven lighting in scene_medium.

    Parameters
    ----------
    image : np.ndarray
        Scene of shape (H, W, 3)
    template : np.ndarray
        Template of shape (th, tw, 3)

    Returns
    -------
    np.ndarray
        Score map of shape (H - th + 1, W - tw + 1), still 2-D: one score
        per position, not one per channel.
    """
    return np.zeros((image.shape[0] - template.shape[0] + 1,
                     image.shape[1] - template.shape[1] + 1))  # comment this line and write your code for the function


def find_peaks(score_map: np.ndarray, threshold: float,
               min_distance: int = 8, max_peaks: int = 20) -> np.ndarray:
    """Pick the strongest well-separated positions out of a score map.

    Take every position scoring >= threshold, sort them best first, then walk
    the list keeping a position only if it is at least min_distance away from
    every position already kept. Distance here is CHEBYSHEV: the larger of the
    row gap and the column gap. Stop once max_peaks are kept.

    Parameters
    ----------
    score_map : np.ndarray
        Output of ncc_map
    threshold : float
        Minimum score to consider
    min_distance : int
        Minimum separation between kept peaks
    max_peaks : int
        Stop after this many

    Returns
    -------
    np.ndarray
        Shape (K, 3), columns (y, x, score), best first. (0, 3) if none pass.
    """
    return np.zeros((0, 3))  # comment this line and write your code for the function


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray,
                        iou_threshold: float = 0.3) -> list:
    """Drop boxes that overlap a better-scoring box.

    Keep the highest-scoring box, discard every remaining box whose IoU with
    it is MORE than iou_threshold (strictly more, so a box exactly at the
    threshold survives), and repeat with what is left.

    Parameters
    ----------
    boxes : np.ndarray
        Shape (N, 4), each row (x, y, w, h)
    scores : np.ndarray
        Shape (N,), higher is better
    iou_threshold : float
        Overlap above which a box is suppressed

    Returns
    -------
    list
        Indices of the kept boxes, best score first
    """
    return []  # comment this line and write your code for the function


def detect_multiscale(image: np.ndarray, template: np.ndarray,
                      scales: tuple = SCALES) -> list:
    """Search every scale, pool the hits, and thin them down.

    For each scale: resize the template with resize_template, skip the scale
    if the result no longer fits inside the image, score it with ncc_map, and
    take the peaks with find_peaks. Record each peak as a detection.

    Then run non_max_suppression across the detections from ALL scales
    together, and return the survivors sorted by decreasing score.

    A detection is a dict with exactly these keys:
        {"x": int, "y": int, "w": int, "h": int, "score": float, "scale": float}

    Parameters
    ----------
    image : np.ndarray
        Scene of shape (H, W)
    template : np.ndarray
        Template at its native size
    scales : tuple
        Multipliers to try

    Returns
    -------
    list
        Detections, best first. Possibly empty.
    """
    return []  # comment this line and write your code for the function


if __name__ == "__main__":

    template = load_image_as_rgb("imgs/waldo_template.png")
    scene_names = ["scene_easy", "scene_medium", "scene_rotated", "scene_hard"]

    print("{:<16}{:>9}{:>8}{:>8}".format("scene", "score", "scale", "IoU"))
    detections, scenes_rgb = [], []
    for name in scene_names:
        scene = load_image_as_rgb("imgs/{}.png".format(name))
        scenes_rgb.append(load_image_as_rgb("imgs/{}.png".format(name)))

        # #############################################
        # Comment these lines and write your code here
        found = []
        best = found[0] if found else None
        # #############################################

        detections.append(best)
        truth = load_ground_truth(name)
        overlap = iou((best["x"], best["y"], best["w"], best["h"]), truth) if best else 0.0
        print("{:<16}{:>9.3f}{:>8.2f}{:>8.3f}".format(
            name, best["score"] if best else 0.0,
            best["scale"] if best else 0.0, overlap))

    # One panel per scene: blue is the truth, green/red is your detection
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, name, rgb, best in zip(axes.ravel(), scene_names, scenes_rgb, detections):
        truth = load_ground_truth(name)
        shown = draw_box(rgb, truth, (0.1, 0.4, 1.0))
        if best is not None:
            box = (best["x"], best["y"], best["w"], best["h"])
            hit = iou(box, truth) >= 0.5
            shown = draw_box(shown, box, (0, 1, 0) if hit else (1, 0, 0))
        ax.imshow(shown)
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("plots/task2_results.png", metadata={"Author": getpass.getuser()})
    print("\nwrote plots/task2_results.png")
