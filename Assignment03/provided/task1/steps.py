"""Task 1, the provided steps the notebook runs.  PROVIDED.

Each step takes your functions as arguments, shows its figure below the cell
and saves it (with any numbers) in outputs/task1/.
"""

import numpy as np

from provided.io import load_frame
from provided.plotting import plot_points, show_images
from provided.runner import FRAMES, attempt, save_json, task_output
from provided.sampling import translate

FRAME = 276
SCALE = 0.25


def load_task_frame(frame=FRAME, scale=SCALE):
    """Frame 276 of the descent at quarter size (432 x 588 pixels)."""
    image = load_frame(FRAMES, frame, scale)
    print(f"frame {frame}, scale {scale}, shape {image.shape}")
    return image


def show_gradients(image, spatial_gradients):
    """Task 1.1: Ix, Iy and the gradient magnitude of the frame."""
    grad = attempt(spatial_gradients, image)
    if grad is None:
        return None
    ix, iy = grad
    show_images({"I": image, "Ix": ix, "Iy": iy, "|gradient|": np.hypot(ix, iy)},
                task_output(1) / "gradients.png", title="Task 1.1: image gradients")
    return grad


def check_constraint(image, spatial_gradients, temporal_difference, u=0.3, v=-0.2):
    """Task 1.2: shift the frame by a known (u, v) and check Ix u + Iy v + It ~= 0."""
    grad = attempt(spatial_gradients, image)
    shifted = translate(image, u, v)
    it = attempt(temporal_difference, image, shifted)
    if grad is None or it is None:
        return None
    ix, iy = grad
    residual = ix * u + iy * v + it
    border = 8                           # away from the zero-gradient border
    interior_it = it[border:-border, border:-border]
    interior = residual[border:-border, border:-border]
    stats = {"true_u": u, "true_v": v,
             "mean_abs_residual": float(np.abs(interior).mean()),
             "max_abs_residual": float(np.abs(interior).max())}
    print(f"  motion constraint on a synthetic shift ({u}, {v}): mean |residual| "
          f"{stats['mean_abs_residual']:.4f}, max {stats['max_abs_residual']:.4f}")
    save_json(task_output(1) / "constraint_check.json", stats)
    # the border is cropped: there `translate` has nothing to sample from on one
    # side, so It is meaningless and would dominate the colour scale
    show_images({"It (interior)": interior_it, "Ix u + Iy v + It (interior)": interior},
                task_output(1) / "constraint.png",
                title="Task 1.2: the motion constraint (border cropped)")
    return stats


def show_features(image, select_features, max_points=200, window_radius=3, min_distance=10,
                  frame=FRAME):
    """Task 1.3: your selected points on the frame."""
    points = attempt(select_features, image, max_points, window_radius, min_distance)
    if points is None:
        return None
    plot_points(image, points, task_output(1) / "features.png",
                title=f"Task 1.3: trackable points, frame {frame}")
    return points
