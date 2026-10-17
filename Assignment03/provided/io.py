"""Image loading and saving.  PROVIDED: you do not need to change this file.

Conventions used everywhere in this project
-------------------------------------------
* An image is a float32 NumPy array of shape (H, W) with values in [0, 1].
* Index an image as image[y, x]  (row first, then column).
* Frames are identified by the number in their file name:
  "12.png", "frame_0012.png" and "img12.jpg" are all frame 12.
"""

from pathlib import Path
import re

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_grayscale(path, scale=1.0):
    """Load an image as float32 grayscale in [0, 1], shape (H, W).

    scale < 1 shrinks the image first (e.g. 0.25 = quarter size). This is only
    to make experiments faster; it is not part of any algorithm you implement.
    """
    with Image.open(path) as im:
        im = im.convert("L")
        if scale != 1.0:
            size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
            im = im.resize(size, Image.BOX)
        image = np.asarray(im, dtype=np.float32) / 255.0
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"{path}: expected a finite 2-D grayscale image")
    return image


def frame_index(path):
    """Frame number taken from a file name: 'frame_0012.png' -> 12."""
    numbers = re.findall(r"\d+", Path(path).stem)
    if not numbers:
        raise ValueError(f"no frame number in file name: {path}")
    return int(numbers[-1])


def list_frames(directory):
    """Return {frame number: path} for every image in a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"frame directory not found: {directory}")
    frames = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            frames[frame_index(path)] = path
    return dict(sorted(frames.items()))


def load_frame(directory, index, scale=1.0):
    """Load one frame by its number."""
    frames = list_frames(directory)
    if index not in frames:
        if not frames:
            raise FileNotFoundError(f"no frames in {directory}")
        nearest = sorted(sorted(frames, key=lambda k: abs(k - index))[:2])
        raise FileNotFoundError(f"frame {index} is not in {directory}; nearest frames "
                                f"you have: {nearest[0]}, {nearest[1]}"
                                if len(nearest) > 1 else
                                f"frame {index} is not in {directory}; you have {nearest[0]}")
    return load_grayscale(frames[index], scale)


def load_sequence(directory, scale=1.0):
    """Load every frame of a directory, in frame-number order."""
    return [load_grayscale(p, scale) for p in list_frames(directory).values()]


def save_image(path, image):
    """Save a float image (values clipped to [0, 1]) as an 8-bit PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    Image.fromarray(np.round(data * 255).astype(np.uint8)).save(path)
    return path


def save_tracks(path, tracks, **info):
    """Save {(frame0, frame1): (points0, points1)} plus metadata to an .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for (a, b), (p0, p1) in tracks.items():
        arrays[f"{a}_{b}_p0"] = np.asarray(p0, dtype=float)
        arrays[f"{a}_{b}_p1"] = np.asarray(p1, dtype=float)
    for key, value in info.items():
        arrays[f"info_{key}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)
    return path


def load_tracks(path):
    """Inverse of save_tracks: returns (tracks, info)."""
    data = np.load(path)
    tracks, info = {}, {}
    for key in data.files:
        if key.startswith("info_"):
            value = data[key]
            info[key[5:]] = value.item() if value.ndim == 0 else value
        elif key.endswith("_p0"):
            a, b, _ = key.split("_")
            tracks[(int(a), int(b))] = (data[key], data[f"{a}_{b}_p1"])
    return dict(sorted(tracks.items())), info
