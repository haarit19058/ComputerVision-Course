"""Small synthetic test images, shared by pytest and the notebook checks.  PROVIDED."""

import numpy as np

from provided.sampling import translate


def _box_blur(image, radius):
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)

    def blur_1d(row):
        return np.convolve(np.pad(row, radius, mode="edge"), kernel, mode="valid")

    image = np.apply_along_axis(blur_1d, 1, image)
    return np.apply_along_axis(blur_1d, 0, image)


def smooth_texture(h=64, w=64, seed=0):
    """Deterministic smooth random texture in [0, 1] (a stand-in for regolith)."""
    image = np.random.default_rng(seed).random((h, w))
    for _ in range(3):
        image = _box_blur(image, 2)
    return (image - image.min()) / (image.max() - image.min())


def part(label):
    """Mark a test as belonging to one subtask, e.g. @part("1.1"), so that the
    notebook's check("1.1") runs exactly the tests of that subtask."""
    def mark(test):
        test.part = label
        return test
    return mark


# the fixtures a test may ask for, by name (pytest provides the same ones)
FIXTURES = {"make_texture": lambda: smooth_texture, "shift": lambda: translate}
