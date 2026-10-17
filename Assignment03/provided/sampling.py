"""Bilinear sampling.  PROVIDED helper, used by the trackers (Tasks 2 and 3).

Tracking needs the image value at non-integer positions such as
(x, y) = (10.3, 7.8).
This function gives you that value; deciding *where* to sample is your job.
"""

import numpy as np


def bilinear_sample(image, x, y):
    """Sample image at real-valued pixel coordinates.

    image : (H, W) array
    x, y  : arrays of the same shape (x = column, y = row)

    Returns (values, valid):
      values : same shape as x, bilinearly interpolated intensities
      valid  : boolean array, False where (x, y) falls outside the image
               (those values are set to 0)
    """
    image = np.asarray(image, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h, w = image.shape

    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    xc = np.clip(x, 0, w - 1)
    yc = np.clip(y, 0, h - 1)

    x0 = np.floor(xc).astype(int)
    y0 = np.floor(yc).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    ax = xc - x0          # horizontal weight of the right-hand neighbour
    ay = yc - y0          # vertical weight of the lower neighbour

    top = (1 - ax) * image[y0, x0] + ax * image[y0, x1]
    bottom = (1 - ax) * image[y1, x0] + ax * image[y1, x1]
    values = (1 - ay) * top + ay * bottom
    return np.where(valid, values, 0.0), valid


def translate(image, dx, dy):
    """Shift image content by (dx, dy) pixels, bilinearly.  PROVIDED.

    new(x, y) = old(x - dx, y - dy). Useful for building a synthetic test or
    demo with a known ground-truth motion.
    """
    y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]].astype(float)
    values, _ = bilinear_sample(image, x - dx, y - dy)
    return values
