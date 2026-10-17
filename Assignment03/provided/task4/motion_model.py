"""Task 4 · Reconstruct the Flight: the camera-motion fit.
PROVIDED: you do not need to change this file.

Between two frames of a camera looking straight down at flat ground, the
image moves by a similarity (equation 4.1 of the handout):

    x_{k+1} = s R(theta) x_k + t        with   R(theta) = [[cos, -sin], [sin, cos]]

with zoom s = Z_k / Z_{k+1}: the ratio of the two altitudes. This file fits
that similarity to your Task 1 to 3 tracks; read it to see how your tracks
become a camera motion, and to prepare your 4.2 answer.

Conventions
  * points are (N, 2) arrays of [x, y] in pixels; `centre` is the image
    centre [cx, cy], and x_k, x_{k+1} above are measured from it
  * a model is a dict {"zoom": s, "rotation": theta (radians),
    "shift": np.array([tx, ty]) (pixels)}
  * writing s R(theta) = [[p, -q], [q, p]] makes the model linear in
    m = [p, q, tx, ty], solved by np.linalg.lstsq
"""

import numpy as np


def similarity_system(points0, points1, centre):
    """The linear system A m ~= b for m = [p, q, tx, ty].

    Each track (x0, y0) -> (x1, y1), measured from `centre`, gives two rows:
        x1 = p x0 - q y0 + tx
        y1 = q x0 + p y0 + ty
    Returns A with shape (2N, 4) and b with shape (2N,).
    """
    a = np.asarray(points0, dtype=float) - centre
    b = np.asarray(points1, dtype=float) - centre
    n = len(a)
    A = np.zeros((2 * n, 4))
    rhs = np.zeros(2 * n)
    A[0::2] = np.c_[a[:, 0], -a[:, 1], np.ones(n), np.zeros(n)]     # x1 = p x0 - q y0 + tx
    A[1::2] = np.c_[a[:, 1], a[:, 0], np.zeros(n), np.ones(n)]      # y1 = q x0 + p y0 + ty
    rhs[0::2], rhs[1::2] = b[:, 0], b[:, 1]
    return A, rhs


def fit_similarity(points0, points1, centre):
    """Least-squares similarity from N >= 2 tracks.

    Solve A m ~= b, then zoom s = sqrt(p^2 + q^2), rotation = atan2(q, p).
    Returns a model dict (see the module docstring).
    """
    A, b = similarity_system(points0, points1, centre)
    (p, q, tx, ty), *_ = np.linalg.lstsq(A, b, rcond=None)
    return {"zoom": float(np.hypot(p, q)), "rotation": float(np.arctan2(q, p)),
            "shift": np.array([tx, ty])}


def apply_similarity(model, points, centre):
    """Where the model sends points of frame k in frame k+1. Returns (N, 2)."""
    c, s = np.cos(model["rotation"]), np.sin(model["rotation"])
    sR = model["zoom"] * np.array([[c, -s], [s, c]])
    return (np.asarray(points, dtype=float) - centre) @ sR.T + model["shift"] + centre


def robust_fit_similarity(points0, points1, centre, rounds=4, factor=3.0, min_threshold=1.0):
    """Fit, then repeatedly drop wrong tracks and refit.

    In each of `rounds` rounds: fit to the tracks still kept, compute every
    track's residual |apply_similarity(model, p0) - p1| in pixels, and keep
    the tracks whose residual is below max(factor * median residual of the
    kept tracks, min_threshold).

    Returns (model, inliers, residuals): the final model, an (N,) boolean
    array of kept tracks and the (N,) residuals under the final model.
    """
    p0, p1 = np.asarray(points0, dtype=float), np.asarray(points1, dtype=float)
    keep = np.ones(len(p0), dtype=bool)
    for _ in range(rounds):
        model = fit_similarity(p0[keep], p1[keep], centre)
        residuals = np.linalg.norm(apply_similarity(model, p0, centre) - p1, axis=1)
        threshold = max(factor * np.median(residuals[keep]), min_threshold)
        keep = residuals < threshold
        if keep.sum() < 3:                              # never fit to fewer than 3 tracks
            keep = residuals <= np.sort(residuals)[min(2, len(residuals) - 1)]
    model = fit_similarity(p0[keep], p1[keep], centre)
    residuals = np.linalg.norm(apply_similarity(model, p0, centre) - p1, axis=1)
    return model, keep, residuals
