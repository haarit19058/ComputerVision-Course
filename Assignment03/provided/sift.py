"""SIFT features and matching.  PROVIDED: you do not need to change this file.

A plain NumPy version of Lowe's SIFT (IJCV 2004), simplified for reading:

  1. scale space   Gaussian blurs in octaves; differences of Gaussians (DoG)
  2. keypoints     DoG extrema, refined to sub-pixel; weak or edge-like ones dropped
  3. orientation   the dominant gradient direction around each keypoint
  4. descriptor    a 4 x 4 grid of 8-bin gradient histograms = 128 numbers
  5. matching      nearest neighbour with Lowe's ratio test

Why it is here: Lucas-Kanade (Tasks 2 and 3) can only follow small motions, because
it starts from a guess and improves it. SIFT matches do not care how far a
point moved. Use them to give Lucas-Kanade a good starting guess
(guess_displacements) when the camera moves a lot, as it does near
touchdown. Lucas-Kanade then refines each point to sub-pixel
accuracy.

Main functions:
    points, descriptors = detect_and_describe(image)
    pairs = match_descriptors(descriptors0, descriptors1)
    matches = sift_matches(image0, image1)                 # (points0, points1)
    initial = guess_displacements(points, *matches)
    tracked, valid = track_with_sift(track_points, image0, image1, points, matches)

track_with_sift runs your Task 3 track_points twice, from (0, 0) and from the
SIFT guesses, and keeps whichever result fits the image better for each
point. That protects points whose nearest matches move differently (for
example where the matches around a point are few or partly wrong).

Points are [x, y] in pixels of the input image (float arrays of shape (N, 2)).
"""

import hashlib
from collections import OrderedDict

import numpy as np

from provided.sampling import bilinear_sample

SIGMA0 = 1.6          # blur of the first image of every octave
INTERVALS = 3         # scales per octave that can hold keypoints
CONTRAST = 0.01       # a keypoint needs |DoG| >= CONTRAST / INTERVALS (images in [0, 1]);
                      # Lowe used 0.03 to 0.04, the low-contrast lunar surface needs less
EDGE_RATIO = 10.0     # larger principal-curvature ratios are edges, not corners
ORIENTATION_BINS = 36
GRID, ORIENTATIONS = 4, 8          # descriptor: GRID x GRID cells, ORIENTATIONS bins each
TWO_PI = 2 * np.pi


# 1. Scale space -------------------------------------------------------------

def gaussian_blur(image, sigma):
    """Separable Gaussian blur (edges repeated)."""
    radius = max(1, int(np.ceil(3 * sigma)))
    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-t ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    h, w = image.shape
    padded = np.pad(image, radius, mode="edge")
    rows = sum(k * padded[:, i:i + w] for i, k in enumerate(kernel))
    return sum(k * rows[i:i + h, :] for i, k in enumerate(kernel))


def build_scale_space(image):
    """Gaussian images and DoG images, one stack per octave."""
    base = gaussian_blur(np.asarray(image, dtype=np.float64),
                         np.sqrt(SIGMA0 ** 2 - 0.5 ** 2))    # assume the camera blur is 0.5
    n_octaves = max(1, int(np.log2(min(base.shape))) - 3)
    step = 2 ** (1 / INTERVALS)
    sigmas = [SIGMA0 * step ** i for i in range(INTERVALS + 3)]
    gaussians, dogs = [], []
    current = base
    for _ in range(n_octaves):
        octave = [current]
        for i in range(1, INTERVALS + 3):
            extra = np.sqrt(sigmas[i] ** 2 - sigmas[i - 1] ** 2)
            octave.append(gaussian_blur(octave[-1], extra))
        gaussians.append(octave)
        dogs.append(np.stack([b - a for a, b in zip(octave, octave[1:])]))
        current = octave[INTERVALS][::2, ::2]     # twice the blur, half the size
    return gaussians, dogs


# 2. Keypoints ---------------------------------------------------------------

def _extrema(dog, threshold):
    """Positions (layer, y, x) that beat all 26 neighbours in the DoG stack."""
    layers, h, w = dog.shape
    centre = dog[1:-1, 1:-1, 1:-1]
    high = np.full(centre.shape, -np.inf)
    low = np.full(centre.shape, np.inf)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                other = dog[1 + dz:layers - 1 + dz, 1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
                high = np.maximum(high, other)
                low = np.minimum(low, other)
    peak = ((centre > high) | (centre < low)) & (np.abs(centre) > 0.5 * threshold)
    s, y, x = np.nonzero(peak)
    return s + 1, y + 1, x + 1


def _refine(dog, s, y, x, threshold):
    """One Newton step to sub-pixel position; drop weak, unstable and edge points."""
    D = dog
    value = D[s, y, x]
    grad = np.stack([(D[s, y, x + 1] - D[s, y, x - 1]) / 2,
                     (D[s, y + 1, x] - D[s, y - 1, x]) / 2,
                     (D[s + 1, y, x] - D[s - 1, y, x]) / 2], axis=1)
    dxx = D[s, y, x + 1] + D[s, y, x - 1] - 2 * value
    dyy = D[s, y + 1, x] + D[s, y - 1, x] - 2 * value
    dss = D[s + 1, y, x] + D[s - 1, y, x] - 2 * value
    dxy = (D[s, y + 1, x + 1] - D[s, y + 1, x - 1] - D[s, y - 1, x + 1] + D[s, y - 1, x - 1]) / 4
    dxs = (D[s + 1, y, x + 1] - D[s + 1, y, x - 1] - D[s - 1, y, x + 1] + D[s - 1, y, x - 1]) / 4
    dys = (D[s + 1, y + 1, x] - D[s + 1, y - 1, x] - D[s - 1, y + 1, x] + D[s - 1, y - 1, x]) / 4
    hessian = np.stack([np.stack([dxx, dxy, dxs], -1),
                        np.stack([dxy, dyy, dys], -1),
                        np.stack([dxs, dys, dss], -1)], -2)
    solvable = np.abs(np.linalg.det(hessian)) > 1e-12
    offset = np.zeros_like(grad)
    offset[solvable] = -np.linalg.solve(hessian[solvable], grad[solvable][..., None])[..., 0]
    contrast = value + 0.5 * np.sum(grad * offset, axis=1)
    trace, det = dxx + dyy, dxx * dyy - dxy ** 2
    keep = (solvable & (np.abs(offset) < 0.6).all(axis=1) & (np.abs(contrast) >= threshold)
            & (det > 0) & (trace ** 2 * EDGE_RATIO < (EDGE_RATIO + 1) ** 2 * det))
    return keep, offset, np.abs(contrast)


def detect_keypoints(gaussians, dogs, max_keypoints):
    """Keypoints as (octave, layer, x, y, sigma) in octave coordinates, strongest first."""
    threshold = CONTRAST / INTERVALS
    found = []
    for octave, dog in enumerate(dogs):
        s, y, x = _extrema(dog, threshold)
        if len(s) == 0:
            continue
        keep, offset, strength = _refine(dog, s, y, x, threshold)
        for k in np.nonzero(keep)[0]:
            layer = s[k] + offset[k, 2]
            sigma = SIGMA0 * 2 ** (layer / INTERVALS)
            found.append((strength[k], octave, int(s[k]), x[k] + offset[k, 0],
                          y[k] + offset[k, 1], sigma))
    found.sort(key=lambda f: -f[0])
    return [f[1:] for f in found[:max_keypoints]]


# 3. Orientation ---------------------------------------------------------------

def _patch_gradients(image, x, y, radius):
    """Pixel grid around (x, y) with its central-difference gradients."""
    h, w = image.shape
    xi, yi = int(round(x)), int(round(y))
    ys, xs = np.mgrid[max(1, yi - radius):min(h - 2, yi + radius) + 1,
                      max(1, xi - radius):min(w - 2, xi + radius) + 1]
    gx = image[ys, xs + 1] - image[ys, xs - 1]
    gy = image[ys + 1, xs] - image[ys - 1, xs]
    return xs, ys, gx, gy


def orientations(image, x, y, sigma):
    """Dominant gradient directions (radians) around (x, y)."""
    window = 1.5 * sigma
    xs, ys, gx, gy = _patch_gradients(image, x, y, int(round(3 * window)))
    if xs.size == 0:
        return []
    weight = np.hypot(gx, gy) * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * window ** 2))
    bins = (np.arctan2(gy, gx) % TWO_PI * ORIENTATION_BINS / TWO_PI).astype(int)
    hist = np.bincount(bins.ravel() % ORIENTATION_BINS, weights=weight.ravel(),
                       minlength=ORIENTATION_BINS)
    for _ in range(2):                                      # light circular smoothing
        hist = (np.roll(hist, 1) + hist + np.roll(hist, -1)) / 3
    angles = []
    left, right = np.roll(hist, 1), np.roll(hist, -1)
    for i in np.nonzero((hist > left) & (hist > right) & (hist >= 0.8 * hist.max()))[0]:
        shift = 0.5 * (left[i] - right[i]) / (left[i] - 2 * hist[i] + right[i])
        angles.append(((i + 0.5 + shift) * TWO_PI / ORIENTATION_BINS) % TWO_PI)
    return angles


# 4. Descriptor ----------------------------------------------------------------

def descriptor(image, x, y, sigma, angle):
    """128-number SIFT descriptor, or None if the patch has no gradient."""
    cell = 3.0 * sigma                                       # width of one grid cell
    radius = int(round(cell * np.sqrt(2) * (GRID + 1) / 2))
    xs, ys, gx, gy = _patch_gradients(image, x, y, radius)
    cos, sin = np.cos(angle), np.sin(angle)
    # position of every pixel in the rotated keypoint frame, in cell units
    rx = (cos * (xs - x) + sin * (ys - y)) / cell
    ry = (-sin * (xs - x) + cos * (ys - y)) / cell
    col, row = rx + GRID / 2 - 0.5, ry + GRID / 2 - 0.5
    inside = (col > -1) & (col < GRID) & (row > -1) & (row < GRID)
    weight = np.hypot(gx, gy) * np.exp(-(rx ** 2 + ry ** 2) / (2 * (GRID / 2) ** 2))
    ori = (np.arctan2(gy, gx) - angle) % TWO_PI * ORIENTATIONS / TWO_PI
    row, col, ori, weight = row[inside], col[inside], ori[inside], weight[inside]

    # trilinear interpolation into (row, column, orientation) bins
    hist = np.zeros((GRID + 2, GRID + 2, ORIENTATIONS))
    r0, c0, o0 = np.floor(row).astype(int), np.floor(col).astype(int), np.floor(ori).astype(int)
    fr, fc, fo = row - r0, col - c0, ori - o0
    for dr, wr in ((0, 1 - fr), (1, fr)):
        for dc, wc in ((0, 1 - fc), (1, fc)):
            for do, wo in ((0, 1 - fo), (1, fo)):
                np.add.at(hist, (r0 + dr + 1, c0 + dc + 1, (o0 + do) % ORIENTATIONS),
                          weight * wr * wc * wo)
    vector = hist[1:-1, 1:-1].ravel()
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    vector = np.minimum(vector / norm, 0.2)                  # damp large gradients
    return vector / np.linalg.norm(vector)


_DESCRIBE_CACHE = OrderedDict()
_DESCRIBE_CACHE_LIMIT = 4     # more than the 2 overlapping frames of one pair


def _describe_uncached(image, max_keypoints):
    gaussians, dogs = build_scale_space(image)
    points, descriptors = [], []
    for octave, layer, x, y, sigma in detect_keypoints(gaussians, dogs, max_keypoints):
        blurred = gaussians[octave][layer]
        for angle in orientations(blurred, x, y, sigma):
            vector = descriptor(blurred, x, y, sigma, angle)
            if vector is not None:
                points.append([x * 2 ** octave, y * 2 ** octave])
                descriptors.append(vector)
    return (np.array(points, dtype=float).reshape(-1, 2),
            np.array(descriptors, dtype=float).reshape(-1, GRID * GRID * ORIENTATIONS))


def detect_and_describe(image, max_keypoints=2000):
    """SIFT keypoints of a grayscale image.

    Returns (points, descriptors): an (N, 2) array of [x, y] in pixels of the
    input image, and an (N, 128) array of unit-length descriptors. A keypoint
    with two strong orientations appears twice.

    Consecutive frame pairs share a frame (pair (a, a+1) and (a+1, a+2) both
    touch a+1), and run scripts process pairs in order, so the same image
    would otherwise be described twice. Results are cached by the image's own
    content (not its identity, which Python can reuse after the array is
    freed), keyed on a hash of its bytes plus max_keypoints.
    """
    key = (hashlib.blake2b(np.ascontiguousarray(image).view(np.uint8),
                          digest_size=16).digest(), image.shape, max_keypoints)
    cached = _DESCRIBE_CACHE.get(key)
    if cached is not None:
        _DESCRIBE_CACHE.move_to_end(key)
        return cached
    result = _describe_uncached(image, max_keypoints)
    _DESCRIBE_CACHE[key] = result
    if len(_DESCRIBE_CACHE) > _DESCRIBE_CACHE_LIMIT:
        _DESCRIBE_CACHE.popitem(last=False)
    return result


# 5. Matching ------------------------------------------------------------------

def match_descriptors(descriptors0, descriptors1, ratio=0.8, mutual=True):
    """Index pairs (i, j) of matching descriptors, shape (K, 2).

    A match is kept if its nearest neighbour is clearly closer than the second
    nearest (Lowe's ratio test) and, with mutual=True, if j's nearest
    neighbour is also i.
    """
    if len(descriptors0) == 0 or len(descriptors1) < 2:
        return np.zeros((0, 2), dtype=int)
    distance = np.sqrt(np.maximum(2 - 2 * descriptors0 @ descriptors1.T, 0))  # unit vectors
    rows = np.arange(len(descriptors0))
    two = np.argpartition(distance, 1, axis=1)[:, :2]
    swap = distance[rows, two[:, 0]] > distance[rows, two[:, 1]]
    two[swap] = two[swap][:, ::-1]
    best, second = two[:, 0], two[:, 1]
    keep = distance[rows, best] < ratio * distance[rows, second]
    if mutual:
        keep &= np.argmin(distance, axis=0)[best] == rows
    return np.stack([rows[keep], best[keep]], axis=1)


def sift_matches(image0, image1, ratio=0.8, max_keypoints=2000):
    """Matched SIFT points: (points0, points1), each (K, 2) of [x, y].

    Some matches are wrong; the median in guess_displacements and the
    better-fit check in track_with_sift take care of them.
    """
    p0, d0 = detect_and_describe(image0, max_keypoints)
    p1, d1 = detect_and_describe(image1, max_keypoints)
    pairs = match_descriptors(d0, d1, ratio)
    return p0[pairs[:, 0]], p1[pairs[:, 1]]


def guess_displacements(points, matches0, matches1, neighbours=8):
    """Starting (u, v) for each point: the median motion of its nearest matches.

    points             : (N, 2) points to track, [x, y]
    matches0, matches1 : matched SIFT points in the two images (sift_matches)
    Returns an (N, 2) array; all zeros if there are no matches. The median
    ignores the occasional wrong match.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(matches0) == 0:
        return np.zeros_like(points)
    motion = np.asarray(matches1, float) - np.asarray(matches0, float)
    k = min(neighbours, len(matches0))
    guesses = np.zeros_like(points)
    for start in range(0, len(points), 1024):                # chunks keep memory small
        chunk = points[start:start + 1024]
        d = np.linalg.norm(chunk[:, None, :] - np.asarray(matches0, float)[None], axis=2)
        nearest = np.argpartition(d, k - 1, axis=1)[:, :k]
        guesses[start:start + 1024] = np.median(motion[nearest], axis=1)
    return guesses


def window_error(image0, image1, points0, points1, radius=7):
    """Mean squared difference between the window around points0[i] in image0
    and the window around points1[i] in image1. Returns an (N,) array, with
    np.inf where a window leaves an image."""
    points0 = np.asarray(points0, dtype=float).reshape(-1, 2)
    points1 = np.asarray(points1, dtype=float).reshape(-1, 2)
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    v0, ok0 = bilinear_sample(image0, points0[:, 0, None, None] + dx, points0[:, 1, None, None] + dy)
    v1, ok1 = bilinear_sample(image1, points1[:, 0, None, None] + dx, points1[:, 1, None, None] + dy)
    error = ((v0 - v1) ** 2).mean(axis=(1, 2))
    return np.where(ok0.all(axis=(1, 2)) & ok1.all(axis=(1, 2)), error, np.inf)


def track_with_sift(track_points, image0, image1, points, matches, window_radius=7,
                    **options):
    """Track points with your track_points, helped by SIFT matches.

    track_points : your Task 3 function
    matches      : (points0, points1) from sift_matches
    options      : passed on to track_points (e.g. max_iterations)

    Every point is tracked twice, starting from (0, 0) and from its SIFT
    guess. The (0, 0) result is kept only if both runs succeed and its window
    matches better (window_error); otherwise the SIFT-guided result is used,
    so a guided track that leaves the image counts as failed rather than
    falling back to a wrong small motion. Returns (tracked, valid) like
    track_points.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    guess = guess_displacements(points, *matches)
    runs = [track_points(image0, image1, points, window_radius, initial=start, **options)
            for start in (None, guess)]
    (zero_tracked, zero_valid), (guided_tracked, guided_valid) = runs
    use_zero = (zero_valid & guided_valid
                & (window_error(image0, image1, points, zero_tracked, window_radius)
                   < window_error(image0, image1, points, guided_tracked, window_radius)))
    tracked = np.where(use_zero[:, None], zero_tracked, guided_tracked)
    valid = np.where(use_zero, zero_valid, guided_valid)
    return tracked, valid
