# ES 666: Computer Vision

## Assignment 2 — Find Waldo

## The Spotter’s Problem

You are volunteering at the lost-and-found desk of a very large outdoor festival. Someone walks up, worried, and hands you a single clear photograph of the friend they came with: a striped shirt, a bobble hat, round glasses. Behind you, covering the whole wall, is the panorama the drone took over the main field ten minutes ago. Two hundred people in it. Several of them are wearing stripes.

Your first instinct is the obvious one. You have a picture of the friend and you have a picture of the crowd, so slide the small picture over the big one and look for the place where the pixels agree best. This works, and it is worth understanding exactly why it works before you find out where it stops working.

Because it does stop. The friend in the panorama is not at the same size as in the photograph, the light on the field is not the light in the photograph, and at some point they turn their head or lean sideways and the whole comparison falls apart. When that happens you need a different idea: forget whole-image agreement, find a handful of small distinctive spots, describe each one in a way that survives being rotated and resized, match them, and let the matches vote on where the answer is.

Both ideas are in this assignment. Neither of them is a neural network, and neither of them is a library call. You will build both from NumPy and compare them honestly.

You are given a small template of Waldo and four crowded scenes. Your job is to find him twice, using two completely different ideas, and then to say which one worked, where it failed, and why.

- **Tasks 1–2** build a *correlation* detector. It slides the template over the scene and asks “how well does the template match here?” at every position and every size.
- **Tasks 3–4** build a *feature* detector. It finds a few dozen distinctive points in the template and in the scene, describes each one, matches them, and lets the matches vote on where Waldo is.

The two behave very differently, and the last part of the assignment is about that difference. Neither one solves every scene — that is the point, not a bug in the assignment.

## Ground rules

1. NumPy only for the algorithms. No `cv2.filter2D`, `cv2.matchTemplate`, `cv2.SIFT`, `cv2.goodFeaturesToTrack`, `scipy.ndimage`, `skimage`, or anything similar. The provided helpers use OpenCV for reading and resizing images only, and that is fine — resampling is not what this assignment teaches.
2. No deep learning. No PyTorch, no TensorFlow, no pretrained anything.
3. Do not touch the helper functions or the imports. They are marked in the files. Fill in only the functions whose body reads `return ... # comment this line` and write your code for the function.

---

# Task 1 — Filtering, and knowing whether it helped

The scenes are noisy. Before you can match anything you need to clean them up, and before you can claim you cleaned them up you need a number that says so.

## Functions to implement

### 1.1 `convolve2d(image, kernel)`

True convolution: flip the kernel on both axes, pad by half the kernel with `mode="reflect"`, and slide. The output is the same shape as the input. Loop over the `kernel`, not over the pixels — a 9 × 9 kernel is 81 whole-array operations, while looping pixels is hundreds of thousands of Python iterations.

The input may be `(H, W)` or `(H, W, 3)` and one code path must handle both. Pad only the two spatial axes and let arithmetic broadcasts over colour by itself. Padding the channel axis as well is the most common bug in this function.

### 1.2 `gaussian_kernel(size, sigma)`

Sample the Gaussian on a centred grid and normalise by the *sum of the samples*, not by the analytic constant `1/(2πσ²)`. Be ready to say why those differ on a finite grid.

### 1.3 `median_filter(image, size)`

The median of each window. Not a convolution — no kernel exists that computes a median.

On colour, take each channel’s median separately; question 1.10 is about what that costs you.

### 1.4 `sobel_edges(image)`

Returns `gx`, `gy`, `magnitude`, `orientation`. Orientation in degrees in `[0, 360)`. All four outputs are 2-D even when the input is RGB, because the gradient is taken on `to_grayscale(image)`.

### 1.5 `psnr(reference, test)`

PSNR = `10 log10(1/MSE)` for images in `[0, 1]`. Identical images give `∞`.

### 1.6 `ssim(reference, test)`

Structural similarity, using a Gaussian window for the local means, variances and covariance, with `C1 = 0.01²` and `C2 = 0.03²`. Return the mean score and the full map.

## Written

### 1.7 Predict first, then measure

Before you run anything, write down two predictions: will a 9 × 9 box filter make PSNR better or worse on the Gaussian-noised image, and will it make SSIM better or worse? Commit to both in writing.

Now run `task1.py` and report the four numbers from the first row of your table to two decimal places. State plainly whether each prediction was right. A wrong prediction honestly reported and then explained earns full marks; a prediction that suspiciously matches the measurement earns none.

### 1.8 The two metrics disagree

One of those two numbers goes up while the other goes down. Explain how one image can be simultaneously further from the original and more similar to it. Which would you trust if the next step is template matching, and why?

### 1.9 What colour costs you

Re-run the whole table on `to_grayscale(clean)` so every filter and both metrics see a single channel. Paste both tables and the lines you changed. PSNR and SSIM barely move — explain why that is expected, and why it does *not* mean the colour is useless. (Task 2 is where the answer shows up.)

### 1.10 Padding

Switch `convolve2d` to zero padding, blur with a large kernel and look at the border. Describe the artefact and say why reflect is the safer default before a template match.

---

# Task 2 — Finding Waldo by sliding the template

The direct approach. Put the template at every position, score the match, and take the best. Doing that naively is far too slow, so most of this task is about making it fast enough to actually run.

## Functions to implement

### 2.1 `integral_image(image)`

A summed-area table with a zero first row and column. Two cumulative sums.

### 2.2 `box_sum(ii, y0, x0, h, w)`

The sum over any box in four lookups, regardless of box size. `y0` and `x0` are arrays, so this does all windows at once.

### 2.3 `ncc_map(image, template)`

Zero-mean normalised cross-correlation at every position:

$$
\mathrm{NCC}
=
\frac{\sum (I-\bar I)(T-\bar T)}
{\sqrt{\sum (I-\bar I)^2}\sqrt{\sum (T-\bar T)^2}}
$$

### 2.4 `find_peaks(score_map, threshold, min_distance, max_peaks)`

Strongest first, keeping a peak only if it is at least `min_distance` from every peak already kept, in Chebyshev distance.

### 2.5 `non_max_suppression(boxes, scores, iou_threshold)`

Standard greedy NMS. A box exactly at the threshold survives.

### 2.6 `detect_multiscale(image, template, scales)`

Score at every scale, pool all the peaks, then suppress across scales together — not scale by scale.

## Written

### 2.7 Why NCC and not sum-of-squared-differences?

Show algebraically that NCC is unchanged by `I → aI + b` for `a > 0`, and say what SSD does under the same change.

### 2.8 Grayscale against colour, measured

Re-run the task with `to_grayscale` applied to both the scene and the template. Report the two tables side by side: which scenes still work, and how much the winning score moves. Relate the size of that change to Waldo’s actual colours. Paste the lines you changed.

### 2.9 The mask you were given and never used

`imgs/waldo_mask.png` ships with this assignment and no task touches it. It marks which pixels inside the template rectangle are actually Waldo rather than background, which matters because `scene_hard` has an occluder over part of him.

Write a masked version of `ncc_map` that correlates only over pixels where the mask is set (centre each channel by its masked mean, and normalise by the masked norms). Report the best score and its scale on `scene_hard`, with and without the mask.

Do not predict the answer from theory — measure it, and report what you actually got even if it surprises you. Then explain the direction of the change in terms of what the unmasked correlation is forced to include.

### 2.10 Scores are not comparable across scales

A smaller template has fewer pixels, so its NCC has a wider spread and its maximum over a whole scene is higher by chance alone. For `scene_hard`, print `.max()` and `.std()` of the score map at every scale, along with the template’s pixel count, and paste the table.

Look at the spread across scales before you answer this: would dividing each map by its own standard deviation make the scales comparable? Justify your answer from your own numbers, not from what standardisation usually does.

---

# Task 3 — Corners, blobs and descriptors

A different idea. Instead of comparing whole rectangles, find a few dozen *distinctive points*, and describe the neighbourhood of each one in a way that survives the image being resized or rotated.

## Functions to implement

### 3.1 `harris_response(image, k, sigma)`

Build the structure tensor from the Sobel gradients you wrote in Task 1, smooth its three entries with a Gaussian, and return:

$$
\det(J) - k\,\mathrm{trace}(J)^2
$$

### 3.2 `detect_corners(response, threshold_ratio, min_distance)`

Threshold relative to the maximum response, then keep the strongest well-separated ones.

### 3.3 `dog_keypoints(image, sigma0, n_scales, ratio, ...)`

Build a Gaussian ladder, difference adjacent levels, and keep points that are extremal among their 26 neighbours in the `3 × 3 × 3` space-scale cube.

Returns `(y, x, σ)`: unlike a Harris corner, a blob has a size, and that size is what makes Task 4 work.

### 3.4 `dominant_orientation(image, y, x, sigma)`

A gradient-orientation histogram over a Gaussian-weighted patch, scaled to `σ`. Return the peak bin, in degrees.

### 3.5 `describe_keypoints(image, keypoints, orientations)`

A 4 × 4 grid of 8-bin orientation histograms, sampled in a frame rotated by the keypoint’s orientation and scaled by its `σ` — 128 numbers per keypoint. Normalise each descriptor to unit length, except that an all-zero descriptor stays all-zero.

## Written

### 3.6 Corner or blob?

A Harris corner is `(y, x)`; a DoG keypoint is `(y, x, σ)`. Explain where that third number comes from, and why Task 4 could not work with Harris corners alone.

### 3.7 The ladder is a trade

Run `dog_keypoints` on the template with at least three different `(sigma0, n_scales, k)` settings. Report the keypoint count and the `σ` range for each, and say what breaks at both ends of the ladder.

### 3.8 More keypoints, fewer matches

`find_waldo` enlarges the template by `upsample=2.0` before detecting. Bigger ought to be better: more pixels, more keypoints, more chances to match.

Test it. Run `find_waldo` on `scene_easy` with `upsample` set to `1.5`, `2.0`, `3.0` and `4.0`, and for each one report two numbers: how many keypoints were found on the template, and how many matches survived (`n_matches`). Paste the loop you wrote.

The trend in the second column is not the one you would guess. State what it is, then explain it. Your explanation must refer to a specific visual property of Waldo’s shirt and to the exact condition inside `match_ratio_test` that this property triggers. An answer that only says “more keypoints means more noise” has not understood it.

### 3.9 Why does this task throw the colour away?

Tasks 1 and 2 work in RGB; this one reduces to luminance. Explain what goes wrong if you try to define a gradient *orientation* on a three-channel image directly, using a red-to-green edge as your example.

---

# Task 4 — Matching, and letting the matches vote

You have descriptors on the template and on the scene. Now match them — and deal with the fact that most of the matches will be wrong.

## Functions to implement

### 4.1 `descriptor_distances(desc_a, desc_b)`

All-pairs Euclidean distance from one matrix product, using

$$
\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2a\cdot b.
$$

No double loop. Clip at zero before the square root.

### 4.2 `match_ratio_test(desc_scene, desc_template, ratio)`

Lowe’s ratio test — but read the docstring before you write it, because the **DIRECTION is the whole point** and it is the opposite of the obvious one. Each scene keypoint asks which of the ~80 template parts it resembles, not the other way round, and there is deliberately no mutual-nearest-neighbour check.

### 4.3 `predict_centres(...)`

The heart of the task. A match pairs two keypoints that each carry a position, a scale and an orientation — so a single match determines the size change

$$
s=\sigma_s/\sigma_t,
$$

the rotation

$$
\Delta\theta=\theta_s-\theta_t,
$$

and therefore a full guess at where the centre of Waldo is:

$$
(o_x,o_y)=\left(\frac{W_t}{2}-x_t,\frac{H_t}{2}-y_t\right)
$$

$$
c_x=x_s+s\left(\cos\Delta\theta\cdot o_x-\sin\Delta\theta\cdot o_y\right)
$$

$$
c_y=y_s+s\left(\sin\Delta\theta\cdot o_x+\cos\Delta\theta\cdot o_y\right)
$$

Vectorise over all matches. A wrong match will place its guess anywhere at all, possibly outside the image. That is expected.

### 4.4 `consensus_centre(predictions, tol, min_votes)`

Find the *densest cluster* of guesses: count how many guesses lie within `tol` of each guess, take the winner as a seed, and keep everything within `tol` of it. Report the median of that cluster, or `None` if it has fewer than `min_votes` members.

### 4.5 `find_waldo(scene, template, upsample)`

Wire it together and return a box, or `None` when there is no consensus. Do not invent a box.

## Written

### 4.6 Why the median fails

Only a small fraction of the matches are correct. Replace the densest-cluster step with a plain median over *all* the predictions, report the IoU you get on each scene, and explain the result in terms of what fraction of the data has to be correct for a median to mean anything.

### 4.7 The abnormality: each method fails where the other does not.

#### (a)

Name the exact thing missing from Task 2’s search space, and prove your claim by explaining why adding twenty more entries to `SCALES` cannot help.

#### (b)

Point to the specific step that buys Task 4, and say what would happen on `scene_rotated` if you passed `orientations=None` to `describe_keypoints`.

#### (c)

Explain why the correlation method is the less precise one. Mention what Task 2 returns exactly and Task 4 only estimates.

### 4.8 The check that every tutorial recommends

`match_ratio_test` deliberately omits a mutual-nearest-neighbour test: it does not require that a scene keypoint’s best template partner also picks that scene keypoint back. Almost every published description of feature matching includes one.

Add it — keep a match only when

```python
np.argmin(distances, axis=0)[j1] == rows
```

— and re-run all four scenes. Report `n_matches` and `n_votes` for each, with and without. Paste the line you added.

Report honestly what happened, then explain it. Your answer must say something about how many scene keypoints can legitimately correspond to the *same template part*, and why that makes a one-to-one pairing the wrong model for finding an object in a scene.

### 4.9 Agreement without an answer key

The last column is the IoU between your Task 2 box and your Task 4 box — no ground truth involved. Explain why disagreement is informative even though it cannot tell you *which* detector is wrong, and construct a concrete case where two independent detectors would agree and both be wrong.

---

# What to submit

A single archive named `<rollnumber>_assignment2.zip` containing:

- `task1.py`, `task2.py`, `task3.py`, `task4.py`
- `subjective_answers_task1.md` … `subjective_answers_task4.md`
- your `plots/` folder

Do not include the virtual environment or the `imgs/` folder.












# Instructions
1) First read this and be prepared, when i ask for task i then only give response
1) I need you to edit the task{i}.py files such that all the necessary results for subjective answers are also covered.
2) I need you to first read the instructions properly 
3) Keep the subjective answers humanized - dont use extensive markdown formatting and make it sound natural - search the web for how to write humanized text
4) Keep the base code changes minimal make sure to inlcude full comments and code structure 
5) make changes on top of the exisiting code with minimal changes


# Task 01 base code
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


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
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

    padded = np.pad(image, pad_width, mode="reflect")
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
    print("\nwrote plots/task1_results.png and plots/task1_edges.png")




# task 02 base Code
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
    ii = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(image, axis=0), axis=1)
    return ii


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
    return ii[y0 + h, x0 + w] - ii[y0, x0 + w] - ii[y0 + h, x0] + ii[y0, x0]


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
    H, W, C = image.shape
    th, tw, _ = template.shape
    
    t_mean = np.mean(template)
    t_c = template - t_mean
    t_norm = np.sqrt(np.sum(t_c ** 2))
    
    out_h, out_w = H - th + 1, W - tw + 1
    numerator = np.zeros((out_h, out_w), dtype=np.float64)
    sum_I = np.zeros((out_h, out_w), dtype=np.float64)
    sum_I2 = np.zeros((out_h, out_w), dtype=np.float64)
    
    y0 = np.arange(out_h)[:, None]
    x0 = np.arange(out_w)[None, :]
    
    for c in range(C):
        img_c = image[:, :, c]
        views = sliding_windows(img_c, th, tw)
        
        # Cross-correlation via einsum as suggested
        numerator += np.einsum('ijkl,kl->ij', views, t_c[:, :, c])
        
        # Local stats via integral images for O(1) complexity per window
        ii = integral_image(img_c)
        ii2 = integral_image(img_c ** 2)
        
        sum_I += box_sum(ii, y0, x0, th, tw)
        sum_I2 += box_sum(ii2, y0, x0, th, tw)
        
    N = th * tw * C
    mean_I = sum_I / N
    
    var_I = sum_I2 - N * (mean_I ** 2)
    var_I = np.maximum(var_I, 0.0) 
    i_norm = np.sqrt(var_I)
    
    denom = i_norm * t_norm
    ncc = np.zeros_like(numerator)
    mask = denom > 1e-8
    ncc[mask] = numerator[mask] / denom[mask]
    
    return ncc


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
    ys, xs = np.nonzero(score_map >= threshold)
    scores = score_map[ys, xs]
    
    if len(scores) == 0:
        return np.zeros((0, 3))
        
    order = np.argsort(scores)[::-1]
    ys = ys[order]
    xs = xs[order]
    scores = scores[order]
    
    kept = []
    for y, x, s in zip(ys, xs, scores):
        if len(kept) == max_peaks:
            break
            
        conflict = False
        for ky, kx, _ in kept:
            if max(abs(y - ky), abs(x - kx)) < min_distance:
                conflict = True
                break
                
        if not conflict:
            kept.append((y, x, s))
            
    if not kept:
        return np.zeros((0, 3))
        
    return np.array(kept, dtype=np.float64)


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
    if len(boxes) == 0:
        return []
        
    order = np.argsort(scores)[::-1]
    kept_indices = []
    
    while order.size > 0:
        i = order[0]
        kept_indices.append(i)
        
        if order.size == 1:
            break
            
        rest_indices = order[1:]
        box_a = tuple(boxes[i])
        
        ious = np.array([iou(box_a, tuple(boxes[j])) for j in rest_indices])
        order = rest_indices[ious <= iou_threshold]
        
    return kept_indices


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
    H, W = image.shape[:2]
    all_detections = []
    all_boxes = []
    all_scores = []
    
    for scale in scales:
        scaled_template = resize_template(template, scale)
        th, tw = scaled_template.shape[:2]
        
        if th > H or tw > W:
            continue
            
        scores = ncc_map(image, scaled_template)
        peaks = find_peaks(scores, threshold=0.5) 
        
        for y, x, score in peaks:
            all_boxes.append([x, y, tw, th])
            all_scores.append(score)
            all_detections.append({
                "x": int(x), "y": int(y), "w": int(tw), "h": int(th),
                "score": float(score), "scale": float(scale)
            })
            
    if not all_detections:
        return []
        
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    
    kept_indices = non_max_suppression(all_boxes, all_scores, iou_threshold=0.3)
    
    return [all_detections[i] for i in kept_indices]


if __name__ == "__main__":

    template = load_image_as_rgb("imgs/waldo_template.png")
    scene_names = ["scene_easy", "scene_medium", "scene_rotated", "scene_hard"]

    print("{:<16}{:>9}{:>8}{:>8}".format("scene", "score", "scale", "IoU"))
    detections, scenes_rgb = [], []
    for name in scene_names:
        scene = load_image_as_rgb("imgs/{}.png".format(name))
        scenes_rgb.append(load_image_as_rgb("imgs/{}.png".format(name)))

        found = detect_multiscale(scene, template)
        best = found[0] if found else None

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


# Task 03 base code

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


# Task 04

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
from task3 import *
# %%
# Helper Functions
# ================================================================
# These are helper functions that you can use in your implementation.
# Don't update these functions.

os.makedirs("plots", exist_ok=True)


def with_unit_scale(keypoints: np.ndarray) -> np.ndarray:
    """Give scale-less keypoints a sigma of 1, so (N, 2) and (N, 3) both work.

    Harris corners are (y, x): a corner has no size of its own. Padding them
    to (y, x, 1) lets the same matching code take either detector -- at the
    cost of the scale ratio then carrying no information at all.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    kp = kp.reshape(-1, kp.shape[-1]) if kp.size else kp.reshape(0, 3)
    if kp.shape[1] >= 3:
        return kp
    return np.concatenate([kp, np.ones((kp.shape[0], 1))], axis=1)


def box_from_centre(cx: float, cy: float, scale: float,
                    template_shape: tuple) -> tuple:
    """A box the shape of the template, scaled and centred on (cx, cy)."""
    th, tw = template_shape
    w = max(1, int(round(tw * scale)))
    h = max(1, int(round(th * scale)))
    return (int(np.floor(cx - w / 2.0)), int(np.floor(cy - h / 2.0)), w, h)


def draw_matches(template: np.ndarray, scene_rgb: np.ndarray,
                 kp_t: np.ndarray, kp_s: np.ndarray,
                 matches: np.ndarray, votes: np.ndarray, path: str) -> None:
    """Template and scene side by side, with a line for every match."""
    zoom = 4
    left = np.repeat(np.repeat(np.dstack([template] * 3), zoom, 0), zoom, 1)
    height = max(left.shape[0], scene_rgb.shape[0])
    canvas = np.ones((height, left.shape[1] + scene_rgb.shape[1], 3))
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:scene_rgb.shape[0], left.shape[1]:] = scene_rgb

    plt.figure(figsize=(14, 7))
    plt.imshow(canvas.astype(np.uint8))
    for (a, b), voted in zip(matches, votes):
        plt.plot([kp_t[a, 1] * zoom, kp_s[b, 1] + left.shape[1]],
                 [kp_t[a, 0] * zoom, kp_s[b, 0]],
                 color="lime" if voted else "red", linewidth=0.7, alpha=0.85)
    plt.title("green = agreed with the consensus, red = outvoted")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, metadata={"Author": getpass.getuser()})
    plt.close()


# ================================================================
# %%
# Task 4
# ======


def descriptor_distances(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """Distance from every descriptor in A to every descriptor in B.

    Parameters
    ----------
    desc_a : np.ndarray
        Shape (Na, D)
    desc_b : np.ndarray
        Shape (Nb, D)

    Returns
    -------
    np.ndarray
        Shape (Na, Nb) of Euclidean distances
    """
    if len(desc_a) == 0 or len(desc_b) == 0:
        return np.zeros((len(desc_a), len(desc_b)))

    norm_a = np.sum(desc_a ** 2, axis=1, keepdims=True)
    norm_b = np.sum(desc_b ** 2, axis=1, keepdims=True)
    cross_term = np.dot(desc_a, desc_b.T)
    
    dist_sq = norm_a + norm_b.T - 2 * cross_term
    dist_sq = np.clip(dist_sq, 0.0, None)
    
    return np.sqrt(dist_sq)


def match_ratio_test(desc_scene: np.ndarray, desc_template: np.ndarray,
                     ratio: float = 0.8) -> np.ndarray:
    """Ask every SCENE keypoint which template part it looks like.

    Parameters
    ----------
    desc_scene : np.ndarray
        Shape (Ns, D), the queries
    desc_template : np.ndarray
        Shape (Nt, D), the small database
    ratio : float
        Lowe's threshold

    Returns
    -------
    np.ndarray
        Shape (M, 2) of (template index, scene index), best match first.
        (0, 2) if nothing survives. Note the column order: template first,
        which is what predict_centres expects.
    """
    if len(desc_scene) == 0 or len(desc_template) == 0:
        return np.zeros((0, 2), dtype=int)

    dists = descriptor_distances(desc_scene, desc_template)
    sorted_indices = np.argsort(dists, axis=1)
    
    best_indices = sorted_indices[:, 0]
    second_indices = sorted_indices[:, 1]
    
    best_dists = dists[np.arange(len(desc_scene)), best_indices]
    second_dists = dists[np.arange(len(desc_scene)), second_indices]
    
    valid_mask = best_dists < (ratio * second_dists)
    
    scene_matches = np.where(valid_mask)[0]
    template_matches = best_indices[valid_mask]
    
    return np.column_stack((template_matches, scene_matches))


def predict_centres(kp_template: np.ndarray, kp_scene: np.ndarray,
                    matches: np.ndarray, ori_template: np.ndarray,
                    ori_scene: np.ndarray, template_shape: tuple) -> np.ndarray:
    """Turn every single match into a full guess at where Waldo is.

    This is the idea the whole task turns on. A matched keypoint carries a
    scale and an orientation as well as a position, so comparing them gives
    the size change and the rotation -- and once you know those, ONE match is
    enough to point at the centre of the whole object.

    For a match pairing template keypoint (y_t, x_t, sigma_t) at angle
    theta_t with scene keypoint (y_s, x_s, sigma_s) at theta_s, and template
    shape (H_t, W_t):

        s      = sigma_s / sigma_t                     the size change
        dtheta = theta_s - theta_t                     the rotation, degrees
        ox, oy = W_t / 2 - x_t,  H_t / 2 - y_t         offset to the centre
        cx     = x_s + s * (cos(dtheta) * ox - sin(dtheta) * oy)
        cy     = y_s + s * (sin(dtheta) * ox + cos(dtheta) * oy)

    A correct match puts (cx, cy) on Waldo. A wrong one puts it anywhere at
    all, possibly off the image. That is expected, and Task 4.4 cleans it up.

    Parameters
    ----------
    kp_template, kp_scene : np.ndarray
        Shapes (Nt, 3) and (Ns, 3), columns (y, x, sigma)
    matches : np.ndarray
        Shape (M, 2) from match_ratio_test
    ori_template, ori_scene : np.ndarray
        Orientations in degrees, one per keypoint
    template_shape : tuple
        (H_t, W_t) of the template these keypoints came from

    Returns
    -------
    np.ndarray
        Shape (M, 3), columns (cx, cy, s), in scene pixels. (0, 3) if no
        matches. Vectorise it -- no Python loop over matches.
    """
    if len(matches) == 0:
        return np.zeros((0, 3))

    t_idx = matches[:, 0]
    s_idx = matches[:, 1]
    
    yt, xt, sigmat = kp_template[t_idx].T
    ys, xs, sigmas = kp_scene[s_idx].T
    
    thetat = ori_template[t_idx]
    thetas = ori_scene[s_idx]
    
    s = sigmas / sigmat
    dtheta = np.radians(thetas - thetat)
    
    Ht, Wt = template_shape
    ox = (Wt / 2.0) - xt
    oy = (Ht / 2.0) - yt
    
    cos_d = np.cos(dtheta)
    sin_d = np.sin(dtheta)
    
    cx = xs + s * (cos_d * ox - sin_d * oy)
    cy = ys + s * (sin_d * ox + cos_d * oy)
    
    return np.column_stack((cx, cy, s))


def consensus_centre(predictions: np.ndarray, tol: float = 12.0,
                     min_votes: int = 8) -> tuple:
    """Find where the guesses pile up -- the DENSEST cluster, not the average.

    Only about one match in eight is correct here. Any method that assumes
    the good ones are a majority will fail, and that includes taking the
    median: the median of all the predictions lands in the middle of the
    scattered wrong ones and points at nothing. Measure that for yourself,
    it is one of the questions.

    Instead:
      1. Compute the distance from every prediction to every other one.
      2. Count, for each prediction, how many lie strictly within tol of it.
      3. The one with the highest count is the seed.
      4. The consensus is every prediction strictly within tol of that seed.
      5. Fewer than min_votes in it means no detection.
      6. Otherwise report the MEDIAN centre and MEDIAN scale of that cluster.

    Parameters
    ----------
    predictions : np.ndarray
        Shape (M, 3) from predict_centres
    tol : float
        How close two guesses must be to count as agreeing, in pixels
    min_votes : int
        Fewest agreeing guesses that counts as a detection

    Returns
    -------
    tuple
        centre (cx, cy) or None, scale (float) or None, and a boolean mask of
        shape (M,) marking which predictions joined the consensus
    """
    M = len(predictions)
    if M < min_votes:
        return None, None, np.zeros(M, dtype=bool)

    coords = predictions[:, :2]
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    
    vote_counts = np.sum(dists <= tol, axis=1)
    best_seed = np.argmax(vote_counts)
    
    consensus_mask = dists[best_seed] <= tol
    
    if np.sum(consensus_mask) < min_votes:
        return None, None, np.zeros(M, dtype=bool)
        
    cluster = predictions[consensus_mask]
    med_cx, med_cy = np.median(cluster[:, :2], axis=0)
    med_scale = np.median(cluster[:, 2])
    
    return (med_cx, med_cy), med_scale, consensus_mask


def find_waldo(scene: np.ndarray, template: np.ndarray,
               upsample: float = 2.0) -> dict:
    """Put it all together: detect, describe, match, vote.

    Parameters
    ----------
    scene : np.ndarray
        Grayscale scene
    template : np.ndarray
        Grayscale template at its native size
    upsample : float
        How much to enlarge the template before detecting

    Returns
    -------
    dict
        Keys: "box" ((x, y, w, h) or None), "n_matches", "n_votes",
        "matches", "votes", "kp_template", "kp_scene". Return None for the
        box when there is no consensus -- do not invent one.
    """
    empty_result = {
        "box": None, "n_matches": 0, "n_votes": 0,
        "matches": np.zeros((0, 2), int),
        "votes": np.zeros(0, bool),
        "kp_template": np.zeros((0, 3)),
        "kp_scene": np.zeros((0, 3))
    }

    if upsample != 1.0:
        th, tw = template.shape
        template_proc = cv2.resize(template, (int(tw * upsample), int(th * upsample)), interpolation=cv2.INTER_CUBIC)
    else:
        template_proc = template

    kp_t = dog_keypoints(template_proc)
    if len(kp_t) == 0:
        return empty_result

    ori_t = np.array([dominant_orientation(template_proc, y, x, s) for y, x, s in kp_t])
    desc_t = describe_keypoints(template_proc, kp_t, ori_t)

    kp_s = dog_keypoints(scene)
    if len(kp_s) == 0:
        empty_result["kp_template"] = kp_t
        return empty_result

    ori_s = np.array([dominant_orientation(scene, y, x, s) for y, x, s in kp_s])
    desc_s = describe_keypoints(scene, kp_s, ori_s)

    matches = match_ratio_test(desc_s, desc_t)
    if len(matches) == 0:
        empty_result["kp_template"] = kp_t
        empty_result["kp_scene"] = kp_s
        return empty_result

    preds = predict_centres(kp_t, kp_s, matches, ori_t, ori_s, template_proc.shape)
    centre, scale, votes = consensus_centre(preds)

    box = None
    if centre is not None:
        box = box_from_centre(centre[0], centre[1], scale, template_proc.shape)

    return {
        "box": box,
        "n_matches": len(matches),
        "n_votes": int(np.sum(votes)),
        "matches": matches,
        "votes": votes,
        "kp_template": kp_t,
        "kp_scene": kp_s
    }


if __name__ == "__main__":

    # The two pipelines want different inputs, and the difference is the
    # point. Correlation (Task 2) keeps all three channels, because Waldo's
    # red is most of what identifies him. The feature pipeline (Task 3)
    # needs a scalar field to take gradients of, so it gets luminance.
    template_rgb = load_image_as_rgb("imgs/waldo_template.png")
    template = to_grayscale(template_rgb)
    scene_names = ["scene_easy", "scene_medium", "scene_rotated", "scene_hard"]

    print("{:<16}{:>9}{:>8}{:>9}{:>10}".format(
        "scene", "matches", "votes", "IoU", "agree?"))
    for name in scene_names:
        scene_rgb = load_image_as_rgb("imgs/{}.png".format(name))
        scene = to_grayscale(scene_rgb)
        truth = load_ground_truth(name)

        # ###############################################
        # Comment these lines and write your code here
        found = find_waldo(scene, template)
        correlation = detect_multiscale(scene_rgb, template_rgb)
        # ###############################################

        feature_box = found["box"]
        corr_box = None
        if correlation:
            best = correlation[0]
            corr_box = (best["x"], best["y"], best["w"], best["h"])

        # Scored against the truth here, for your report only. Notice that
        # find_waldo itself never sees it.
        overlap = iou(feature_box, truth) if feature_box else 0.0
        # Do the two completely different methods agree with each other?
        agreement = (iou(feature_box, corr_box)
                     if feature_box and corr_box else 0.0)
        print("{:<16}{:>9}{:>8}{:>9.3f}{:>10.3f}".format(
            name, found["n_matches"], found["n_votes"], overlap, agreement))

        draw_matches(template, scene_rgb, found["kp_template"],
                        found["kp_scene"], found["matches"], found["votes"],
                        f"plots/task4_{name}_matches.png")

        shown = draw_box(scene_rgb, truth, (0.1, 0.4, 1.0))
        if feature_box:
            shown = draw_box(shown, feature_box,
                             (0, 1, 0) if overlap >= 0.5 else (1, 0, 0))
        plt.figure(figsize=(9, 7))
        plt.imshow(shown)
        plt.title("{}: blue = truth, green/red = your detection".format(name))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("plots/task4_{}.png".format(name),
                    metadata={"Author": getpass.getuser()})
        plt.close()

    print("\nwrote plots/task4_matches.png and plots/task4_<scene>.png")
    print("The last column is agreement between Task 2 and Task 4, computed")
    print("without any ground truth. Compare it against the IoU column.")