"""All the pictures: galleries, the similarity matrix, match graphs, spanning trees, the triangulation
experiments and the 3D views."""

import collections
import time

import numpy as np
import cv2
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import plotly
import plotly.graph_objects as go
import scipy.sparse as sp

from .data import _GROUPS, _gt_points, _robust_subset, label_name, load_image, thumbnail
from .geometry import camera_center, ransac_fundamental_cv
from .graph import skeleton_nodes, word_matches
from ._yours import yours
from .reference import ref
from .sfm import camera_centres_vs_gt, pair_matches, scene_image, sift_features
from .style import FAINT, GREEN, GREY, INK, INK2, OCHRE, PALETTE, PAPER, RED, RULE, SIM_CMAP


# plotly 6 and later store arrays in the notebook as base64; plotly 5 writes every number as text, in full
_PLOTLY_TEXT = int(plotly.__version__.split(".")[0]) < 6


def _finite(P):
    """P as an (M,3) float array and the indices of its rows with no NaN or inf (one bad point must not blank a plot)."""
    P = np.asarray(P, float).reshape(-1, 3)
    return P, np.flatnonzero(np.isfinite(P).all(axis=1))


def _subset(P, max_points, seed=0):
    """Indices of at most max_points finite rows of P, far strays left out (display only)."""
    P, ok = _finite(P)
    return ok[_robust_subset(P[ok], max_points, seed)] if len(ok) else ok


def _frame(ax, color, width=4):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(color)
        s.set_linewidth(width)
    ax.set_xticks([])
    ax.set_yticks([])


def show_thumbs(rome, ids, titles=None, cols=8, borders=None, size=1.5, suptitle=""):
    """A grid of thumbnails of photos ids, with optional titles and border colours."""
    n = len(ids)
    if n == 0:
        print("no photos to show" + (f" ({suptitle})" if suptitle else ""))
        return
    cols = min(cols, max(n, 1))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(min(12, size * cols), size * 0.8 * rows + 0.4), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k < n:
            ax.imshow(thumbnail(rome, ids[k]))
            if borders is not None:
                ax.axis("on")
                _frame(ax, borders[k])
            if titles is not None:
                ax.set_title(titles[k], fontsize=7)
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def show_crate(rome, n=24, seed=0):
    """n random photos from the crate (the grey caption is the hidden label), and how many of each landmark."""
    ids = np.sort(np.random.default_rng(seed).choice(len(rome["images"]), n, replace=False))
    show_thumbs(rome, ids, [f"{i}: {label_name(rome['labels'][i], rome)}" for i in ids], cols=8,
                suptitle=f"{n} random photos of {len(rome['images'])}")
    counts = collections.Counter(rome["labels"].tolist())
    print("hidden labels: " + ", ".join(f"{label_name(g, rome)} {counts[g]}" for g in sorted(counts) if g >= 0)
          + f", strays {counts[-1]}")


def show_words(rome, i, j, max_shared=150):
    """Photos i and j with their keypoints. Words the two share are drawn as coloured dots, one colour per word (the
    same colour in both photos); all other keypoints are small grey dots."""
    off, xy, w = rome["offsets"], rome["xy"], rome["words"]
    wi, wj = w[off[i]:off[i + 1]], w[off[j]:off[j + 1]]
    shared = np.intersect1d(wi, wj)
    if len(shared) > max_shared:
        shared = np.random.default_rng(0).choice(shared, max_shared, replace=False)
    cmap = matplotlib.colors.ListedColormap(PALETTE + [RED])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, k, wk in ((axes[0], i, wi), (axes[1], j, wj)):
        p = xy[off[k]:off[k + 1]]
        ax.imshow(load_image(rome, k))
        ax.scatter(p[:, 0], p[:, 1], s=2, color=PAPER, alpha=0.6)
        m = np.isin(wk, shared)
        ax.scatter(p[m, 0], p[m, 1], s=22, c=cmap((wk[m] * 7919) % 11 / 10.0), edgecolors=PAPER, linewidths=0.4)
        ax.set_title(f"photo {k} ({label_name(rome['labels'][k], rome)}): {len(wk)} keypoints, {m.sum()} on shared words",
                     fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_similarity_matrix(T, labels, n=600, seed=0, rome=None):
    """Left: cosine similarity T T^T of n random photos, sorted by hidden landmark (strays last), each landmark's block
    outlined. Right: the same averaged per pair of landmarks (one cell per pair), which shows the pattern at a
    glance: photos of one landmark are more alike than photos of two. rome: the crate the labels come from, for the
    landmark names (default: the crate loaded last)."""
    labels = np.asarray(labels)
    ids = np.random.default_rng(seed).choice(T.shape[0], min(n, T.shape[0]), replace=False)
    key = np.where(labels[ids] < 0, 10 ** 6, labels[ids])
    ids = ids[np.lexsort((ids, key))]
    Ts = sp.csr_matrix(T)[ids]
    S = (Ts @ Ts.T).toarray()
    np.fill_diagonal(S, np.nan)
    lab = labels[ids]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(lab)) + 1])
    ends = np.concatenate([starts[1:], [len(lab)]])
    names = [label_name(lab[a], rome) for a in starts]
    M = np.array([[np.nanmean(S[a:b, c:d]) for c, d in zip(starts, ends)] for a, b in zip(starts, ends)])
    lo, hi = np.nanpercentile(S, 50), np.nanpercentile(S, 99.5)
    norm = matplotlib.colors.PowerNorm(0.5, vmin=lo, vmax=hi)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.55))
    im = ax.imshow(S, cmap=SIM_CMAP, norm=norm, interpolation="antialiased", interpolation_stage="rgba")
    for a, b in zip(starts, ends):                          # outline each landmark's block on the diagonal
        ax.add_patch(matplotlib.patches.Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False, ec=INK, lw=0.8))
    ax.set_yticks((starts + ends - 1) / 2)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks([])
    ax.set_title(f"{len(ids)} photos, sorted by hidden landmark")

    norm2 = matplotlib.colors.PowerNorm(0.5, vmin=np.nanmin(M), vmax=np.nanmax(M))
    im2 = bx.imshow(M, cmap=SIM_CMAP, norm=norm2)
    for r in range(len(M)):
        for c in range(len(M)):
            dark = np.isfinite(M[r, c]) and float(norm2(M[r, c])) > 0.55   # the same scale as the cell's colour
            bx.text(c, r, f"{M[r, c]:.3f}", ha="center", va="center", fontsize=6.5, color=PAPER if dark else INK)
    bx.set_xticks(range(len(M)))
    bx.set_xticklabels(names, fontsize=7, rotation=60, ha="right")
    bx.set_yticks(range(len(M)))
    bx.set_yticklabels(names, fontsize=7)
    bx.set_title("mean similarity per pair of landmarks")
    for spine in bx.spines.values():
        spine.set_visible(False)
    for img, axis, label, ext in [(im, ax, "cosine similarity", "min"), (im2, bx, "mean cosine similarity", "neither")]:
        # left: half the pairs lie below the median, where the scale starts; the arrow says they are clipped
        cb = fig.colorbar(img, ax=axis, label=label, shrink=0.8, pad=0.03, extend=ext)
        cb.outline.set_visible(False)
    plt.show()


def show_neighbours(rome, idx, score, queries=(0, 1, 2, 3, 4), k=8):
    """Each row: a query photo (dark ink border), then its k best neighbours. Green border: same hidden landmark as the
    query; red: not. Strays (label -1) are never "right", so all their neighbours are red. k is capped at the number
    of columns of idx."""
    labels = rome["labels"]
    idx, score = np.asarray(idx), np.asarray(score)
    k = max(0, min(int(k), idx.shape[1] if idx.ndim == 2 else 0))
    queries = list(queries)
    rows = len(queries)
    if rows == 0:
        print("no queries to show")
        return
    fig, axes = plt.subplots(rows, k + 1, figsize=(12, 12 / (k + 1) * 0.85 * rows + 0.3), squeeze=False)
    for r, q in enumerate(queries):
        for c in range(k + 1):
            ax = axes[r, c]
            i = q if c == 0 else idx[q, c - 1]
            ax.imshow(thumbnail(rome, i))
            if c == 0:
                _frame(ax, INK, 5)
                ax.set_title(f"query {q}\n{label_name(labels[q], rome)}", fontsize=7)
            else:
                same = labels[q] >= 0 and labels[i] == labels[q]
                _frame(ax, GREEN if same else RED, 4)
                ax.set_title(f"{score[q, c - 1]:.2f}", fontsize=7)
    plt.tight_layout()
    plt.show()


def plot_retrieval_precision(prec, top_k=10):
    """Precision at k (from retrieval_precision) against k."""
    k = np.arange(1, len(prec) + 1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(k, prec, "o-", color=INK, ms=4)
    if top_k <= len(prec):
        ax.axvline(top_k, color=GREY, ls="--", lw=1)
        ax.annotate(f"k = {top_k}: {prec[top_k - 1]:.0%}", (top_k, prec[top_k - 1]), xytext=(8, 8),
                    textcoords="offset points")
    ax.set_xlabel("k (neighbours per photo)")
    ax.set_ylabel("share of right neighbours")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title("retrieval precision at k (landmark photos only)")
    plt.tight_layout()
    plt.show()


def _side_by_side(img1, img2):
    h = max(img1.shape[0], img2.shape[0])
    canvas = np.full((h, img1.shape[1] + img2.shape[1], 3), 255, dtype=np.uint8)
    canvas[:img1.shape[0], :img1.shape[1]] = img1
    canvas[:img2.shape[0], img1.shape[1]:] = img2
    return canvas


def show_matches(img1, img2, x1, x2, inliers=None, max_lines=200, title=""):
    """Matches drawn as lines between two images side by side (dark ink). With an inlier mask: inliers green, outliers
    red."""
    x1, x2 = np.asarray(x1, float).reshape(-1, 2), np.asarray(x2, float).reshape(-1, 2)
    n = len(x1)
    if inliers is not None:
        inliers = np.asarray(inliers, bool).reshape(-1)
    w1 = img1.shape[1]
    canvas = _side_by_side(img1, img2)
    fig, ax = plt.subplots(figsize=(12, 12 * canvas.shape[0] / canvas.shape[1] + 0.4))
    ax.imshow(canvas)
    rng = np.random.default_rng(0)
    groups = [(np.ones(n, bool), INK)] if inliers is None else \
        [(~np.asarray(inliers, bool), RED), (np.asarray(inliers, bool), GREEN)]
    for mask, col in groups:
        idx = np.flatnonzero(mask)
        if len(idx) > max_lines:
            idx = np.sort(rng.choice(idx, max_lines, replace=False))
        segs = [[(x1[i, 0], x1[i, 1]), (x2[i, 0] + w1, x2[i, 1])] for i in idx]
        ax.add_collection(LineCollection(segs, colors=col, linewidths=0.7, alpha=0.8))
        ax.scatter(x1[idx, 0], x1[idx, 1], s=6, color=col)
        ax.scatter(x2[idx, 0] + w1, x2[idx, 1], s=6, color=col)
    info = f"{n} matches" if inliers is None else \
        f"{int(np.sum(inliers))} inliers (green), {n - int(np.sum(inliers))} outliers (red)"
    ax.set_title((title + ": " if title else "") + info)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_word_matches(rome, i, j, thresh=2.0, min_inliers=20):
    """What verify_pairs does to one pair: the word matches of photos i and j, split by RANSAC on F."""
    x1, x2 = word_matches(rome, i, j)
    _, inl = ransac_fundamental_cv(x1, x2, thresh=thresh) if len(x1) >= 8 else (None, np.zeros(len(x1), bool))
    li, lj = label_name(rome["labels"][i], rome), label_name(rome["labels"][j], rome)
    verdict = "a real pair" if inl.sum() >= min_inliers else f"rejected (fewer than {min_inliers})"
    show_matches(load_image(rome, i), load_image(rome, j), x1, x2, inl,
                 title=f"photos {i} ({li}) and {j} ({lj}), {len(x1)} word matches, {verdict}")


def _draw_pair(ax, img1, img2, x1, x2, groups, max_lines=150, seed=0):
    """Two photos side by side on ax, with matches as lines: groups is a list of (mask, colour)."""
    canvas = _side_by_side(img1, img2)
    w1 = img1.shape[1]
    ax.imshow(canvas)
    rng = np.random.default_rng(seed)
    for mask, col in groups:
        idx = np.flatnonzero(mask)
        if len(idx) > max_lines:
            idx = np.sort(rng.choice(idx, max_lines, replace=False))
        segs = [[(x1[k, 0], x1[k, 1]), (x2[k, 0] + w1, x2[k, 1])] for k in idx]
        ax.add_collection(LineCollection(segs, colors=col, linewidths=0.6, alpha=0.75))
        ax.scatter(x1[idx, 0], x1[idx, 1], s=4, color=col)
        ax.scatter(x2[idx, 0] + w1, x2[idx, 1], s=4, color=col)
    ax.set_xlim(0, canvas.shape[1])
    ax.set_ylim(canvas.shape[0], 0)
    ax.axis("off")


def show_match_filters(rome, i, j, ratio=0.8, thresh=2.0):
    """How two photos are matched keypoint by keypoint, in three stages, on SIFT of photos i and j:
    plain nearest neighbours (OpenCV), plus Lowe's ratio test (OpenCV), plus the mutual check (YOUR match_descriptors).
    Each set is then checked with RANSAC on F: green lines fit one geometry, red lines do not (at most 150 of each
    are drawn). Prints the counts; returns dict(sift_s, match_s): seconds of SIFT per photo and of your matching."""
    t0 = time.time()
    x1, d1 = sift_features(rome, i)
    x2, d2 = sift_features(rome, j)
    t_sift = (time.time() - t0) / 2
    d1, d2 = d1.astype(np.float32), d2.astype(np.float32)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    plain = np.array([(m.queryIdx, m.trainIdx) for m in bf.match(d1, d2)], dtype=np.int64).reshape(-1, 2)
    knn = bf.knnMatch(d1, d2, k=2)
    lowe = np.array([(a.queryIdx, a.trainIdx) for a, b in knn if a.distance < ratio * b.distance],
                    dtype=np.int64).reshape(-1, 2)
    t0 = time.time()
    mine = np.asarray(yours("match_descriptors")(d1, d2, ratio), dtype=np.int64).reshape(-1, 2)
    t_match = time.time() - t0
    stages = [("every keypoint takes its nearest neighbour", plain),
              (f"+ ratio test: best clearly better than second best (ratio {ratio})", lowe),
              ("+ mutual check: both photos agree (your match_descriptors)", mine)]
    img1, img2 = load_image(rome, i), load_image(rome, j)
    h = max(img1.shape[0], img2.shape[0]) / (img1.shape[1] + img2.shape[1])
    fig, axes = plt.subplots(3, 1, figsize=(11, 3 * 11 * h + 1.2))
    rows = []
    for ax, (name, m) in zip(axes, stages):
        a, b = x1[m[:, 0]], x2[m[:, 1]]
        _, inl = ransac_fundamental_cv(a, b, thresh=thresh, iters=2000) if len(m) >= 8 else \
            (None, np.zeros(len(m), bool))
        n_in = int(inl.sum())
        _draw_pair(ax, img1, img2, a, b, [(~inl, RED), (inl, GREEN)])
        share = n_in / max(len(m), 1)
        ax.set_title(f"{name}: {len(m)} matches, {n_in} fit one F ({share:.0%})", fontsize=10)
        rows.append((name.split(":")[0].split("(")[0].strip(), len(m), n_in, share))
    plt.tight_layout()
    plt.show()
    print(f"{len(x1)} and {len(x2)} keypoints; SIFT {1000 * t_sift:.0f} ms per photo, your match_descriptors "
          f"{1000 * t_match:.0f} ms for the pair")
    for name, n, n_in, share in rows:
        print(f"  {n:5d} matches, {n_in:4d} fit one F ({share:4.0%})   {name}")
    return dict(sift_s=t_sift, match_s=t_match)


def show_homography_test(scene, results, thresh=2.0):
    """The homography test on many pairs of one component, and two of them in detail.

    results: a list of (i, j, ratio, n_H, n_F) from homography_ratio, one per pair (local photo numbers).
    Left: H inliers against F inliers, one dot per pair; on the red line one homography explains every match.
    Right: the ratio against the number of matches (F inliers). Then the pair with the highest ratio (no step between
    the cameras) and the one with the lowest (real parallax), each with YOUR homography_ratio: red lines fit the
    homography, green ones only fit F."""
    res = sorted(results, key=lambda r: r[2])
    if not res:
        print("no pairs to show: results is empty")
        return
    nF = np.array([r[4] for r in res], float)
    nH = np.array([r[3] for r in res], float)
    ratio = np.array([r[2] for r in res], float)
    hi, lo = res[-1], res[0]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    top = max(nF.max(), nH.max(), 1) * 1.08
    ax.plot([0, top], [0, top], color=RED, lw=1, ls="--", label="ratio 1: one homography explains all")
    ax.scatter(nF, nH, s=18, color=INK, zorder=3)
    ax2.axhline(1.0, color=RED, lw=1, ls="--")
    ax2.scatter(nF, ratio, s=18, color=INK, zorder=3)
    for r, tag in ((hi, "A"), (lo, "B")):
        ax.annotate(tag, (r[4], r[3]), xytext=(6, -12), textcoords="offset points", color=RED, fontsize=11)
        ax2.annotate(tag, (r[4], r[2]), xytext=(6, -12), textcoords="offset points", color=RED, fontsize=11)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("F inliers (fit some two view geometry)")
    ax.set_ylabel("H inliers (fit one homography)")
    ax.set_title(f"{len(res)} pairs, one dot each")
    ax.legend(loc="upper left", fontsize=8)
    ax2.set_xlim(0, top)
    ax2.set_ylim(0, 1.08)
    ax2.set_xlabel("F inliers (matches)")
    ax2.set_ylabel("homography ratio")
    ax2.set_title("more matches, higher ratio")
    plt.tight_layout()
    plt.show()
    homography_ratio = yours("homography_ratio")
    for r, tag, what in ((hi, "A", "the highest ratio, hardly any step between the cameras"),
                         (lo, "B", "the lowest ratio, a real step and parallax")):
        i, j = int(r[0]), int(r[1])
        x1, x2 = pair_matches(scene, i, j)
        rt, _, inl_H, inl_F = homography_ratio(x1, x2, thresh=thresh)
        inl_H, inl_F = np.asarray(inl_H, bool), np.asarray(inl_F, bool)
        img1, img2 = scene_image(scene, i), scene_image(scene, j)
        h = max(img1.shape[0], img2.shape[0]) / (img1.shape[1] + img2.shape[1])
        fig, axp = plt.subplots(figsize=(11, 11 * h + 0.5))
        _draw_pair(axp, img1, img2, x1, x2, [(inl_F & ~inl_H, GREEN), (inl_H, RED)], max_lines=200)
        axp.set_title(f"{tag} (photos {i} and {j}), {what}. {inl_H.sum()} fit H (red), "
                      f"{np.sum(inl_F & ~inl_H)} only fit F (green), ratio {rt:.2f}", fontsize=10)
        plt.tight_layout()
        plt.show()


def show_reprojection(scene, i, j, x1, x2, x1_hat, x2_hat, scale=None):
    """Experiment A: how well triangulated points explain the pixels they came from.

    x1, x2 (N,2) are the observed pixels in photos i and j (local numbers), x1_hat, x2_hat (N,2) the triangulated
    points projected back into each photo. Each photo shows the observed points with a short line to where the point
    reprojects, drawn scale times longer than it is (the errors are often well under a pixel; by default the scale
    draws the 90th percentile error at about 6% of the photo's size); the histogram holds the errors of both
    photos."""
    x1, x2 = np.asarray(x1, float).reshape(-1, 2), np.asarray(x2, float).reshape(-1, 2)
    x1_hat, x2_hat = np.asarray(x1_hat, float).reshape(-1, 2), np.asarray(x2_hat, float).reshape(-1, 2)
    e1, e2 = np.linalg.norm(x1_hat - x1, axis=1), np.linalg.norm(x2_hat - x2, axis=1)
    both = np.concatenate([e1, e2])
    both = both[np.isfinite(both)]
    top = max(float(np.percentile(both, 95)) if len(both) else 1.0, 1e-6)
    if scale is None:
        size = max(scene_image(scene, i).shape[:2])
        p90 = float(np.percentile(both, 90)) if len(both) else 1.0
        scale = int(np.clip(round(0.06 * size / max(p90, 1e-3)), 1, 1000))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    for ax, img, x, x_hat, e, k in ((axes[0], scene_image(scene, i), x1, x1_hat, e1, i),
                                    (axes[1], scene_image(scene, j), x2, x2_hat, e2, j)):
        ax.imshow(img)
        ok = np.isfinite(x_hat).all(axis=1)
        tip = x + scale * (x_hat - x)
        seg = np.stack([x[ok], tip[ok]], axis=1)
        ax.add_collection(LineCollection(seg, colors=RED, linewidths=1.6, zorder=4))
        ax.scatter(x[:, 0], x[:, 1], s=3, color=INK, zorder=3)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.set_axis_off()
        ax.set_title(f"photo {k}: median error {np.median(e[ok]) if ok.any() else np.nan:.2f} px "
                     f"(lines {scale} times longer)", fontsize=10)
    ax = axes[2]
    cap = 3 * top
    ax.hist(np.clip(both, 0, cap), bins=40, color=FAINT, edgecolor=INK2, linewidth=0.4)
    med = float(np.median(both)) if len(both) else np.nan
    ax.axvline(med, color=RED, lw=1.2, label=f"median {med:.2f} px")
    ax.set_xlabel("reprojection error (px)" + (f"; the last bar holds {np.sum(both > cap)} above {cap:.2g} px"
                                                  if np.any(both > cap) else ""))
    ax.set_ylabel("errors")
    ax.set_title(f"{len(both):,} errors: {len(x1):,} matches, each seen in two photos", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def baseline_experiment(noise=1.0, baselines=None, n=300, seed=0):
    """Experiment B: how the step between two cameras (the baseline) sets the accuracy of triangulated depth.

    Synthetic and known: n points 8 to 12 units in front, camera 1 at the origin and camera 2 the baseline b to its
    right, both looking straight ahead (focal 800 px). Every observed pixel gets Gaussian noise of noise px, then YOUR
    triangulate_points rebuilds the points. Left: the median 3D error against b, next to the textbook estimate for
    the depth error, about Z^2 noise / (f b), the depth error a pixel of disparity error makes. Right: the points
    seen from above, rebuilt from the shortest baseline and from one 50 times longer.
    Returns (baselines, median 3D errors)."""
    triangulate_points = yours("triangulate_points")
    rng = np.random.default_rng(seed)
    baselines = np.geomspace(0.02, 4.0, 12) if baselines is None else np.asarray(baselines, float)
    f, Z = 800.0, rng.uniform(8.0, 12.0, n)
    X = np.column_stack([rng.uniform(-0.3, 0.3, n) * Z, rng.uniform(-0.2, 0.2, n) * Z, Z])
    K = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1.0]])

    def seen(P):
        p = np.column_stack([X, np.ones(n)]) @ P.T
        return p[:, :2] / p[:, 2:] + rng.normal(0, noise, (n, 2))
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    errors, rebuilt = [], {}
    for b in baselines:
        P2 = K @ np.hstack([np.eye(3), np.array([[-b], [0], [0]])])    # centre at (b, 0, 0): t = -R C
        Xh = np.asarray(triangulate_points(P1, P2, seen(P1), seen(P2)), float).reshape(-1, 3)
        errors.append(float(np.median(np.linalg.norm(Xh - X, axis=1))))
        rebuilt[float(b)] = Xh
    errors = np.array(errors)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax.loglog(baselines, errors, "o-", color=RED, lw=1.4, ms=4, label="median 3D error, measured")
    guess = 0.674 * np.sqrt(2) * noise * np.median(Z) ** 2 / (f * baselines)   # median of |N(0, 2 noise^2)| ...
    ax.loglog(baselines, guess, "--", color=INK2, lw=1, label="Z² σ / (f b): depth from disparity")
    ax.set_xlabel("baseline b (distance between the cameras)")
    ax.set_ylabel("median 3D error (scene units)")
    ax.set_title(f"{noise:g} px noise, points about {np.median(Z):.0f} units away", fontsize=10)
    ax.legend(fontsize=8)
    short = float(baselines[0])
    long_ = float(baselines[np.argmin(np.abs(baselines - 50 * short))])
    ax2.scatter(X[:, 0], X[:, 2], s=18, facecolors="none", edgecolors=INK2, linewidths=0.6, label="true points",
                zorder=2)
    for b, colour in ((short, RED), (long_, GREEN)):
        Xh = rebuilt[b]
        near = np.isfinite(Xh).all(axis=1) & (Xh[:, 2] > 0) & (Xh[:, 2] < 30)
        ax2.scatter(Xh[near, 0], Xh[near, 2], s=6, color=colour, alpha=0.8, zorder=3,
                    label=f"rebuilt, b = {b:.2g} ({np.sum(~near)} beyond the plot)")
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(0, 30)
    ax2.set_xlabel("x (sideways)")
    ax2.set_ylabel("z (depth)")
    ax2.set_title("seen from above: a short baseline smears points along their rays", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.show()
    return baselines, errors


def compare_with_model(X_yours, X_model, title=""):
    """Your triangulated points next to the model's for the same tracks: seen from above (x and z), and a histogram
    of the distance between the two, as a share of the model's size (the spread of its points)."""
    X_yours, X_model = np.asarray(X_yours, float).reshape(-1, 3), np.asarray(X_model, float).reshape(-1, 3)
    size = float(np.linalg.norm(np.percentile(X_model, 95, axis=0) - np.percentile(X_model, 5, axis=0))) or 1.0
    gap = np.linalg.norm(X_yours - X_model, axis=1) / size * 100
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    lo, hi = np.percentile(X_model, [2, 98], axis=0)
    pad = 0.15 * (hi - lo)
    ax.scatter(X_model[:, 0], X_model[:, 2], s=10, facecolors="none", edgecolors=INK2, linewidths=0.5,
               label="the model's points (bundle adjusted)")
    ax.scatter(X_yours[:, 0], X_yours[:, 2], s=3, color=RED, label="yours, from two photos")
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[2] - pad[2], hi[2] + pad[2])
    ax.set_xlabel("x")
    ax.set_ylabel("z (depth)")
    ax.set_title("seen from above", fontsize=10)
    ax.legend(fontsize=8)
    finite = gap[np.isfinite(gap)]
    cap = max(float(np.percentile(finite, 95)) * 3, 1e-6) if len(finite) else 1.0
    ax2.hist(np.clip(finite, 0, cap), bins=40, color=FAINT, edgecolor=INK2, linewidth=0.4)
    med = float(np.median(finite)) if len(finite) else np.nan
    ax2.axvline(med, color=RED, lw=1.2, label=f"median {med:.2f}%")
    ax2.set_xlabel("distance to the model's point (% of the model's size)"
                   + (f"; the last bar holds {np.sum(finite > cap)} above {cap:.2g}%" if np.any(finite > cap) else ""))
    ax2.set_ylabel("points")
    ax2.set_title(f"{len(X_model):,} tracks", fontsize=10)
    ax2.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


_LAYOUTS = {}


def _spring_layout(n, edges, iters=60, seed=0, max_repel=1200):
    """Force directed layout (Fruchterman and Reingold) of one connected graph, in [-1, 1]^2."""
    if n <= 3:
        return np.array([[0.0, 0.0], [0.6, 0.0], [0.3, 0.5]])[:n]
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 2))
    k, temp = 1.0 / np.sqrt(n), 0.3
    e = np.asarray(edges, np.int64).reshape(-1, 2)
    m = min(n, max_repel)
    for _ in range(iters):
        # every pair pushes apart; a big graph is pushed by a random sample of m nodes, scaled up to all n (memory and
        # time grow as n * m, not n * n)
        other = pos if m == n else pos[rng.choice(n, m, replace=False)]
        d = pos[:, None, :] - other[None, :, :]
        dist2 = np.maximum(np.einsum("ijk,ijk->ij", d, d), 1e-6)
        disp = np.einsum("ijk,ij->ik", d, k * k / dist2) * (n / m)
        de = pos[e[:, 0]] - pos[e[:, 1]]
        f = de * np.linalg.norm(de, axis=1, keepdims=True) / k        # every edge pulls together
        np.add.at(disp, e[:, 0], -f)
        np.add.at(disp, e[:, 1], f)
        ln = np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), 1e-9)
        pos += disp / ln * np.minimum(ln, temp)
        temp *= 0.94
    pos -= pos.mean(axis=0)
    return pos / max(np.abs(pos).max(), 1e-9)


def graph_layout(n, edges, seed=0):
    """2D positions (n,2) for drawing a match graph: each component laid out on its own (largest first, packed in
    rows), the photos with no edge in a thin band underneath. Cached, so redrawing the same graph is instant."""
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    key = (n, edges.tobytes().__hash__(), seed)
    if key in _LAYOUTS:
        return _LAYOUTS[key]
    lab = ref.connected_components(n, edges)
    size = np.bincount(lab)
    pos = np.zeros((n, 2))
    big = [c for c in range(len(size)) if size[c] >= 5]              # already sorted, largest first
    radius = {c: np.sqrt(size[c]) for c in big}
    width = max(1.8 * np.sqrt(sum(r * r for r in radius.values())), 2.2 * max(radius.values(), default=1.0))
    order_e = lab[edges[:, 0]]
    sort_e = np.argsort(order_e, kind="stable")
    starts = np.searchsorted(order_e[sort_e], np.arange(len(size) + 1))
    x = y = row_h = 0.0
    for c in big:
        mem = np.flatnonzero(lab == c)
        r = radius[c]
        if x + r > width and x > 0:
            x, y, row_h = 0.0, y - row_h - 0.6, 0.0
        loc = -np.ones(n, np.int64)
        loc[mem] = np.arange(len(mem))
        p = _spring_layout(len(mem), loc[edges[sort_e[starts[c]:starts[c + 1]]]],
                           iters=60 if len(mem) < 1500 else 35, seed=seed)
        pos[mem] = p * r / 2 + [x + r / 2, y - r / 2]
        x += r + 0.6
        row_h = max(row_h, r)
    top = y - row_h - 1.0
    small = [c for c in range(len(size)) if 1 < size[c] < 5]          # pairs, triangles: a band of tiny rings
    if small:
        cols = max(int(width // 2.2), 1)
        for k, c in enumerate(small):
            mem = np.flatnonzero(lab == c)
            ang = 2 * np.pi * np.arange(len(mem)) / len(mem)
            pos[mem] = np.column_stack([np.cos(ang), np.sin(ang)]) * 0.5 + [(k % cols) * 2.2 + 0.5, top - (k // cols) * 2.0]
        top -= (len(small) - 1) // cols * 2.0 + 2.5
    alone = np.flatnonzero(size[lab] == 1)
    if len(alone):
        cols = max(int(np.ceil(len(alone) / max(1, int(np.ceil(np.sqrt(len(alone)) / 4))))), 1)
        gap = width / cols
        pos[alone] = np.column_stack([(np.arange(len(alone)) % cols) * gap, top - (np.arange(len(alone)) // cols) * gap])
    _LAYOUTS[key] = pos
    return pos


def _node_colors(values, top=10):
    """Colours for labels: 0 .. top-1 get their own colour, larger numbers a faint grey, -1 (stray) dark grey."""
    values = np.asarray(values)
    return [PALETTE[v] if 0 <= v < min(top, len(PALETTE)) else (INK2 if v < 0 else FAINT) for v in values]


def plot_match_graph(n, edges, colors_by, title="", max_edges=60000, kind="labels", rome=None):
    """The match graph: one dot per photo, one line per edge. colors_by, with kind saying which it is:
    kind="labels": hidden landmark labels (-1 = stray, dark grey); kind="components": component numbers (0 .. 9
    coloured, smaller components faint grey). rome: the crate the labels come from, for the landmark names in the
    legend (default: the crate loaded last)."""
    if kind not in ("labels", "components"):
        raise ValueError(f'kind must be "labels" or "components", got {kind!r}')
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    pos = graph_layout(n, edges)
    colors_by = np.asarray(colors_by)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    e = edges if len(edges) <= max_edges else edges[np.random.default_rng(0).choice(len(edges), max_edges, False)]
    ax.add_collection(LineCollection(pos[e], colors=INK2, linewidths=0.3, alpha=0.3, zorder=1))
    deg = np.bincount(edges.ravel(), minlength=n)
    ax.scatter(pos[:, 0], pos[:, 1], s=np.where(deg > 0, 5, 1), c=_node_colors(colors_by), zorder=2, linewidths=0)
    alone = np.flatnonzero(deg == 0)
    if len(alone):
        ax.annotate(f"{len(alone):,} photos with no verified pair", (pos[alone, 0].min(), pos[alone, 1].max()),
                    xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2, va="bottom")
    if kind == "labels":
        groups = _GROUPS if rome is None else rome["groups"]
        names = [(label_name(g, rome), PALETTE[g] if g < len(PALETTE) else FAINT) for g in range(len(groups))]
        if len(colors_by) and colors_by.min() < 0:
            names.append(("stray", INK2))
    else:
        top = int(colors_by.max()) + 1 if len(colors_by) else 0
        names = [(f"component {c}", PALETTE[c]) for c in range(min(10, top))]
        if top > 10:
            names.append(("smaller components", FAINT))
    for name, col in names:
        ax.scatter([], [], s=25, color=col, label=name)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    ax.set_title(title or f"{n} photos, {len(edges):,} edges")
    plt.tight_layout()
    plt.show()


def show_components(rome, comp, n_components=9, n_photos=8, seed=0):
    """One row of photos per component (the biggest first), with its size and the landmark most of them show."""
    comp, labels = np.asarray(comp), rome["labels"]
    rng = np.random.default_rng(seed)
    n_components = min(n_components, comp.max() + 1)
    fig, axes = plt.subplots(n_components, n_photos, figsize=(12, 12 / n_photos * 0.8 * n_components + 0.3),
                             squeeze=False)
    for c in range(n_components):
        mem = np.flatnonzero(comp == c)
        pick = np.sort(rng.choice(mem, min(n_photos, len(mem)), replace=False))
        lm = labels[mem][labels[mem] >= 0]
        major = np.bincount(lm).argmax() if len(lm) else -1
        for k in range(n_photos):
            ax = axes[c, k]
            ax.axis("off")
            if k < len(pick):
                ax.imshow(thumbnail(rome, pick[k]))
                ax.axis("on")
                _frame(ax, PALETTE[c % len(PALETTE)], 3)
        axes[c, 0].set_title(f"component {c}: {len(mem)} photos, mostly {label_name(major, rome)} "
                             f"({np.mean(labels[mem] == major):.0%})", fontsize=9, loc="left")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.05, hspace=0.3)
    plt.show()


def plot_spanning_tree(n, edges, keep, title="", size=100):
    """All verified edges of one component (thin grey), the spanning tree and the other photos (dark ink), and the
    skeleton photos (red) that
    run_sfm_skeleton reconstructs first (the tree with its leaves peeled off, see skeleton_nodes)."""
    edges, keep = np.asarray(edges, np.int64).reshape(-1, 2), np.asarray(keep, bool)
    pos = graph_layout(n, edges[keep], seed=1)
    skel = skeleton_nodes(n, edges[keep], size)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_collection(LineCollection(pos[edges[~keep]], colors=INK2, linewidths=0.3, alpha=0.25))
    ax.add_collection(LineCollection(pos[edges[keep]], colors=INK, linewidths=1.1))
    rest = np.setdiff1d(np.arange(n), skel)
    ax.scatter(pos[rest, 0], pos[rest, 1], s=10, color=INK, zorder=3, label=f"other photos ({len(rest)})")
    ax.scatter(pos[skel, 0], pos[skel, 1], s=30, color=RED, zorder=4, label=f"skeleton ({len(skel)})")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    ax.set_title(title or f"spanning tree: {keep.sum()} of {len(edges)} edges")
    plt.tight_layout()
    plt.show()


def _marker_colors(colors):
    """Plotly marker settings for (M,3) uint8 colours that stay small in a saved notebook (a colour table)."""
    c = np.asarray(colors, np.uint8).reshape(-1, 3).astype(np.int64)
    if len(c) == 0:
        return dict(color=GREY)
    for bits in (3, 4):
        q = c >> bits
        used, idx = np.unique((q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2], return_inverse=True)
        if len(used) <= 4096:
            break
    rgb = np.stack([used >> 16, (used >> 8) & 255, used & 255], axis=1) * (1 << bits) + (1 << bits) // 2
    names = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    n = len(names)
    scale = [[0, names[0]], [1, names[0]]] if n == 1 else [[k / (n - 1), name] for k, name in enumerate(names)]
    return dict(color=idx.ravel().astype(np.uint16), colorscale=scale, cmin=0, cmax=max(n - 1, 1), showscale=False)


def _frustum(R, t, size, aspect=0.75, half_w=0.5):
    """Line segments of a small camera pyramid (depth size, half width half_w * size), in world coordinates."""
    C = camera_center(R, t)
    w, h = half_w * size, half_w * size * aspect
    corners = (np.array([[-w, -h, size], [w, -h, size], [w, h, size], [-w, h, size]]) - t) @ R
    segs = [np.array([C, c]) for c in corners]
    segs += [np.array([corners[k], corners[(k + 1) % 4]]) for k in range(4)]
    return segs


def _points_trace(P, colors, name, size=None, color=None, **kw):
    P32 = np.asarray(P, np.float32).reshape(-1, 3)
    if _PLOTLY_TEXT and len(P32):     # 5 significant digits of the scene size: a third of the text, the same picture
        ok = P32[np.isfinite(P32).all(axis=1)]
        span = float(np.ptp(ok, axis=0).max()) if len(ok) else 1.0
        P32 = np.round(P32.astype(np.float64), int(max(0, 4 - np.floor(np.log10(max(span, 1e-9))))))
    size = size or (2 if len(P32) < 20000 else 1.5)
    if colors is not None:
        marker = dict(size=size, **_marker_colors(colors))
    else:
        marker = dict(size=size, color=color or GREY)
    return go.Scatter3d(x=P32[:, 0], y=P32[:, 1], z=P32[:, 2], mode="markers", marker=marker, name=name,
                        hoverinfo="skip", **kw)


def _cams_trace(poses, size, shapes, color, name):
    """One line trace with the pyramids of all poses [(R, t)]; shapes [(half_w, aspect)]."""
    xs, ys, zs = [], [], []
    for (R, t), (hw, asp) in zip(poses, shapes):
        for s in _frustum(np.asarray(R, float), np.asarray(t, float).reshape(3), size, asp, hw):
            xs += [float(s[0, 0]), float(s[1, 0]), None]
            ys += [float(s[0, 1]), float(s[1, 1]), None]
            zs += [float(s[0, 2]), float(s[1, 2]), None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=1.5), name=name, hoverinfo="skip")


def _scene_view(P, centers, Rs, frame_cams=True, elev=0.8, zoom=1.0):
    """Plotly scene settings that frame a 3D view: up is the mean of the cameras' -y axes, the eye above and behind
    the cameras (elev: how far above), zoom > 1 comes closer. With frame_cams=False the frame holds the points only
    (the cameras just say where the front is). Cameras far from the points are left out. Plain paper, no grid."""
    P, ok = _finite(P)
    P = P[ok]
    centers, ok = _finite(centers)
    centers = centers[ok]
    Rs = np.asarray(Rs, float).reshape(-1, 3, 3)
    Rs = Rs[np.isfinite(Rs).all(axis=(1, 2))]
    up = -Rs[:, 1].mean(axis=0) if len(Rs) else np.zeros(3)
    if not np.linalg.norm(up) > 1e-6:          # no camera (or they cancel out): z up, as plotly does
        up = np.array([0.0, 0.0, 1.0])
    up = up / np.linalg.norm(up)
    if not len(P) and not len(centers):        # nothing to frame: a unit cube round the origin
        P = np.array([[-1.0, -1, -1], [1, 1, 1]])
    mid = np.median(P, axis=0) if len(P) else centers.mean(axis=0)
    if len(P) > 10 and len(centers):
        reach = 3.0 * np.percentile(np.linalg.norm(P - mid, axis=1), 90)
        near = np.linalg.norm(centers - mid, axis=1) < reach
        centers = centers[near] if near.sum() >= 1 else centers
    allp = np.vstack([P, centers]) if len(centers) and (frame_cams or not len(P)) else P
    back = (np.median(centers, axis=0) if len(centers) else mid + up) - mid
    back = back - (back @ up) * up
    back = back / max(np.linalg.norm(back), 1e-9)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    side = max((hi - lo).max(), 1e-9)
    focus = (mid - (lo + hi) / 2) / side
    radius = np.percentile(np.linalg.norm(P - mid, axis=1), 60) / side if len(P) > 10 else 0.5
    view = (1.8 * back + elev * up) / np.linalg.norm(1.8 * back + elev * up)
    eye = focus + np.clip(5.0 * radius, 0.6, 2.5) / zoom * view
    axis = lambda a, b: dict(range=[a, b], title="", showgrid=False, zeroline=False, showbackground=False,
                             showticklabels=False, showspikes=False, linecolor=RULE)
    return dict(aspectmode="manual", aspectratio=dict(zip("xyz", np.maximum((hi - lo) / side, 0.05).tolist())),
                bgcolor=PAPER,
                xaxis=axis(lo[0], hi[0]), yaxis=axis(lo[1], hi[1]), zaxis=axis(lo[2], hi[2]),
                camera=dict(up=dict(zip("xyz", up.tolist())), eye=dict(zip("xyz", eye.tolist())),
                            center=dict(zip("xyz", focus.tolist()))))


def _layout_3d(fig, P, centers, Rs, title, **view):
    """Paper background, serif font, a short title, and the framed scene of _scene_view (view: its options)."""
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=650, margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(x=0, y=1, itemsizing="constant"), paper_bgcolor=PAPER,
                      font=dict(family="DejaVu Serif, Georgia, serif", color=INK),
                      scene=_scene_view(P, centers, Rs, **view))


def _cam_size(centers, P):
    centers = np.asarray(centers, float).reshape(-1, 3)
    centers = centers[np.isfinite(centers).all(axis=1)]          # one bad centre must not hide every camera
    allp = np.vstack([np.asarray(P, float).reshape(-1, 3), np.asarray(centers, float).reshape(-1, 3)])
    allp = allp[np.isfinite(allp).all(axis=1)]
    span = max(float(np.ptp(allp, axis=0).max()), 1e-6) if len(allp) > 1 else 1.0
    if len(centers) > 1:
        sub = centers[np.random.default_rng(0).choice(len(centers), min(len(centers), 400), replace=False)]
        gaps = np.linalg.norm(sub[:, None] - sub[None], axis=2) + np.eye(len(sub)) * 1e9
        return float(np.clip(0.15 * np.median(gaps.min(axis=1)), 0.003 * span, 0.012 * span))
    return 0.08 * span


def plot_reconstruction(rec, scene, dense=None, title="", max_points=None, cameras=None, show=True):
    """Interactive 3D view (plotly) of a reconstruction: the points in their colours, skeleton cameras in red, the
    others in ochre. dense: the result of densify, drawn instead of the sparse points, with no cameras (cameras=True
    adds them) and up to 240,000 points (max_points: more is denser, but every point is stored in the notebook).
    show=False returns the figure."""
    is_dense = dense is not None
    pts, colors = (dense["pts"], dense["colors"]) if is_dense else (rec["pts"], rec["colors"])
    cameras = (not is_dense) if cameras is None else cameras
    pts = np.asarray(pts, float).reshape(-1, 3)
    idx = _subset(pts, max_points or (240000 if is_dense else 150000))
    P = pts[idx]
    keys = sorted(rec["cams"])
    poses = [rec["cams"][k] for k in keys]
    centers = np.array([camera_center(R, t) for R, t in poses]).reshape(-1, 3)
    name = f"{'dense ' if is_dense else ''}points ({len(pts):,})"
    fig = go.Figure(_points_trace(P, None if colors is None else np.asarray(colors)[idx], name,
                                  size=1.6))
    if cameras:
        shapes = [(scene["sizes"][k][0] / (2 * scene["K"][k][0, 0]), scene["sizes"][k][1] / scene["sizes"][k][0])
                  for k in keys]
        size = _cam_size(centers, P)
        skel = set(rec.get("skeleton", keys))
        for sel, col, nm in ((False, OCHRE, "placed by PnP"), (True, RED, "skeleton")):
            ks = [k for k in range(len(keys)) if (keys[k] in skel) == sel]
            if ks:
                fig.add_trace(_cams_trace([poses[k] for k in ks], size, [shapes[k] for k in ks], col,
                                          f"{nm} ({len(ks)})"))
    _layout_3d(fig, P, centers, [R for R, _ in poses],
               title or f"{scene['name']}: {len(keys)} cameras, {len(pts):,} points",
               frame_cams=False, elev=0.5 if is_dense else 0.8, zoom=1.2 if is_dense else 1.1)
    if not show:
        return fig
    fig.show()


def plot_alignment(rec, scene, s, R, t, show=True):
    """Our map moved into the survey frame with (s, R, t): survey points (their colours), survey camera centres
    (dark ink), ours (red), a grey line from each of ours to its survey position."""
    C_est, C_gt = camera_centres_vs_gt(rec, scene)
    C_al = s * C_est @ np.asarray(R).T + t
    P_gt, rgb = _gt_points(scene["root"], scene["group"])
    P_gt, ok = _finite(P_gt)
    P_gt, rgb = P_gt[ok], None if rgb is None else np.asarray(rgb)[ok]
    ours = s * np.asarray(rec["pts"], float).reshape(-1, 3) @ np.asarray(R).T + t
    oi = _subset(ours, 20000)
    fig = go.Figure()
    if len(P_gt):
        fig.add_trace(_points_trace(P_gt, rgb, f"survey points ({len(P_gt):,})", size=1.5))
    fig.add_trace(_points_trace(ours[oi], None, f"our points ({len(ours):,})", size=1.5, color=RED))
    segs = np.full((3 * len(C_al), 3), np.nan)
    segs[0::3], segs[1::3] = C_gt, C_al
    fig.add_trace(go.Scatter3d(x=segs[:, 0], y=segs[:, 1], z=segs[:, 2], mode="lines", line=dict(color=GREY, width=2),
                               name="error", hoverinfo="skip"))
    for C, col, name in ((C_gt, INK, "survey cameras"), (C_al, RED, "our cameras")):
        C32 = C.astype(np.float32)
        fig.add_trace(go.Scatter3d(x=C32[:, 0], y=C32[:, 1], z=C32[:, 2], mode="markers",
                                   marker=dict(size=3, color=col), name=f"{name} ({len(C)})", hoverinfo="skip"))
    keys = [k for k in sorted(rec["cams"]) if scene["gt"]["has"][k]]
    P_view = P_gt if len(P_gt) else ours[oi]
    _layout_3d(fig, P_view, C_gt, scene["gt"]["R"][keys], f"{scene['name']}: our map in the survey frame")
    if not show:
        return fig
    fig.show()


def plot_merge(maps, s, R, t, show=True, max_points=30000):
    """Left: the two survey maps as they come, each in its own frame (scale, turn and place are arbitrary).
    Right: the Basilica map moved into the Square's frame with (s, R, t). Square in ink, Basilica in ochre, the shared
    anchors in red. Each map shows at most max_points points per panel."""
    from plotly.subplots import make_subplots
    A, B = maps["square"], maps["basilica"]
    move = lambda X: s * np.asarray(X, float) @ np.asarray(R).T + t
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], horizontal_spacing=0.01,
                        subplot_titles=("before: two frames", "after umeyama: one map"))
    for col, (PB, LB) in enumerate(((B["pts"], maps["links_basilica"]), (move(B["pts"]), move(maps["links_basilica"]))),
                                   start=1):
        first = col == 1
        ia, ib = _subset(A["pts"], max_points), _subset(PB, max_points)
        fig.add_trace(_points_trace(A["pts"][ia], None, "Square map", size=1.5, color=INK), row=1, col=col)
        fig.add_trace(_points_trace(PB[ib], None, "Basilica map", size=1.5, color=OCHRE), row=1, col=col)
        for L, name in ((maps["links_square"], "anchors in the Square map"), (LB, "anchors in the Basilica map")):
            L32 = np.asarray(L, np.float32)
            fig.add_trace(go.Scatter3d(x=L32[:, 0], y=L32[:, 1], z=L32[:, 2], mode="markers", name=name,
                                       marker=dict(size=2, color=RED, symbol="diamond" if "Square" in name else
                                                   "circle"), hoverinfo="skip", showlegend=first), row=1, col=col)
        for tr in fig.data[-4:]:
            tr.showlegend = first
        P = np.vstack([A["pts"][ia], PB[ib]])
        fig.update_layout({("scene" if first else "scene2"): _scene_view(P, maps["links_square"], A["Rs"])})
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=50, b=0), legend=dict(x=0, y=1, itemsizing="constant"),
                      paper_bgcolor=PAPER,
                      font=dict(family="DejaVu Serif, Georgia, serif", color=INK),
                      title=dict(text="St Peter's: two surveys, one map", font=dict(size=15)))
    fig.update_annotations(font=dict(size=12, color=INK2))
    if not show:
        return fig
    fig.show()


def _letterbox(img, w=200, h=150):
    """img scaled to fit a w x h tile, keeping its shape, centred on paper (None: a blank tile)."""
    tile = np.empty((h, w, 3), np.uint8)
    tile[:] = np.round(np.array(matplotlib.colors.to_rgb(PAPER)) * 255).astype(np.uint8)
    if img is None:
        return tile
    s = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    tile[y:y + small.shape[0], x:x + small.shape[1]] = small[..., :3]
    return tile


def show_atlas(rome, comp, n_components=9, seed=0):
    """The collection, sorted: one card per group (the biggest components), a mosaic of 4 of its photos (shapes kept),
    its size and the landmark most of them show."""
    comp, labels = np.asarray(comp), rome["labels"]
    n_components = min(n_components, comp.max() + 1)
    cols = 3
    rows = int(np.ceil(n_components / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.3 * rows), squeeze=False)
    rng = np.random.default_rng(seed)
    for c, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if c >= n_components:
            continue
        mem = np.flatnonzero(comp == c)
        pick = rng.choice(mem, min(4, len(mem)), replace=False)
        tiles = [_letterbox(thumbnail(rome, i, 200)) for i in pick]
        tiles += [_letterbox(None)] * (4 - len(tiles))                    # blank paper tiles
        mosaic = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        ax.imshow(mosaic)
        lm = labels[mem][labels[mem] >= 0]
        major = np.bincount(lm).argmax() if len(lm) else -1
        ax.set_title(f"component {c}: {label_name(major, rome)}\n{len(mem)} photos, {np.mean(labels[mem] == major):.0%} pure",
                     fontsize=10)
    fig.suptitle("The atlas: the photos sorted into groups", fontsize=13)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.04, hspace=0.22)
    plt.show()


def plot_dense_merge(dense_a, dense_b, s, R, t, title="", max_points=240000, show=True):
    """Two dense clouds in one view, in the photos' colours: dense_a as it is, dense_b moved into dense_a's frame by
    (s, R, t), that is s * X @ R.T + t. Click a legend entry to hide one of them (the legend marks dense_a in ink and
    dense_b in ochre). At most max_points points in all. Seen from the side of dense_a's cameras, slightly above.
    show=False returns the figure."""
    A = np.asarray(dense_a["pts"], float).reshape(-1, 3)
    B = s * np.asarray(dense_b["pts"], float).reshape(-1, 3) @ np.asarray(R).T + t
    n_a = round(max_points * len(A) / max(len(A) + len(B), 1))
    ia, ib = _subset(A, n_a), _subset(B, max_points - n_a)
    fig = go.Figure()
    for P, d, idx, name, col in ((A, dense_a, ia, f"{dense_a['name']} ({len(A):,} points)", INK),
                                 (B, dense_b, ib, f"{dense_b['name']}, moved ({len(B):,} points)", OCHRE)):
        # the cloud in its photo colours, without a legend entry; a fixed colour swatch stands for it in the legend
        # (a photo colour can be white on paper), and clicking the swatch hides the cloud with it
        fig.add_trace(_points_trace(P[idx], np.asarray(d["colors"]).reshape(-1, 3)[idx], name, size=1.4,
                                    legendgroup=name, showlegend=False))
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=6, color=col),
                                   name=name, legendgroup=name, hoverinfo="skip"))
    poses = list(dense_a.get("cams", {}).values())
    _layout_3d(fig, np.vstack([A[ia], B[ib]]), [camera_center(R_, t_) for R_, t_ in poses], [R_ for R_, _ in poses],
               title or f"{dense_a['name']} and {dense_b['name']}: one dense map", frame_cams=False, elev=0.5,
               zoom=1.2)
    if not show:
        return fig
    fig.show()
