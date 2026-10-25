"""The crate: photos, hidden labels, survey cameras (ground truth) and the precomputed visual words."""

import os
import collections
import functools
import time

import numpy as np
import pandas as pd
import cv2


_GROUPS = []          # landmark names of the crate loaded last (label k is _GROUPS[k], -1 a stray); each crate also
                      # keeps its own list in rome["groups"], which label_name(label, rome) uses


def pretty(name):
    """'st_peters_square' -> 'St Peters Square'."""
    return str(name).replace("_", " ").replace("(", "").replace(")", "").strip().title()


def load_mini_rome(root):
    """Load the crate from folder root (mini_rome or mini_rome_small). Photos are read lazily, only when needed.

    Returns a dict:
      root, ids (N,) photo ids, images (N,) file paths, sizes (N,2) (width, height),
      labels (N,) int: the hidden landmark of every photo, -1 for a stray; groups: names, labels[i] -> groups[...],
      K (N,3,3): the survey K of a photo (the role EXIF plays), or a guess (focal = 1.2 x the long side) for strays,
      gt: dict(index (M,) photo index, K, R (M,3,3), t (M,3), group (M,) name, secondary (M,) bool): the survey
          cameras ("ground truth"; every landmark has its own frame). A photo placed by two surveys has two rows; the
          one in the other landmark's frame is marked secondary.
      offsets (N+1,), xy (P,2) float32 keypoint pixels, words (P,) int64 visual words, vocab (n_words, 128):
          keypoints of photo i are xy[offsets[i]:offsets[i+1]], their words words[offsets[i]:offsets[i+1]].
      attribution: a table with source, author and licence of every photo.
    """
    t0 = time.time()
    meta = pd.read_csv(os.path.join(root, "meta", "images.csv"))
    ids = meta["id"].astype(str).to_numpy()
    files = meta["file"].astype(str).to_numpy() if "file" in meta else np.array([f"{i}.jpg" for i in ids])
    paths = np.array([f if os.path.isabs(f) else os.path.join(root, "images", os.path.basename(f)) for f in files])
    sizes = meta[["width", "height"]].to_numpy().astype(int)
    pos = {k: n for n, k in enumerate(ids)}

    lab = pd.read_csv(os.path.join(root, "meta", "labels.csv"))
    group_of = dict(zip(lab["id"].astype(str), lab["group"].astype(str)))
    names = [group_of.get(k, "distractor") for k in ids]
    stray = {"distractor", "stray", "none", "nan", ""}
    groups = sorted({g for g in names if g.lower() not in stray})
    labels = np.array([groups.index(g) if g in groups else -1 for g in names], dtype=np.int64)
    _GROUPS[:] = groups

    gt = dict(index=np.zeros(0, np.int64), K=np.zeros((0, 3, 3)), R=np.zeros((0, 3, 3)), t=np.zeros((0, 3)),
              group=np.zeros(0, dtype=str), secondary=np.zeros(0, bool))
    cam_path = os.path.join(root, "gt", "cameras.npz")
    if os.path.exists(cam_path):
        z = np.load(cam_path, allow_pickle=False)
        rows = [k for k, i in enumerate(z["ids"].astype(str)) if i in pos]
        sec = z["secondary"][rows].astype(bool) if "secondary" in z else np.zeros(len(rows), bool)
        gt = dict(index=np.array([pos[i] for i in z["ids"].astype(str)[rows]], dtype=np.int64),
                  K=z["K"][rows].astype(float), R=z["R"][rows].astype(float), t=z["t"][rows].astype(float),
                  group=z["group"].astype(str)[rows], secondary=sec)
    N = len(ids)
    f = 1.2 * sizes.max(axis=1)
    K = np.zeros((N, 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 2, 2] = f, f, 1.0
    K[:, 0, 2], K[:, 1, 2] = sizes[:, 0] / 2.0, sizes[:, 1] / 2.0
    has_gt = np.zeros(N, bool)
    for k in np.flatnonzero(~gt.get("secondary", np.zeros(len(gt["index"]), bool))):
        K[gt["index"][k]] = gt["K"][k]                 # the survey of the photo's own landmark gives its K
        has_gt[gt["index"][k]] = True

    w = np.load(os.path.join(root, "features", "words.npz"))
    offsets = w["offsets"].astype(np.int64)
    if len(offsets) != N + 1:
        raise ValueError(f"features/words.npz holds {len(offsets) - 1} photos, meta/images.csv {N}")
    vocab = np.load(os.path.join(root, "features", "vocab.npy"))
    att_path = os.path.join(root, "meta", "attribution.csv")
    rome = dict(root=root, ids=ids, images=paths, sizes=sizes, labels=labels, groups=groups, K=K, has_gt=has_gt,
                gt=gt, offsets=offsets, xy=w["xy"].astype(np.float32), words=w["word"].astype(np.int64),
                vocab=vocab, attribution=pd.read_csv(att_path) if os.path.exists(att_path) else None)
    counts = collections.Counter(labels.tolist())
    print(f"{N} photos ({N - counts[-1]} of {len(groups)} landmarks, {counts[-1]} strays), "
          f"{len(rome['words']):,} keypoints, {len(vocab):,} visual words, {has_gt.sum()} photos with a survey camera, "
          f"loaded in {time.time() - t0:.1f} s")
    return rome


@functools.lru_cache(maxsize=48)
def _read_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image(rome, i):
    """Photo i as an RGB uint8 array (the last few are kept in memory)."""
    return _read_rgb(str(rome["images"][i]))


@functools.lru_cache(maxsize=1024)
def _thumb(path, size):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    s = size / max(img.shape[:2])
    return cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                      interpolation=cv2.INTER_AREA)


def thumbnail(rome, i, size=160):
    """A small copy of photo i (long side size px), cached."""
    return _thumb(str(rome["images"][i]), size)


def label_name(label, rome=None):
    """Readable name of a hidden label (-1 is a stray) of the crate rome (default: the crate loaded last)."""
    groups = _GROUPS if rome is None else rome["groups"]
    return "stray" if label < 0 or label >= len(groups) else pretty(groups[label])


def _robust_subset(pts, max_points, seed=0):
    """Indices of pts without far strays (display only), at most max_points of them."""
    keep = np.isfinite(pts).all(axis=1)
    if keep.sum() > 10:
        med = np.median(pts[keep], axis=0)
        dist = np.linalg.norm(pts - med, axis=1)
        keep &= dist < 2.5 * np.percentile(dist[keep], 90)
    idx = np.flatnonzero(keep)
    if len(idx) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_points, replace=False))
    return idx


def _gt_points(root, group, max_points=60000):
    path = os.path.join(root, "gt", f"points_{group}.npz")
    if group is None or not os.path.exists(path):
        return np.zeros((0, 3)), None
    z = np.load(path)
    xyz = z["xyz"].astype(float)
    rgb = z["rgb"] if "rgb" in z else None
    idx = _robust_subset(xyz, max_points)
    return xyz[idx], None if rgb is None else rgb[idx]
