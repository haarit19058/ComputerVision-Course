"""Library versions of the 9 student functions: same inputs, same outputs (types, shapes, conventions).

They are available as ref.<name> (ref.tfidf_matrix, ...) after `from toolbox import *`. The checks compare your
function with them, and you can borrow one to keep going: tfidf_matrix = ref.tfidf_matrix. They only use numpy,
scipy, sklearn, skimage, OpenCV and the provided helpers, never your functions."""

import types

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as _cc_scipy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
import skimage.transform
import cv2



def tfidf_matrix(offsets, words, n_words):
    """sklearn's TfidfTransformer (idf = ln(N / n_w) + 1, unit rows) on the (N, n_words) matrix of word counts."""
    offsets = np.asarray(offsets, dtype=np.int64)
    rows = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    C = sp.csr_matrix((np.ones(len(rows)), (rows, np.asarray(words, dtype=np.int64))),
                      shape=(len(offsets) - 1, n_words))
    C.sum_duplicates()
    if C.shape[0] == 0:                                  # no photos at all (sklearn wants at least one)
        return sp.csr_matrix((0, n_words), dtype=np.float64)
    with np.errstate(divide="ignore"):                   # words in no photo: their idf is never used
        T = TfidfTransformer(smooth_idf=False, norm="l2").fit_transform(C)
    return sp.csr_matrix(T, dtype=np.float64)


def top_k_neighbours(T, k):
    """sklearn's brute force cosine NearestNeighbors with k + 1 neighbours; the image itself is dropped."""
    T = sp.csr_matrix(T)
    N = T.shape[0]
    dist, nb = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(T).kneighbors(T)
    idx = np.zeros((N, k), dtype=np.int64)
    score = np.zeros((N, k))
    for i in range(N):
        keep = np.flatnonzero(nb[i] != i)[:k]          # the image itself is usually first, but not always (ties)
        idx[i], score[i] = nb[i, keep], 1.0 - dist[i, keep]
    return idx, score


def match_descriptors(d1, d2, ratio=0.8):
    """OpenCV's brute force matcher (L2): knnMatch(d1, d2, k=2) and the ratio test, then match(d2, d1) for the
    mutual check. Ties: OpenCV keeps the first (smallest) index, as yours should. It is about 3 times slower than a
    NumPy version built on one matrix product: OpenCV's brute force matcher does not use a BLAS matrix product, and it
    runs twice (knnMatch forward, match backward for the mutual check)."""
    d1, d2 = np.asarray(d1, dtype=np.float32), np.asarray(d2, dtype=np.float32)
    if min(len(d1), len(d2)) < 2 or d1.ndim != 2 or d2.ndim != 2:            # knnMatch needs 2 neighbours
        return np.empty((0, 2), np.int64)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    back = np.zeros(len(d2), dtype=np.int64)
    for m in bf.match(d2, d1):
        back[m.queryIdx] = m.trainIdx
    pairs = [(a.queryIdx, a.trainIdx) for a, b in bf.knnMatch(d1, d2, k=2)
             if a.distance < ratio * b.distance and back[a.trainIdx] == a.queryIdx]
    return np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def query_expansion(edges, tested):
    """(A @ A)[i, k] > 0 means i and k share a neighbour: keep i < k, drop the pairs already tested."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    done = np.array(sorted(tested), dtype=np.int64).reshape(-1, 2)
    n = int(max(edges.max(), done.max(initial=-1))) + 1      # big enough for every photo in edges and in tested
    A = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64)
    A2 = sp.triu(A @ A, k=1).tocoo()
    pairs = np.stack([A2.row, A2.col], axis=1).astype(np.int64)
    if len(done):                                            # pair (i, k) <-> the number i n + k, no collisions
        pairs = pairs[~np.isin(pairs[:, 0] * n + pairs[:, 1], done[:, 0] * n + done[:, 1])]
    return np.unique(pairs, axis=0).reshape(-1, 2)


def connected_components(n, edges):
    """scipy's connected_components, labels renumbered: largest component first, ties by smallest member."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    A = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, lab = _cc_scipy(A, directed=False)
    size = np.bincount(lab)
    _, smallest = np.unique(lab, return_index=True)          # scipy numbers components 0, 1, ... with no gaps
    order = sorted(range(len(size)), key=lambda c: (-size[c], smallest[c]))
    new_label = np.zeros(len(size), dtype=np.int64)
    new_label[order] = np.arange(len(size))
    return new_label[lab]


def homography_dlt(x1, x2):
    """cv2.findHomography(x1, x2, 0): least squares on all matches (OpenCV polishes the DLT a little), H[2,2] = 1.
    OpenCV returns some H even when the matches fix no proper one, so those cases are caught first (ValueError):
    when either photo has fewer than 4 distinct points, or all but at most one of its distinct points lie on one line
    (3 of 4 on a line, all but one on a line, a repeated point), the matches leave more than one H, or only a singular
    one."""
    def all_but_one_on_a_line(x, rel=1e-6):
        """True when the distinct points of x (N,2) are fewer than 4, or lie on one line once at most one of them is
        left out. Such points cannot fix a proper homography: OpenCV would still return some H."""
        x = np.unique(x, axis=0)
        if len(x) < 4:
            return True
        x = x - x.mean(axis=0)
        full = np.cov(x.T, bias=True)
        n = len(x)
        if n < 3:
            return True
        # the 2x2 covariance with each point k left out, from running sums (one row per k), plus the full set
        s, q = x.sum(axis=0), np.einsum("ni,nj->ij", x, x)
        m = (s - x) / (n - 1)
        cov = (q - np.einsum("ni,nj->nij", x, x)) / (n - 1) - np.einsum("ni,nj->nij", m, m)
        cov = np.concatenate([full[None], cov])
        a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
        half, root = (a + c) / 2, np.sqrt(((a - c) / 2) ** 2 + b ** 2)
        small, big = np.maximum(half - root, 0.0), half + root           # the two variances: across and along a line
        return bool(np.any(np.sqrt(small) <= rel * np.sqrt(np.maximum(big, 1e-300))))

    x1 = np.asarray(x1, dtype=np.float64).reshape(-1, 2)
    x2 = np.asarray(x2, dtype=np.float64).reshape(-1, 2)
    if len(x2) != len(x1) or len(x1) < 4:
        raise ValueError(f"a homography needs at least 4 matches ({len(x1)} and {len(x2)} points given)")
    if all_but_one_on_a_line(x1) and all_but_one_on_a_line(x2):
        raise ValueError("no unique homography: more than one fits (points on one line, or only 3 distinct?)")
    if all_but_one_on_a_line(x1) or all_but_one_on_a_line(x2):
        raise ValueError("no proper homography: the only one that fits is singular")
    H, _ = cv2.findHomography(x1, x2, 0)
    if H is None or abs(H[2, 2]) < 1e-12:
        raise ValueError("OpenCV found no homography")
    return H / H[2, 2]


def homography_ratio(x1, x2, thresh=2.0, iters=500, seed=0):
    """cv2.findHomography with RANSAC next to cv2.findFundamentalMat with RANSAC, same threshold and rounds;
    ratio = H inliers / max(F inliers, 1). No homography (H None, no H inliers) when the points lie on one line."""
    x1 = np.asarray(x1, dtype=np.float64).reshape(-1, 2)
    x2 = np.asarray(x2, dtype=np.float64).reshape(-1, 2)
    if len(x1) < 8 or len(x2) != len(x1):
        raise ValueError(f"homography_ratio needs 8 or more matches, got {len(x1)}")
    flat = [np.linalg.svd(x - x.mean(axis=0), compute_uv=False) for x in (x1, x2)]
    cv2.setRNGSeed(int(seed))
    if any(sv[-1] <= 1e-8 * max(sv[0], 1e-300) for sv in flat):     # all on one line: no homography at all
        H, mask_H = None, None                                        # (OpenCV would still return one)
    else:
        H, mask_H = cv2.findHomography(x1, x2, cv2.RANSAC, float(thresh), maxIters=int(iters))
    _, mask_F = cv2.findFundamentalMat(x1, x2, cv2.FM_RANSAC, float(thresh), 0.999, int(iters))
    inl_H = np.zeros(len(x1), bool) if mask_H is None else mask_H.ravel().astype(bool)
    inl_F = np.zeros(len(x1), bool) if mask_F is None else mask_F.ravel().astype(bool)
    H = None if H is None else H / H[2, 2]
    return float(inl_H.sum() / max(inl_F.sum(), 1)), H, inl_H, inl_F


def triangulate_points(P1, P2, pts1, pts2):
    """cv2.triangulatePoints: the same linear (DLT) triangulation, then each homogeneous column divided by its
    fourth coordinate."""
    P1, P2 = np.asarray(P1, dtype=np.float64), np.asarray(P2, dtype=np.float64)
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    if len(pts1) == 0:
        return np.zeros((0, 3))
    Xh = cv2.triangulatePoints(P1, P2, pts1.T.copy(), pts2.T.copy())
    return (Xh[:3] / Xh[3]).T.astype(np.float64)


def umeyama(X, Y):
    """skimage's 3D SimilarityTransform from X to Y; its 4x4 matrix is [[s R, t], [0, 1]].

    For points on one line (or one plane) skimage may return a mirror (det < 0). A mirror in a plane that holds all
    the centred points, (I - 2 n n^T) with n normal to them, leaves those points where they are, so A (I - 2 n n^T)
    fits exactly as well and is s times a true rotation; t is recomputed for it.
    """
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if hasattr(skimage.transform.SimilarityTransform, "from_estimate"):     # skimage >= 0.26
        M = skimage.transform.SimilarityTransform.from_estimate(X, Y).params
    else:
        tf = skimage.transform.SimilarityTransform(dimensionality=3)
        tf.estimate(X, Y)
        M = tf.params
    A = M[:3, :3]
    if np.linalg.det(A) < 0:
        n = np.linalg.svd(X - X.mean(axis=0))[2][-1]       # the direction the centred points do not spread in
        A = A @ (np.eye(3) - 2.0 * np.outer(n, n))
    s = float(np.cbrt(np.linalg.det(A)))
    return s, A / s, Y.mean(axis=0) - A @ X.mean(axis=0)


# ref.<name>: drop in for your function of the same name, e.g. tfidf_matrix = ref.tfidf_matrix
ref = types.SimpleNamespace(tfidf_matrix=tfidf_matrix, top_k_neighbours=top_k_neighbours, match_descriptors=match_descriptors, query_expansion=query_expansion, connected_components=connected_components, homography_dlt=homography_dlt, homography_ratio=homography_ratio, triangulate_points=triangulate_points, umeyama=umeyama)
