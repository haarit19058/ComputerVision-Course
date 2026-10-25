"""Checks for your 9 functions. Each test_<name>() runs your function (the one defined in the notebook) next to
the library version ref.<name> on the same input, plus cases with a known answer and edge cases (empty input,
ties, duplicates, degenerate geometry). Every line names its case. Nothing is graded."""

import contextlib
import functools

import numpy as np
import scipy.sparse as sp

from ._yours import NotDefinedYet, yours
from .geometry import rodrigues
from .reference import ref


def make_bags(n_images=200, n_words=1000, n_groups=4, stray_frac=0.2, seed=0):
    """Fake photos as bags of words: photos of one landmark share many of that landmark's words, strays do not.

    Returns (offsets (N+1,), words (P,), labels (N,), -1 for a stray), as in the crate.
    """
    rng = np.random.default_rng(seed)
    own = [rng.choice(n_words, 60, replace=False) for _ in range(n_groups)]
    labels = np.where(rng.random(n_images) < stray_frac, -1, rng.integers(0, n_groups, n_images))
    bags = []
    for lab in labels:
        m = int(rng.integers(20, 60))
        common = rng.zipf(1.6, m) % n_words                      # a few very common words, many rare ones
        if lab >= 0:
            mine = rng.choice(own[lab], int(0.6 * m))
            common = np.concatenate([common[: m - len(mine)], mine])
        bags.append(rng.permutation(common))
    offsets = np.concatenate([[0], np.cumsum([len(b) for b in bags])]).astype(np.int64)
    return offsets, np.concatenate(bags).astype(np.int64), labels.astype(np.int64)


def make_graph(n=60, n_groups=4, p_in=0.15, n_cross=3, n_alone=5, seed=0):
    """A random match graph: dense inside groups, a few edges between groups, a few photos with no edge at all.

    Returns (edges (E,2) int64 with i < j, weights (E,) int64 inlier counts; ties happen on purpose).
    """
    rng = np.random.default_rng(seed)
    group = rng.integers(0, n_groups, n)
    group[rng.choice(n, n_alone, replace=False)] = -1
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if group[i] >= 0 and group[i] == group[j] and rng.random() < p_in]
    cand = [(i, j) for i in range(n) for j in range(i + 1, n) if group[i] >= 0 and group[j] >= 0
            and group[i] != group[j]]
    pairs += [cand[k] for k in rng.choice(len(cand), min(n_cross, len(cand)), replace=False)]
    edges = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    return edges, rng.integers(15, 60, len(edges)).astype(np.int64)


def make_point_pairs(n=40, noise=0.0, seed=0):
    """Points X (n,3) and Y = s R X + t (+ Gaussian noise). Returns (X, Y, s, R, t)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) * [3.0, 2.0, 1.0]
    R = rodrigues(rng.normal(size=3))
    s, t = float(rng.uniform(0.3, 3.0)), rng.normal(size=3) * 5
    Y = s * X @ R.T + t + noise * rng.normal(size=(n, 3))
    return X, Y, s, R, t


def make_descriptors(n1=1500, n2=1800, n_shared=400, noise=4.0, seed=0):
    """Fake SIFT descriptors: d2 (n2,128), and d1 (n1,128) holding noisy copies of n_shared rows of d2 (shuffled)
    among unrelated rows. Values are whole numbers 0 .. 255 stored as float32, like SIFT.

    Returns (d1, d2, truth (n_shared,2) int64 pairs (i, j), sorted by i)."""
    rng = np.random.default_rng(seed)
    d2 = rng.integers(0, 256, (n2, 128)).astype(np.float32)
    d1 = rng.integers(0, 256, (n1, 128)).astype(np.float32)
    rows1 = np.sort(rng.choice(n1, n_shared, replace=False))
    rows2 = rng.choice(n2, n_shared, replace=False)
    d1[rows1] = np.clip(np.round(d2[rows2] + rng.normal(0, noise, (n_shared, 128))), 0, 255)
    return d1, d2, np.stack([rows1, rows2], axis=1).astype(np.int64)


def make_two_views(kind="depth", n=200, noise=0.5, outliers=0.0, seed=0):
    """Matches between two synthetic photos (640 x 480, focal 600 px) of n points.

    kind: "rotation" (the second camera only turns, no step), "planar" (a flat wall, the camera steps and turns) or
    "depth" (points spread in depth, the camera steps and turns). A share outliers of the matches is replaced by
    random pixels. Returns (x1 (n,2), x2 (n,2), K, R, t, outlier (n,) bool)."""
    rng = np.random.default_rng(seed)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    z = np.full(n, 8.0) if kind == "planar" else rng.uniform(4.0, 12.0, n)
    X = np.column_stack([rng.uniform(-0.38, 0.38, n) * z, rng.uniform(-0.28, 0.28, n) * z, z])
    R = rodrigues(np.array([0.03, 0.12, 0.02]))
    t = np.zeros(3) if kind == "rotation" else np.array([-1.2, 0.1, 0.3])

    def proj(Rc, tc):
        p = (X @ Rc.T + tc) @ K.T
        return p[:, :2] / p[:, 2:]
    x1 = proj(np.eye(3), np.zeros(3)) + rng.normal(0, noise, (n, 2))
    x2 = proj(R, t) + rng.normal(0, noise, (n, 2))
    bad = rng.choice(n, int(round(outliers * n)), replace=False)
    x2[bad] = rng.uniform([0, 0], [640, 480], (len(bad), 2))
    outlier = np.zeros(n, bool)
    outlier[bad] = True
    return x1, x2, K, R, t, outlier


def make_triangulation(n=50, noise=0.0, seed=0, step=(1.5, 0.2, 0.1)):
    """Two general cameras (each turned a little and moved, focal 500 to 900 px, 640 x 480 photos) and n points 6 to
    10 units in front of both, the second camera about step from the first (default: 1.5 units to the side; (0, 1.5,
    0) puts it above). noise is Gaussian pixel noise on the observations.
    Returns (P1 (3,4), P2 (3,4), x1 (n,2), x2 (n,2), X (n,3) the true points)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform([-2.0, -1.5, 6.0], [2.0, 1.5, 10.0], (n, 3))

    def camera(centre):
        f = rng.uniform(500, 900)
        K = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1.0]])
        R = rodrigues(rng.normal(0, 0.05, 3))
        return K @ np.hstack([R, (-R @ centre).reshape(3, 1)])

    P1 = camera(rng.normal(0, 0.2, 3))
    P2 = camera(np.asarray(step, float) + rng.normal(0, 0.2, 3))

    def seen(P):
        p = np.column_stack([X, np.ones(n)]) @ P.T
        return p[:, :2] / p[:, 2:] + rng.normal(0, noise, (n, 2))
    return P1, P2, seen(P1), seen(P2), X


_CHECK_LOG = []


_TODO_NAMES = ("tfidf_matrix", "top_k_neighbours", "match_descriptors", "query_expansion", "connected_components",
               "homography_dlt", "homography_ratio", "triangulate_points", "umeyama")


def check(name, cond, detail=""):
    """Print one PASS / FAIL line."""
    ok = bool(cond)
    _CHECK_LOG.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok or not detail else f"   ({detail})"))
    return ok


@contextlib.contextmanager
def _case(name):
    """One case of a test: if your function crashes on it, that is one FAIL line naming the case (with the error),
    and the test goes on with the next case. A function that is not written yet still stops the whole test."""
    try:
        yield
    except (NotImplementedError, NotDefinedYet):
        raise
    except Exception as e:  # noqa: BLE001  any crash is reported as a failed case
        check(name, False, f"raised {type(e).__name__}: {str(e)[:120]}")


def _raises_value_error(name, f, *args):
    """A case where f(*args) must raise ValueError."""
    try:
        f(*args)
    except (NotImplementedError, NotDefinedYet):
        raise
    except np.linalg.LinAlgError as e:  # a ValueError too, but a crash of the linear algebra, not a raised one
        return check(name, False, f"raised LinAlgError instead of ValueError: {str(e)[:100]}")
    except ValueError:
        return check(name, True)
    except Exception as e:  # noqa: BLE001
        return check(name, False, f"raised {type(e).__name__} instead of ValueError: {str(e)[:100]}")
    return check(name, False, "returned normally; it should raise ValueError")


def _friendly_test(test):
    """Wrap a test: a missing function becomes one friendly line, a real error still shows its traceback."""
    @functools.wraps(test)
    def run():
        target = test.__name__[len("test_"):]
        print(f"Checking {target} ...")
        start = len(_CHECK_LOG)
        try:
            test()
        except NotImplementedError as e:
            missing = str(e) or target
            extra = "" if missing == target else f" ({target} uses it)"
            print(f"  not implemented yet: {missing}{extra}. Write it, run its cell, then run this check again.")
            return
        except NotDefinedYet as e:
            print(f"  {str(e).rstrip('.')}, then run this check again.")
            return
        results = _CHECK_LOG[start:]
        if results and all(results):
            print(f"All {len(results)} checks passed.")
        elif results:
            print(f"{results.count(False)} of {len(results)} checks failed.")
    return run


def _dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M, float)


@_friendly_test
def test_tfidf_matrix():
    f = yours("tfidf_matrix")
    with _case("hand case"):
        # 3 photos: [w0, w0, w1], [w1, w2], [w2]; word 3 is never used
        T = f(np.array([0, 3, 5, 6]), np.array([0, 0, 1, 1, 2, 2]), 4)
        check("returns a scipy.sparse CSR matrix of shape (N, n_words)", sp.issparse(T) and T.format == "csr"
              and T.shape == (3, 4), f"got {type(T).__name__} {getattr(T, 'shape', None)}; .tocsr() at the end?")
        D = _dense(T)
        want = np.array([[0.94825, 0.31753, 0, 0], [0, 0.70711, 0.70711, 0], [0, 0, 1, 0]])
        check("hand case: tf = counts, idf = ln(N / n_w) + 1, unit rows", np.allclose(D, want, atol=1e-4),
              f"got\n{np.round(D, 4)}\nwant\n{want}")
        check("a word never used gives a column of zeros (no nan, no inf)", np.isfinite(D).all() and
              np.all(D[:, 3] == 0), f"column 3: {D[:, 3]}")
    with _case("a photo with no words"):
        offsets, words = np.array([0, 2, 2, 3]), np.array([5, 1, 5])          # photo 1 has no words at all
        D, Dr = _dense(f(offsets, words, 8)), _dense(ref.tfidf_matrix(offsets, words, 8))
        check("a photo with no words gives a row of zeros (no nan)", np.isfinite(D).all() and np.all(D[1] == 0),
              f"row 1: {D[1]}")
        check("... and it still counts in N (idf = ln(3 / n_w) + 1): the same matrix as sklearn",
              D.shape == Dr.shape and np.allclose(D, Dr, atol=1e-12), f"got\n{np.round(D[[0, 2]][:, [1, 5]], 4)}\n"
              f"want\n{np.round(Dr[[0, 2]][:, [1, 5]], 4)} (rows 0 and 2, words 1 and 5)")
    with _case("no words at all"):
        T = f(np.array([0, 0, 0]), np.zeros(0, dtype=np.int64), 5)
        check("no words in any photo: a (2, 5) matrix of zeros", T.shape == (2, 5) and np.all(_dense(T) == 0),
              f"got shape {T.shape}")
    with _case("one photo, list inputs, the last word id"):
        D = _dense(f([0, 3], [0, 1, 1], 2))
        check("one photo, plain lists, word n_words - 1: counts only (idf = 1)", D.shape == (1, 2) and
              np.allclose(D, [[1 / np.sqrt(5), 2 / np.sqrt(5)]]), f"got {np.round(D, 4).tolist()}")
    with _case("300 photos against sklearn"):
        offsets, words, _ = make_bags(n_images=300, n_words=2000, seed=1)
        T, Tr = f(offsets, words, 2000), ref.tfidf_matrix(offsets, words, 2000)
        norms = np.sqrt(np.asarray(sp.csr_matrix(T).multiply(sp.csr_matrix(T)).sum(axis=1)).ravel())
        check("every row has length 1", np.allclose(norms, 1.0),
              f"row lengths from {norms.min():.4f} to {norms.max():.4f}")
        gap = np.abs(_dense(T) - _dense(Tr)).max()
        check("300 photos: same matrix as sklearn's TfidfTransformer(smooth_idf=False)", gap < 1e-9,
              f"largest difference {gap:.2e}; raw counts for tf, natural log, + 1 in the idf?")


@_friendly_test
def test_top_k_neighbours():
    f = yours("top_k_neighbours")
    with _case("hand case"):
        T = sp.csr_matrix([[1.0, 0], [0.8, 0.6], [0, 1.0], [0.6, 0.8]])
        idx, score = f(T, 2)
        arrays = isinstance(idx, np.ndarray) and isinstance(score, np.ndarray)
        idx, score = np.asarray(idx), np.asarray(score)
        check("returns idx (N,k) int64 and score (N,k) float64 numpy arrays", arrays and
              idx.shape == (4, 2) and score.shape == (4, 2) and idx.dtype == np.int64 and score.dtype == np.float64,
              f"{idx.shape} {idx.dtype}, {score.shape} {score.dtype}" + ("" if arrays else
                                                                         "; return numpy arrays, not lists"))
        want = np.array([[1, 3], [3, 0], [3, 1], [1, 2]])
        check("hand case: 4 photos, k = 2, best first, never the photo itself", np.array_equal(idx, want) and
              np.allclose(score, [[0.8, 0.6], [0.96, 0.8], [0.8, 0.6], [0.96, 0.8]]), f"got idx {idx.tolist()}")
    with _case("ties and k = N - 1"):
        T = sp.csr_matrix(np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32))   # photos 0, 1, 2 alike
        idx, score = f(T, 2)
        idx, score = np.asarray(idx), np.asarray(score)
        check("ties: photo 0 gets both look-alikes 1 and 2 (any order), score 1 each",
              sorted(idx[0].tolist()) == [1, 2] and np.allclose(score[0], 1.0), f"got {idx[0].tolist()}")
        check("float32 input: scores still float64", score.dtype == np.float64, f"got {score.dtype}")
        idx, score = f(T, 3)
        idx, score = np.asarray(idx), np.asarray(score)
        ok = idx.shape == (4, 3) and all(sorted(idx[i].tolist()) == [j for j in range(4) if j != i] for i in range(4))
        check("k = N - 1 (its limit): every other photo exactly once", ok, f"got {idx.tolist()}")
    with _case("a photo with no words"):
        idx, score = f(sp.csr_matrix(np.array([[1.0, 0], [0, 0], [0.6, 0.8]])), 2)
        idx, score = np.asarray(idx), np.asarray(score)
        check("a photo with no words (a zero row): k neighbours, not itself, scores 0",
              1 not in idx[1].tolist() and np.allclose(score[1], 0) and idx[0, 0] == 2, f"got {idx.tolist()}")
    with _case("1500 photos against sklearn"):
        offsets, words, _ = make_bags(n_images=1500, n_words=3000, seed=2)       # more than one block of rows
        T = ref.tfidf_matrix(offsets, words, 3000)
        idx, score = f(T, 10)
        idx_r, score_r = ref.top_k_neighbours(T, 10)
        idx, score = np.asarray(idx), np.asarray(score)
        check("1500 photos: never the photo itself, scores in decreasing order",
              not np.any(idx == np.arange(1500)[:, None]) and np.all(np.diff(score, axis=1) <= 1e-12))
        check("1500 photos: same scores as sklearn's NearestNeighbors(metric='cosine')", np.allclose(score, score_r,
              atol=1e-9), f"largest difference {np.abs(score - score_r).max():.2e}")
        same = np.mean(idx == idx_r)
        check(f"... and the same neighbours ({100 * same:.1f}%, ties may swap)", same > 0.99)


def _clear_rows(d1, d2, ratio):
    """Rows of d1 whose ratio test is clear cut: the nearest squared distance is farther from ratio^2 times the
    second nearest than float32 rounding of |a|^2 + |b|^2 - 2 a.b can move it. Right at the ratio, rounding decides,
    so two correct matchers may differ."""
    a, b = np.asarray(d1, np.float64), np.asarray(d2, np.float64)
    sa, sb = np.sum(a * a, axis=1), np.sum(b * b, axis=1)
    dist2 = np.maximum(sa[:, None] + sb[None, :] - 2.0 * (a @ b.T), 0)
    two = np.partition(dist2, 1, axis=1)[:, :2]
    return np.abs(two[:, 0] - ratio ** 2 * two[:, 1]) > 2e-6 * (sa + sb.max())   # float32 rounding, with room


def _on_rows(m, rows):
    """The matches (i, j) of m whose row i is marked in the bool mask rows."""
    m = np.asarray(m, dtype=np.int64).reshape(-1, 2)
    return m[rows[m[:, 0]]]


@_friendly_test
def test_match_descriptors():
    f = yours("match_descriptors")
    rng = np.random.default_rng(10)
    with _case("hand case"):
        d2 = rng.integers(0, 200, (8, 128)).astype(np.float32)
        d1 = d2[[5, 0, 3]] + rng.integers(-2, 3, (3, 128))
        m = f(d1, d2)
        array, m = isinstance(m, np.ndarray), np.asarray(m)
        check("returns an (M,2) int64 numpy array", array and m.ndim == 2 and m.shape[1:] == (2,) and
              m.dtype == np.int64, f"got {m.shape} {m.dtype}" + ("" if array else "; return a numpy array, not a list"))
        check("hand case: 3 noisy copies of rows 5, 0, 3 give (0,5), (1,0), (2,3), sorted by i",
              np.array_equal(m, [[0, 5], [1, 0], [2, 3]]), f"got {m.tolist()}")
        m = np.asarray(f(d1.tolist(), d2.tolist()))
        check("plain lists work too", np.array_equal(m, [[0, 5], [1, 0], [2, 3]]), f"got {m.tolist()}")
    with _case("fewer than 2 descriptors"):
        d = rng.integers(0, 200, (6, 128)).astype(np.float32)
        out = [np.asarray(f(a, b)) for a, b in ((d[:1], d), (d, d[:1]), (d, np.zeros((0, 128), np.float32)))]
        check("fewer than 2 descriptors on either side: an empty (0, 2) int64 array",
              all(o.shape == (0, 2) and o.dtype == np.int64 for o in out), f"got shapes {[o.shape for o in out]}")
    with _case("uint8 input"):
        d1, d2, truth = make_descriptors(n1=60, n2=80, n_shared=25, seed=11)
        m8 = np.asarray(f(d1.astype(np.uint8), d2.astype(np.uint8)))
        m32 = np.asarray(f(d1, d2))
        check("uint8 input (as SIFT stores it): same matches as float32, all of them right",
              np.array_equal(m8, m32) and np.array_equal(m32, truth), f"uint8 gave {len(m8)} matches, float32 "
              f"{len(m32)}, want {len(truth)}; uint8 arithmetic wraps around: convert to float32 first")
    with _case("duplicate descriptors in d1"):
        d1, d2, _ = make_descriptors(n1=1500, n2=50, n_shared=0, seed=12)
        d1[3] = d1[1400] = d2[7]                                     # two copies, in different blocks of rows
        m = np.asarray(f(d1, d2))
        check("two copies of one descriptor in d1: only the first is matched (mutual check, ties to the smaller)",
              np.array_equal(m, [[3, 7]]), f"got {m.tolist()}")
    with _case("duplicate descriptors in d2"):
        d2 = rng.integers(0, 256, (12, 128)).astype(np.float32)
        d2[9] = d2[4]
        d1 = np.stack([d2[4] + rng.integers(-2, 3, 128), d2[1] + rng.integers(-2, 3, 128)])
        m = np.asarray(f(d1, d2))
        check("two copies of one descriptor in d2: ambiguous, so not matched; the other match stays",
              np.array_equal(m, [[1, 1]]), f"got {m.tolist()}")
    with _case("two equally close neighbours"):
        base = rng.integers(60, 190, 128).astype(np.float32)
        step = np.zeros(128, np.float32)
        step[:20] = 9
        d2 = np.stack([rng.integers(0, 256, 128), base + step, rng.integers(0, 256, 128), base - step]).astype(
            np.float32)
        m = np.asarray(f(np.stack([base, rng.integers(0, 256, 128).astype(np.float32)]), d2))
        check("a descriptor exactly as close to two others: rejected by the ratio test", len(m) == 0,
              f"got {m.tolist()}")
    with _case("ratio"):
        base = rng.uniform(40, 215, (150, 128)).astype(np.float32)         # 150 marks, each with a look-alike
        away = rng.normal(size=(150, 128))
        away *= rng.uniform(3, 40, (150, 1)) / np.linalg.norm(away, axis=1, keepdims=True)
        d2 = np.concatenate([base, base + away]).astype(np.float32)
        d1 = (base + rng.normal(0, 1.5, base.shape)).astype(np.float32)
        loose, strict = np.asarray(f(d1, d2, ratio=0.9)), np.asarray(f(d1, d2, ratio=0.6))
        inside = set(map(tuple, strict.tolist())) <= set(map(tuple, loose.tolist()))
        check(f"a stricter ratio keeps fewer matches ({len(strict)} at 0.6, {len(loose)} at 0.9), all among the "
              "looser ones", len(strict) < len(loose) and inside)
        for r, mine in ((0.6, strict), (0.9, loose)):
            theirs, clear = ref.match_descriptors(d1, d2, ratio=r), _clear_rows(d1, d2, r)
            check(f"ratio {r}: the same {len(theirs)} matches as OpenCV (distances compared, not squared distances)",
                  np.array_equal(_on_rows(mine, clear), _on_rows(theirs, clear)), f"got {len(mine)}")
    with _case("1500 x 1800 against OpenCV"):
        d1, d2, truth = make_descriptors(seed=14)
        m, m_r = np.asarray(f(d1, d2)), ref.match_descriptors(d1, d2)
        got, true = set(map(tuple, m.tolist())), set(map(tuple, truth.tolist()))
        found, false = len(got & true) / len(true), len(got - true)
        check(f"1500 x 1800 descriptors (more than one block): {found:.0%} of the {len(true)} true pairs found, "
              f"{false} false ones", found > 0.9 and false <= 2)
        clear = _clear_rows(d1, d2, 0.8)
        check(f"... the same {len(m_r)} matches as OpenCV's BFMatcher (knnMatch, ratio, mutual)",
              np.array_equal(_on_rows(m, clear), _on_rows(m_r, clear)), f"got {len(m)}")


@_friendly_test
def test_query_expansion():
    f = yours("query_expansion")
    with _case("hand cases"):
        edges = np.array([[0, 1], [1, 2], [2, 3], [5, 6]])
        tested = {(0, 1), (1, 2), (2, 3), (5, 6)}
        new = f(edges, tested)
        array, new = isinstance(new, np.ndarray), np.asarray(new)
        check("returns an (M,2) int64 numpy array", array and new.ndim == 2 and new.shape[1:] == (2,) and
              new.dtype == np.int64, f"got {new.shape} {new.dtype}" + ("" if array else
                                                                       "; return a numpy array, not a list"))
        check("hand case: path 0-1-2-3 gives (0,2) and (1,3)", new.dtype == np.int64 and
              np.array_equal(new, [[0, 2], [1, 3]]), f"got {new.tolist()} ({new.dtype})")
        new = np.asarray(f(edges, tested | {(0, 2)}))
        check("pairs already tested are left out", np.array_equal(new, [[1, 3]]), f"got {new.tolist()}")
        new = np.asarray(f(np.array([[0, 3], [0, 1], [0, 2]]), set()))
        check("star: every pair of photo 0's neighbours, i < k, sorted", np.array_equal(new, [[1, 2], [1, 3], [2, 3]]),
              f"got {new.tolist()}")
    with _case("empty results"):
        new = np.asarray(f(np.array([[0, 1], [2, 3]]), {(0, 1), (2, 3)}))
        check("nothing new gives shape (0, 2)", new.shape == (0, 2), f"got shape {new.shape}")
        new = np.asarray(f(np.zeros((0, 2), dtype=np.int64), set()))
        check("no edges at all gives shape (0, 2)", new.shape == (0, 2), f"got shape {new.shape}")
    with _case("edges written backwards, as a list"):
        new = np.asarray(f([[1, 0], [2, 1]], {(0, 1), (1, 2)}))
        check("edges as a list, written (j, i): still (0, 2) with i < k", np.array_equal(new, [[0, 2]]),
              f"got {new.tolist()}")
    with _case("duplicate edges and self loops"):
        new = np.asarray(f(np.array([[0, 1], [0, 1], [1, 2], [2, 2]]), {(0, 1), (1, 2)}))
        check("a repeated edge and a self loop: (0, 2) once, never (i, i)", np.array_equal(new, [[0, 2]]),
              f"got {new.tolist()}")
    with _case("random graph against scipy"):
        edges, _ = make_graph(n=80, seed=3)
        tested = set(map(tuple, edges[::2].tolist()))
        new, new_r = np.asarray(f(edges, tested)), ref.query_expansion(edges, tested)
        check(f"random graph: same {len(new_r)} pairs as A @ A with scipy.sparse", np.array_equal(new, new_r),
              f"got {len(new)} pairs")


@_friendly_test
def test_connected_components():
    f = yours("connected_components")
    with _case("hand cases"):
        lab = f(7, np.array([[0, 1], [2, 3], [3, 4]]))
        array, lab = isinstance(lab, np.ndarray), np.asarray(lab)
        check("returns labels (n,) int64, a numpy array", array and lab.shape == (7,) and lab.dtype == np.int64,
              f"{lab.shape} {lab.dtype}" + ("" if array else "; return a numpy array, not a list"))
        check("hand case: {2,3,4} is 0, {0,1} is 1, then the lone photos 5 and 6",
              np.array_equal(lab, [1, 1, 0, 0, 0, 2, 3]), f"got {lab.tolist()}")
        lab = np.asarray(f(4, np.array([[2, 3], [0, 1]])))
        check("equal sizes: the component with the smaller photo first", np.array_equal(lab, [0, 0, 1, 1]),
              f"got {lab.tolist()}")
        lab = np.asarray(f(4, np.array([[3, 0], [1, 2]])))
        check("equal sizes, edges (3, 0) and (1, 2): {0, 3} first (its smallest photo is 0), then {1, 2}",
              np.array_equal(lab, [0, 1, 1, 0]), f"got {lab.tolist()}; ties by the smallest member, not the root")
    with _case("no edges"):
        lab = np.asarray(f(3, np.zeros((0, 2), np.int64)))
        check("no edges: every photo alone", np.array_equal(lab, [0, 1, 2]), f"got {lab.tolist()}")
        lab = np.asarray(f(1, np.zeros((0, 2), np.int64)))
        check("a single photo and no edges: [0]", np.array_equal(lab, [0]), f"got {lab.tolist()}")
    with _case("isolated photos after the last edge"):
        lab = np.asarray(f(4, [[3, 2]]))
        check("edges as a list; photos 0 and 1 have no edge: each alone, after {2, 3}",
              np.array_equal(lab, [1, 2, 0, 0]), f"got {lab.tolist()}")
    with _case("duplicate edges and self loops"):
        lab = np.asarray(f(5, np.array([[0, 1], [1, 0], [0, 1], [3, 3]])))
        check("a repeated edge and a self loop change nothing", np.array_equal(lab, [0, 0, 1, 2, 3]),
              f"got {lab.tolist()}")
    with _case("random graph against scipy"):
        edges, _ = make_graph(n=200, n_groups=8, p_in=0.03, seed=4)
        lab, lab_r = np.asarray(f(200, edges)), ref.connected_components(200, edges)
        check(f"random graph: same {lab_r.max() + 1} components as scipy's csgraph.connected_components",
              np.array_equal(lab, lab_r), f"{lab.max() + 1} components; numbered by size?")


def _apply_h(H, x):
    p = np.column_stack([np.asarray(x, float), np.ones(len(x))]) @ np.asarray(H, float).T
    return p[:, :2] / p[:, 2:]


def _random_h(rng):
    return np.array([[1.0 + rng.normal(0, 0.1), rng.normal(0, 0.1), rng.normal(0, 40)],
                     [rng.normal(0, 0.1), 1.0 + rng.normal(0, 0.1), rng.normal(0, 40)],
                     [rng.normal(0, 2e-4), rng.normal(0, 2e-4), 1.0]])


@_friendly_test
def test_homography_dlt():
    f = yours("homography_dlt")
    rng = np.random.default_rng(20)
    H0 = _random_h(rng)
    with _case("exactly 4 points"):
        x1 = np.array([[10.0, 20], [600, 40], [580, 450], [30, 470]])
        H = f(x1, _apply_h(H0, x1))
        array, H = isinstance(H, np.ndarray), np.asarray(H, float)
        check("returns H (3,3), a numpy array, with H[2,2] = 1", array and H.shape == (3, 3) and
              abs(H[2, 2] - 1) < 1e-12, f"got shape {H.shape}, H[2,2] = {H[2, 2] if H.shape == (3, 3) else None}"
              + ("" if array else "; return a numpy array, not a list"))
        check("exactly 4 points: the exact H", np.allclose(H, H0, atol=1e-8), f"largest difference "
              f"{np.abs(H - H0).max():.2e}")
    with _case("50 points, no noise"):
        x1 = rng.uniform(0, 640, (50, 2))
        x2 = _apply_h(H0, x1)
        H, Hr = np.asarray(f(x1, x2), float), ref.homography_dlt(x1, x2)
        check("50 points, no noise: the exact H", np.allclose(H, H0, atol=1e-8), f"largest difference "
              f"{np.abs(H - H0).max():.2e}")
        gap = np.abs(_apply_h(H, x1) - _apply_h(Hr, x1)).max()
        check("... and the same mapping as OpenCV's findHomography(x1, x2, 0)", gap < 1e-3, f"{gap:.2e} px apart")
        H = np.asarray(f(x1.astype(np.float32).tolist(), x2.astype(np.float32)), float)
        check("float32 and list input work", np.abs(_apply_h(H, x1) - x2).max() < 0.01)
    with _case("degenerate input"):
        line = np.column_stack([np.linspace(0, 600, 6), 0.5 * np.linspace(0, 600, 6) + 40])
        _raises_value_error("6 points on one line: ValueError", f, line, _apply_h(H0, line))
        three = np.array([[0.0, 0], [100, 100], [250, 250], [50, 400]])
        _raises_value_error("4 points, 3 of them on one line: ValueError", f, three, _apply_h(H0, three))
        tri = np.array([[0.0, 0], [100, 0], [0, 100]])
        _raises_value_error("fewer than 4 matches: ValueError", f, tri, tri + 5)
        quad = np.array([[10.0, 20], [600, 40], [580, 450], [30, 470]])
        flat = np.array([[0.0, 0], [100, 100], [200, 200], [50, 400]])
        _raises_value_error("x1 a proper quadrilateral, 3 of the 4 points of x2 on one line (the only H that fits is "
                            "singular): ValueError", f, quad, flat)
    with _case("scale normalisation"):
        errs, errs_r = [], []
        for k in range(10):
            r = np.random.default_rng(30 + k)
            Hs = np.array([[1.0, 0.02, 15], [0.01, 1.0, 3], [0, 0, 1]])
            x1 = r.uniform(0, 600, (30, 2)) + 20000.0
            x2 = _apply_h(Hs, x1) + r.normal(0, 0.5, (30, 2))
            test = r.uniform(0, 600, (200, 2)) + 20000.0
            errs.append(np.median(np.linalg.norm(_apply_h(f(x1, x2), test) - _apply_h(Hs, test), axis=1)))
            errs_r.append(np.median(np.linalg.norm(_apply_h(ref.homography_dlt(x1, x2), test) - _apply_h(Hs, test),
                                                   axis=1)))
        e, er = np.median(errs), np.median(errs_r)
        check(f"noisy pixels near (20000, 20000): error {e:.2f} px, OpenCV {er:.2f} px (normalise the points!)",
              e < 1.5 * er + 0.05, "without hartley_normalize the matrix A is badly scaled")


@_friendly_test
def test_homography_ratio():
    f = yours("homography_ratio")
    with _case("pure rotation"):
        x1, x2, K, R, _, out_mask = make_two_views("rotation", outliers=0.15, seed=21)
        out = f(x1, x2)
        ratio, H, inl_H, inl_F = out
        arrays = all(isinstance(a, np.ndarray) for a in (H, inl_H, inl_F))
        inl_H, inl_F = np.asarray(inl_H), np.asarray(inl_F)
        check("returns (ratio, H (3,3), inliers_H (N,) bool, inliers_F (N,) bool), numpy arrays", arrays and
              len(out) == 4 and np.ndim(ratio) == 0 and np.shape(H) == (3, 3) and inl_H.shape == (200,) and
              inl_F.shape == (200,) and inl_H.dtype == bool and inl_F.dtype == bool,
              f"got {type(ratio).__name__}, {np.shape(H)}, {inl_H.shape} {inl_H.dtype}, {inl_F.shape} {inl_F.dtype}"
              + ("" if arrays else "; return numpy arrays, not lists"))
        check(f"camera only turns (no step), 15% outliers: ratio {float(ratio):.2f}, near 1", ratio > 0.9)
        gap = np.median(np.linalg.norm(_apply_h(H, x1[inl_H]) - _apply_h(K @ R @ np.linalg.inv(K), x1[inl_H]),
                                       axis=1))
        check(f"... H is K R K^-1 (median {gap:.2f} px off)", gap < 1.0)
        n_bad = int(np.sum(inl_H & out_mask))
        check(f"... the 30 random matches are left out ({n_bad} of them counted as H inliers), the true ones kept "
              f"({np.sum(inl_H & ~out_mask)} of 170)", n_bad <= 3 and np.sum(inl_H & ~out_mask) >= 160)
    with _case("the 2 px threshold"):
        rng = np.random.default_rng(29)
        x1 = rng.uniform([0, 0], [640, 480], (200, 2))
        x2 = _apply_h(_random_h(rng), x1)
        off = np.arange(200) >= 150                # the last 50 matches sit 3.5 px off, 50 at 1.6 px, 100 near 0.3 px
        near = np.arange(200) < 100
        turn = rng.uniform(0, 2 * np.pi, 100)
        ring = np.column_stack([np.cos(turn), np.sin(turn)])
        x2[off] += 3.5 * ring[50:]
        x2[~off & ~near] += 1.6 * ring[:50]
        x2[near] += rng.normal(0, 0.3, (100, 2))
        inl_H = np.asarray(f(x1, x2)[2])
        check(f"default thresh = 2 px: matches 0.3 and 1.6 px off are H inliers ({np.sum(inl_H & ~off)} of 150), "
              f"matches 3.5 px off are not ({np.sum(inl_H & off)} of 50 counted)", np.sum(inl_H & ~off) >= 140 and
              np.sum(inl_H & off) <= 2, "compare the distance itself with thresh (not its square), in pixels")
        inl_4 = np.asarray(f(x1, x2, thresh=4.0)[2])
        check(f"thresh = 4 px (the pipeline uses it): now the 3.5 px matches count too ({np.sum(inl_4 & off)} of 50)",
              np.sum(inl_4 & off) >= 40, "use the thresh argument, not a fixed 2")
    with _case("flat wall"):
        x1, x2 = make_two_views("planar", seed=22)[:2]
        ratio = float(f(x1, x2)[0])
        check(f"a flat wall seen from two stations: ratio {ratio:.2f}, near 1 as well", ratio > 0.9)
    with _case("depth and a step"):
        x1, x2 = make_two_views("depth", outliers=0.1, seed=23)[:2]
        ratio, _, inl_H, inl_F = f(x1, x2)
        check(f"a 3D scene with depth and a step: ratio {float(ratio):.2f}, low (F explains {np.sum(inl_F)}, "
              f"H only {np.sum(inl_H)})", ratio < 0.3)
    with _case("all outliers"):
        rng = np.random.default_rng(24)
        x1, x2 = rng.uniform(0, 640, (200, 2)), rng.uniform(0, 480, (200, 2))
        ratio, _, inl_H, inl_F = f(x1, x2)
        check(f"all outliers: few inliers for H ({np.sum(inl_H)}) and F ({np.sum(inl_F)}), ratio a finite number",
              np.sum(inl_H) < 20 and np.sum(inl_F) < 40 and np.isfinite(ratio))
        again = f(x1, x2)                      # here a different sample nearly always gives different inliers
        check("the same seed gives the same result", float(again[0]) == float(ratio) and
              np.array_equal(np.asarray(again[2]), np.asarray(inl_H)),
              "draw the samples with np.random.default_rng(seed)")
    with _case("no homography at all"):
        u = np.random.default_rng(25).uniform(0, 600, 40)
        x1, x2 = np.column_stack([u, 0.5 * u + 30]), np.column_stack([1.1 * u + 5, 0.4 * u + 60])
        ratio, H, inl_H, inl_F = f(x1, x2)
        inl_H = np.asarray(inl_H)
        r_ref, H_ref = ref.homography_ratio(x1, x2)[:2]
        check("all matches on one line (every sample of 4 is degenerate): H is None, inliers_H all False, ratio 0",
              H is None and inl_H.shape == (40,) and inl_H.dtype == bool and not inl_H.any() and float(ratio) == 0
              and H_ref is None and r_ref == 0, f"got H {None if H is None else 'a matrix'}, "
              f"{int(inl_H.sum())} H inliers, ratio {float(ratio):.2f}")
    with _case("against OpenCV"):
        mine, theirs = [], []
        for kind, seed in (("rotation", 26), ("planar", 27), ("depth", 28)):
            x1, x2 = make_two_views(kind, outliers=0.1, seed=seed)[:2]
            mine.append(float(f(x1, x2)[0]))
            theirs.append(float(ref.homography_ratio(x1, x2)[0]))
        same = abs(mine[0] - theirs[0]) < 0.1 and abs(mine[1] - theirs[1]) < 0.1 and max(mine[2], theirs[2]) < 0.3
        check("rotation, wall, depth: ratios " + ", ".join(f"{a:.2f}" for a in mine) + " vs OpenCV's "
              + ", ".join(f"{b:.2f}" for b in theirs), same)


@_friendly_test
def test_umeyama():
    f = yours("umeyama")
    with _case("noiseless"):
        X, Y, s0, R0, t0 = make_point_pairs(n=30, seed=6)
        s, R, t = f(X, Y)
        arrays = isinstance(R, np.ndarray) and isinstance(t, np.ndarray)
        R, t = np.asarray(R, float), np.asarray(t, float)
        check("returns (s, R (3,3), t (3,)), R and t numpy arrays", arrays and np.ndim(s) == 0 and
              R.shape == (3, 3) and t.shape == (3,),
              f"{np.shape(s)}, {R.shape}, {t.shape}" + ("" if arrays else "; return numpy arrays, not lists"))
        check("noiseless: the exact s, R and t", abs(s - s0) < 1e-9 and np.allclose(R, R0, atol=1e-9) and
              np.allclose(t, t0, atol=1e-9), f"s {float(s):.4f} (want {s0:.4f}), R off by "
              f"{np.abs(R - R0).max():.2g}, t off by {np.abs(t - t0).max():.2g}")
        check("Y = s R X + t: apply it as s * X @ R.T + t", np.allclose(s * X @ R.T + t, Y, atol=1e-8))
    with _case("the mirror case"):
        X, _, _, _, _ = make_point_pairs(n=30, seed=6)
        s, R, t = f(X, X * [1, 1, -1])                                      # a mirror image
        R, t = np.asarray(R, float), np.asarray(t, float)
        check("a mirror image still gives a rotation (det R = +1)", abs(np.linalg.det(R) - 1) < 1e-9,
              f"det R = {np.linalg.det(R):.3f}; the sign matrix E?")
        sr, Rr, tr = ref.umeyama(X, X * [1, 1, -1])
        check("... and the best one: the same s, R, t as skimage", abs(s - sr) < 1e-6 and
              np.allclose(R, Rr, atol=1e-6) and np.allclose(t, tr, atol=1e-6),
              f"s {float(s):.4f} (want {sr:.4f}); E in the scale too, s = trace(diag(D) E) / ...?")
    with _case("3 points"):
        X, Y, s0, R0, t0 = make_point_pairs(n=3, seed=8)
        s, R, t = f(X, Y)
        R, t = np.asarray(R, float), np.asarray(t, float)
        check("3 points (the fewest): the exact s, R, t", abs(s - s0) < 1e-8 and np.allclose(R, R0, atol=1e-8) and
              np.allclose(t, t0, atol=1e-8), f"s off by {abs(s - s0):.2g}, R by {np.abs(R - R0).max():.2g}, "
              f"t by {np.abs(t - t0).max():.2g}")
    with _case("collinear points"):
        X = np.column_stack([np.linspace(-2, 3, 10), 2 * np.linspace(-2, 3, 10), np.zeros(10)])
        R0 = rodrigues(np.array([0.3, -0.2, 0.5]))
        Y = 2.0 * X @ R0.T + [1, 2, 3]
        s, R, t = f(X, Y)
        R = np.asarray(R, float)
        check("points on one line (the turn about the line is free): still s = 2, det R = +1, X lands on Y",
              abs(s - 2) < 1e-8 and abs(np.linalg.det(R) - 1) < 1e-8 and np.allclose(s * X @ R.T + t, Y, atol=1e-8),
              f"s {float(s):.4f}, det R {np.linalg.det(R):.3f}")
    with _case("float32 and list input"):
        X, Y, s0, _, _ = make_point_pairs(n=20, seed=9)
        s, R, t = f(X.astype(np.float32).tolist(), Y.astype(np.float32))
        check("float32 and list input work", abs(s - s0) < 1e-4 and np.allclose(s * X @ np.asarray(R).T + t, Y,
              atol=1e-3))
    with _case("noisy points against skimage"):
        X, Y, _, _, _ = make_point_pairs(n=50, noise=0.3, seed=7)
        s, R, t = f(X, Y)
        sr, Rr, tr = ref.umeyama(X, Y)
        check("noisy points: same s, R, t as skimage's SimilarityTransform", abs(s - sr) < 1e-6 and
              np.allclose(R, Rr, atol=1e-6) and np.allclose(t, tr, atol=1e-6),
              f"s {float(s):.5f} vs {sr:.5f}; scale from the singular values and the variance of X?")


@_friendly_test
def test_triangulate_points():
    f = yours("triangulate_points")
    with _case("hand case"):
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])                 # camera 1 at the origin, focal 1
        P2 = np.hstack([np.eye(3), np.array([[-1.0], [0], [0]])])     # camera 2 one unit to the right
        X0 = np.array([[0.5, -0.3, 5.0]])
        out = f(P1, P2, [[0.1, -0.06]], [[0.1 - 0.2, -0.06]])
        array, out = isinstance(out, np.ndarray), np.asarray(out, float)
        check("returns the points (N,3) as a numpy array of floats", array and out.shape == (1, 3),
              f"got shape {out.shape}" + ("" if array else "; return a numpy array, not a list"))
        check("hand case: the point 5 units in front, seen 0.2 apart, lands at (0.5, -0.3, 5)",
              np.allclose(out, X0, atol=1e-9), f"got {np.round(out, 4).tolist()}; rows u P[2] - P[0] and "
              "v P[2] - P[1]? divided by the fourth coordinate?")
    with _case("exact correspondences"):
        P1, P2, x1, x2, X = make_triangulation(n=50, seed=1)
        out = np.asarray(f(P1, P2, x1, x2), float)
        err = np.linalg.norm(out - X, axis=1).max() if out.shape == X.shape else np.inf
        check(f"50 exact matches, general cameras: every point recovered (worst error {err:.2g})", err < 1e-6,
              f"shape {out.shape}; the last row of Vt is the solution of A X = 0")
    with _case("any scale of P"):
        out = np.asarray(f(3.7 * P1, -0.5 * P2, x1, x2), float)
        check("P1 and P2 at another scale (even negative) give the same points: x ~ P X is up to scale",
              out.shape == X.shape and np.allclose(out, X, atol=1e-6), "divide by the fourth coordinate itself")
    with _case("list input"):
        out = np.asarray(f(P1.tolist(), P2.tolist(), x1[:5].tolist(), x2[:5].tolist()), float)
        check("lists work as input", out.shape == (5, 3) and np.allclose(out, X[:5], atol=1e-6))
    with _case("noisy observations against OpenCV"):
        P1, P2, x1, x2, X = make_triangulation(n=200, noise=0.5, seed=2)
        out = np.asarray(f(P1, P2, x1, x2), float)
        ref_out = ref.triangulate_points(P1, P2, x1, x2)
        mine = np.median(np.linalg.norm(out - X, axis=1)) if out.shape == X.shape else np.inf
        theirs = np.median(np.linalg.norm(ref_out - X, axis=1))
        check(f"0.5 px noise: median 3D error {mine:.4f}, as accurate as cv2.triangulatePoints ({theirs:.4f})",
              mine <= 1.1 * theirs + 1e-9, "the same DLT rows and the SVD solution?")
        p = np.column_stack([out, np.ones(len(out))]) @ P1.T if out.shape == X.shape else np.full((1, 3), np.nan)
        reproj = np.median(np.linalg.norm(p[:, :2] / p[:, 2:] - x1, axis=1))
        check(f"... and the points reproject onto their pixels in photo 1 (median {reproj:.2f} px, noise 0.5 px)",
              reproj < 1.0)
    with _case("noisy, the second camera above and diagonal"):
        worst = []
        for step, seed in (((0.2, 1.5, 0.1), 4), ((1.1, 1.1, 0.1), 5)):    # now the v rows carry the depth too
            P1, P2, x1, x2, X = make_triangulation(n=200, noise=0.5, seed=seed, step=step)
            out = np.asarray(f(P1, P2, x1, x2), float)
            mine = np.median(np.linalg.norm(out - X, axis=1)) if out.shape == X.shape else np.inf
            theirs = np.median(np.linalg.norm(ref.triangulate_points(P1, P2, x1, x2) - X, axis=1))
            worst.append(mine / theirs)
        check(f"0.5 px noise, camera 2 above and diagonal: as accurate as OpenCV (error {max(worst):.2f} times "
              "OpenCV's)", max(worst) <= 1.1, "all four rows of A, two per camera (u and v)?")
    with _case("many points"):
        P1, P2, x1, x2, X = make_triangulation(n=2000, seed=3)
        out = np.asarray(f(P1, P2, x1, x2), float)
        worst = np.abs(out - X).max() if out.shape == X.shape else np.inf
        check("2,000 points: shape (2000, 3), all recovered", worst < 1e-6,
              f"shape {out.shape}" + (f", worst error {worst:.2g}" if np.isfinite(worst) else ""))


def run_all_tests():
    """Run every check in order."""
    for test in (test_tfidf_matrix, test_top_k_neighbours, test_match_descriptors, test_query_expansion,
                 test_connected_components, test_homography_dlt, test_homography_ratio, test_triangulate_points,
                 test_umeyama):
        test()
        print()
