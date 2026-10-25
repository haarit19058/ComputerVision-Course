"""Two view geometry, triangulation, PnP and bundle adjustment, solved and ready to use (NumPy, with
OpenCV versions where speed matters). Every function takes one K per photo."""

import numpy as np
import cv2
import scipy.sparse as sp
from scipy.optimize import least_squares


def to_h(x):
    """Append a column of ones: (N,d) -> (N,d+1) homogeneous coordinates."""
    x = np.asarray(x, dtype=float)
    return np.hstack([x, np.ones((len(x), 1))])


def skew(v):
    """Cross product matrix [v]x, so that skew(v) @ w == np.cross(v, w)."""
    v = np.asarray(v, dtype=float).ravel()
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def nearest_rotation(M):
    """The rotation matrix closest to a 3x3 matrix M (Frobenius norm): U diag(1, 1, det(U V^T)) V^T."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def rodrigues(r):
    """Axis angle vector r (..., 3) to rotation matrix (..., 3, 3). Works on a single vector or a batch."""
    r = np.asarray(r, dtype=float)
    th = np.linalg.norm(r, axis=-1)
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1.0 - th ** 2 / 6.0, np.sin(ths) / ths)
    b = np.where(small, 0.5 - th ** 2 / 24.0, (1.0 - np.cos(ths)) / ths ** 2)
    S = np.zeros(r.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2], S[..., 1, 2] = -r[..., 2], r[..., 1], -r[..., 0]
    S[..., 1, 0], S[..., 2, 0], S[..., 2, 1] = r[..., 2], -r[..., 1], r[..., 0]
    return np.eye(3) + a[..., None, None] * S + b[..., None, None] * (S @ S)


def rodrigues_inv(R):
    """Rotation matrix (3,3) to axis angle vector (3,), with |r| in [0, pi]. Inverse of rodrigues."""
    R = np.asarray(R, dtype=float)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])     # 2 sin(th) k
    s, c = np.linalg.norm(v) / 2.0, (np.trace(R) - 1.0) / 2.0
    th = np.arctan2(s, c)                                  # precise at every angle (arccos is not near 0 and pi)
    if s > 1e-7:
        return th * v / (2.0 * s)
    if c > 0:                                              # th near 0: r = v / 2
        return 0.5 * v
    S = (R + R.T) / 2.0                                    # th near pi: S = cos(th) I + (1 - cos(th)) k k^T
    B = (S - c * np.eye(3)) / (1.0 - c)
    k = B[:, np.argmax(np.diag(B))]
    k = k / np.linalg.norm(k)
    return th * (k if k @ v >= 0 else -k)


def _two_K(K):
    """K (3,3) for both photos of a pair, or (2,3,3) = [K1, K2]. Returns (K1, K2)."""
    K = np.asarray(K, float)
    return (K, K) if K.ndim == 2 else (K[0], K[1])


def projection_matrix(K, R, t):
    """P = K [R | t], shape (3,4)."""
    return np.asarray(K, float) @ np.hstack([np.asarray(R, float), np.asarray(t, float).reshape(3, 1)])


def project(X, R, t, K):
    """Project world points X (N,3) into a camera. Returns (uv (N,2) pixels, depth (N,) along the optical axis)."""
    Xc = np.asarray(X, float) @ np.asarray(R, float).T + np.asarray(t, float).reshape(3)
    p = Xc @ np.asarray(K, float).T
    z = np.where(np.abs(p[:, 2]) < 1e-12, 1e-12, p[:, 2])
    return p[:, :2] / z[:, None], Xc[:, 2]


def reprojection_error(X, R, t, K, x):
    """Distance in pixels between the projection of X (N,3) and the observed pixels x (N,2). inf behind the camera."""
    uv, depth = project(X, R, t, K)
    err = np.linalg.norm(uv - np.asarray(x, float), axis=1)
    return np.where(depth > 0, err, np.inf)


def camera_center(R, t):
    """Camera centre in world coordinates, C = -R^T t."""
    return -np.asarray(R, float).T @ np.asarray(t, float).reshape(3)


def pixel_colors(img, x):
    """RGB colour (N,3) uint8 of an image at pixels x (N,2), rounded to the nearest pixel."""
    x = np.asarray(x, float)
    h, w = img.shape[:2]
    return img[np.clip(np.round(x[:, 1]).astype(int), 0, h - 1), np.clip(np.round(x[:, 0]).astype(int), 0, w - 1)]


def ray_angles(X, C1, C2):
    """Angle in degrees, at each point X (N,3), between the rays to the camera centres C1 and C2 (parallax)."""
    a = np.asarray(X, float) - C1
    b = np.asarray(X, float) - C2
    cos = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def hartley_normalize(pts):
    """Move points (N,2) so their centroid is the origin and their mean distance to it is sqrt(2).
    Returns (pts_n (N,2), T (3,3)) with T @ [p, 1] = [p_n, 1]."""
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0)
    s = np.sqrt(2) / max(np.linalg.norm(pts - c, axis=1).mean(), 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    return s * (pts - c), T


def essential_from_F(F, K):
    """Essential matrix from F: E = K2^T F K1 (K is (3,3) for both photos or [K1, K2]), singular values -> (s, s, 0)."""
    K1, K2 = _two_K(K)
    U, S, Vt = np.linalg.svd(K2.T @ np.asarray(F, float) @ K1)
    s = (S[0] + S[1]) / 2.0
    return U @ np.diag([s, s, 0.0]) @ Vt


def fundamental_from_poses(K1, R1, t1, K2, R2, t2):
    """F between two cameras with known poses, unit norm: F = K2^-T [t]x R K1^-1 with R = R2 R1^T, t = t2 - R t1."""
    R = np.asarray(R2, float) @ np.asarray(R1, float).T
    t = np.asarray(t2, float).reshape(3) - R @ np.asarray(t1, float).reshape(3)
    F = np.linalg.inv(K2).T @ skew(t) @ R @ np.linalg.inv(K1)
    return F / np.linalg.norm(F)


def decompose_E(E):
    """The 4 poses (R, t) that fit an essential matrix E = [t]x R, as cv2.decomposeEssentialMat.
    Returns [(R1, t), (R1, -t), (R2, t), (R2, -t)] with R1 = U W V^T, R2 = U W^T V^T, |t| = 1."""
    U, _, Vt = np.linalg.svd(np.asarray(E, float))
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R1, R2, t = U @ W @ Vt, U @ W.T @ Vt, U[:, 2].copy()
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def eight_point(x1, x2):
    """Normalised 8 point algorithm: F (3,3), rank 2, unit norm, from N >= 8 matches (cv2.FM_8POINT)."""
    n1, T1 = hartley_normalize(x1)
    n2, T2 = hartley_normalize(x2)
    u, v = n1[:, 0], n1[:, 1]
    u2, v2 = n2[:, 0], n2[:, 1]
    A = np.column_stack([u2 * u, u2 * v, u2, v2 * u, v2 * v, v2, u, v, np.ones(len(u))])
    _, _, Vt = np.linalg.svd(A)
    U, S, Vt = np.linalg.svd(Vt[-1].reshape(3, 3))
    S[2] = 0
    F = T2.T @ (U @ np.diag(S) @ Vt) @ T1
    return F / np.linalg.norm(F)


def epipolar_distance(F, x1, x2):
    """Distance in px of each x2 from the epipolar line F x1 (cv2.computeCorrespondEpilines)."""
    lines = to_h(x1) @ np.asarray(F, dtype=float).T
    x2 = np.asarray(x2, dtype=float)
    num = lines[:, 0] * x2[:, 0] + lines[:, 1] * x2[:, 1] + lines[:, 2]
    return np.abs(num) / np.maximum(np.hypot(lines[:, 0], lines[:, 1]), 1e-12)


def ransac_fundamental(x1, x2, thresh=1.0, iters=1000, seed=0):
    """RANSAC around eight_point: (F, inliers bool (N,)) (cv2.findFundamentalMat with FM_RANSAC).
    Needs at least 8 matches (ValueError otherwise). The same seed gives the same result."""
    x1, x2 = np.asarray(x1, dtype=float).reshape(-1, 2), np.asarray(x2, dtype=float).reshape(-1, 2)
    if len(x1) < 8 or len(x2) != len(x1):
        raise ValueError(f"ransac_fundamental needs at least 8 matches, got {len(x1)} and {len(x2)} points")
    rng = np.random.default_rng(seed)
    best, F_best = None, None
    for _ in range(iters):
        idx = rng.choice(len(x1), 8, replace=False)
        F = eight_point(x1[idx], x2[idx])
        inliers = epipolar_distance(F, x1, x2) < thresh
        if best is None or inliers.sum() > best.sum():
            best, F_best = inliers, F
    if best.sum() < 8:                       # too few to refit on: keep the best sample's F
        return F_best, best
    F = eight_point(x1[best], x2[best])
    return F, epipolar_distance(F, x1, x2) < thresh


def ransac_fundamental_cv(x1, x2, thresh=1.0, iters=1000, seed=0):
    """OpenCV's RANSAC for F (USAC_FAST: RANSAC that also polishes good models and gives up early on hopeless ones),
    same inputs and outputs as ransac_fundamental, a few hundred times faster. Returns (None, all False) if no F
    (also for fewer than 8 matches). seed is not used: USAC draws its samples from its own fixed random state (not
    even cv2.setRNGSeed changes it), so the same input always gives the same output."""
    x1, x2 = np.asarray(x1, np.float32), np.asarray(x2, np.float32)
    if len(x1) < 8:
        return None, np.zeros(len(x1), bool)
    F, mask = cv2.findFundamentalMat(x1, x2, getattr(cv2, "USAC_FAST", cv2.FM_RANSAC), thresh, 0.999, iters)
    if F is None or mask is None or F.shape != (3, 3):
        return None, np.zeros(len(x1), bool)
    return F / np.linalg.norm(F), mask.ravel().astype(bool)


def triangulate(P1, P2, x1, x2):
    """The pipeline's own triangulation of N points: X (N,3), OpenCV's cv2.triangulatePoints divided by the fourth row.
    (Your triangulate_points, Task 8, does the same with your own DLT; the pipeline does not depend on it.)"""
    P1, P2 = np.asarray(P1, dtype=np.float64), np.asarray(P2, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64).reshape(-1, 2)
    x2 = np.asarray(x2, dtype=np.float64).reshape(-1, 2)
    if len(x1) == 0:
        return np.zeros((0, 3))
    X_h = cv2.triangulatePoints(P1, P2, np.ascontiguousarray(x1.T), np.ascontiguousarray(x2.T))
    return (X_h[:3] / X_h[3]).T


def choose_pose(E, K, x1, x2):
    """The one of the 4 poses of decompose_E that puts the points in front of both cameras (cv2.recoverPose).

    K is (3,3) for both photos or [K1, K2] (one K per photo). Returns (R, t, X (N,3), counts list of 4 ints).
    """
    K1, K2 = _two_K(K)
    P1 = projection_matrix(K1, np.eye(3), np.zeros(3))
    best, counts = None, []
    for R, t in decompose_E(E):
        X = triangulate(P1, projection_matrix(K2, R, t), x1, x2)
        in_front = (X[:, 2] > 0) & ((X @ R.T + t)[:, 2] > 0)
        counts.append(int(in_front.sum()))
        if counts[-1] == max(counts):
            best = (R, t, X)
    R, t, X = best
    return R, t, X, counts


def pnp(X, x, K):
    """Camera pose (R, t) from N >= 6 3D-2D matches (cv2.solvePnP).

    Linear DLT in normalised coordinates: the 12 numbers of [R | t] from 2 equations per match, then the nearest
    rotation. When the points lie on one plane (or nearly: their smallest spread is under 10% of the largest), the
    DLT has no unique answer, so OpenCV's SQPnP, which handles flat point sets, gives the pose instead. Points on one
    line fix no pose at all: SQPnP raises cv2.error on them, and the DLT gives some pose (RANSAC then discards it).
    """
    X, x, K = np.asarray(X, float).reshape(-1, 3), np.asarray(x, float).reshape(-1, 2), np.asarray(K, float)
    spread = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    if spread[2] < 0.1 * spread[0]:
        try:
            ok, rvec, tvec = cv2.solvePnP(np.ascontiguousarray(X), np.ascontiguousarray(x), K, None,
                                          flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:                                  # points on one line (or all the same): no SQPnP pose
            ok = False
        if ok:
            return rodrigues(rvec.ravel()), tvec.ravel().astype(float)
    y = to_h(x) @ np.linalg.inv(K).T
    u, v = y[:, 0] / y[:, 2], y[:, 1] / y[:, 2]
    Xh = to_h(X)
    zeros = np.zeros_like(Xh)
    A = np.vstack([np.hstack([Xh, zeros, -u[:, None] * Xh]),
                   np.hstack([zeros, Xh, -v[:, None] * Xh])])
    _, _, Vt = np.linalg.svd(A)
    M = Vt[-1].reshape(3, 4)
    if np.linalg.det(M[:, :3]) < 0:
        M = -M
    scale = np.linalg.svd(M[:, :3], compute_uv=False).mean()
    return nearest_rotation(M[:, :3]), M[:, 3] / scale


def refine_pose(X, x, K, R, t, f_scale=2.0):
    """Polish a camera pose by minimising the reprojection error of 3D-2D matches (least_squares, Huber loss)."""
    X, x, K = np.asarray(X, float), np.asarray(x, float), np.asarray(K, float)
    R, t = np.asarray(R, float), np.asarray(t, float).reshape(3)

    def residuals(p):
        uv, _ = project(X, rodrigues(p[:3]) @ R, t + p[3:], K)
        return (uv - x).ravel()

    res = least_squares(residuals, np.zeros(6), loss="huber", f_scale=f_scale, method="trf", max_nfev=50)
    return nearest_rotation(rodrigues(res.x[:3]) @ R), t + res.x[3:]


def refine_relative_pose(x1, x2, K, R, t, f_scale=1.0):
    """Polish a two view pose (R, t) on the epipolar error of F = K2^-T [t]x R K1^-1 (Huber loss). |t| stays 1."""
    K1, K2 = _two_K(K)
    x1, x2, K1i, K2i = to_h(x1), to_h(x2), np.linalg.inv(K1), np.linalg.inv(K2)
    R, t = np.asarray(R, float), np.asarray(t, float).reshape(3)

    def pose(p):
        t2 = t + p[3:]
        return rodrigues(p[:3]) @ R, t2 / np.linalg.norm(t2)

    def residuals(p):
        Rp, tp = pose(p)
        F = K2i.T @ skew(tp) @ Rp @ K1i
        Fx1, Ftx2 = x1 @ F.T, x2 @ F
        den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
        return np.sum(x2 * Fx1, axis=1) / np.sqrt(den)

    res = least_squares(residuals, np.zeros(6), loss="huber", f_scale=f_scale, method="trf", max_nfev=100)
    R2, t2 = pose(res.x)
    return nearest_rotation(R2), t2


def pnp_ransac(X, x, K, thresh=4.0, iters=300, seed=0):
    """Robust camera pose from 3D-2D matches that contain wrong ones (RANSAC around pnp).

    X (N,3) world points, x (N,2) pixels, K (3,3) of this photo. Returns (R, t, inliers bool (N,)).
    Each round: 6 random matches, pnp (the DLT, or SQPnP for a flat sample), count the matches within a loose
    threshold (3 x thresh). It stops early once
    enough rounds were tried for the inlier share seen so far (99.9% sure). The best pose is then polished with
    refine_pose on its inliers a few times, at the real threshold.
    """
    X, x = np.asarray(X, float), np.asarray(x, float)
    n = len(X)
    if n < 6:
        raise ValueError(f"pnp_ransac needs at least 6 matches, got {n}")
    rng = np.random.default_rng(seed)
    loose = 3.0 * thresh
    best, best_count, needed, k = None, -1, iters, 0
    while k < min(iters, needed):
        k += 1
        idx = rng.choice(n, 6, replace=False)
        try:
            R, t = pnp(X[idx], x[idx], K)
        except (np.linalg.LinAlgError, cv2.error):
            continue
        count = int(np.sum(reprojection_error(X, R, t, K, x) < loose))
        if count > best_count:
            best, best_count = (R, t), count
            fail_all = 1.0 - (best_count / n) ** 6         # chance that one sample of 6 holds a wrong match
            needed = 10 if fail_all <= 0 else (iters if fail_all >= 1 else np.log(1e-3) / np.log(fail_all) + 10)
    if best is None:
        raise RuntimeError("pnp_ransac: no pose could be computed")
    R, t = best
    inl = reprojection_error(X, R, t, K, x) < loose
    for _ in range(4):
        if inl.sum() < 6:
            break
        R, t = refine_pose(X[inl], x[inl], K, R, t, f_scale=thresh / 2.0)
        new = reprojection_error(X, R, t, K, x) < thresh
        if np.array_equal(new, inl):
            break
        inl = new
    return R, t, reprojection_error(X, R, t, K, x) < thresh


# Bundle adjustment. params = [r_0, t_0, ..., r_{N-1}, t_{N-1}, X_0, ..., X_{M-1}]: 6 numbers per camera (axis angle
# r and translation t), then 3 per point. K is (3,3) for all cameras or (N,3,3), one per camera.

def pack_params(Rs, ts, pts):
    """Cameras (Rs (N,3,3), ts (N,3)) and points (M,3) into one flat vector of length 6N + 3M."""
    cams = [np.concatenate([rodrigues_inv(R), np.asarray(t, float).reshape(3)]) for R, t in zip(Rs, ts)]
    return np.concatenate([*cams, np.asarray(pts, float).ravel()])          # no cameras: just the points


def unpack_params(params, n_cams, n_pts):
    """Inverse of pack_params: returns (Rs (N,3,3), ts (N,3), pts (M,3))."""
    params = np.asarray(params, float)
    cams = params[:6 * n_cams].reshape(n_cams, 6)
    pts = params[6 * n_cams:6 * n_cams + 3 * n_pts].reshape(n_pts, 3)
    return rodrigues(cams[:, :3]), cams[:, 3:].copy(), pts


def reprojection_residuals(params, n_cams, n_pts, cam_idx, pt_idx, obs_xy, K):
    """All reprojection errors, projected minus observed, interleaved [dx_0, dy_0, ...] (cv2.projectPoints).
    K is (3,3) for every camera or (n_cams,3,3), one per camera."""
    Rs, ts, pts = unpack_params(params, n_cams, n_pts)
    Xc = np.einsum("nij,nj->ni", Rs[cam_idx], pts[pt_idx]) + ts[cam_idx]
    K = np.asarray(K, dtype=float)
    p = Xc @ K.T if K.ndim == 2 else np.einsum("nij,nj->ni", K[cam_idx], Xc)
    return (p[:, :2] / p[:, 2:] - obs_xy).ravel()


def jacobian_sparsity(n_cams, n_pts, cam_idx, pt_idx):
    """Which residuals depend on which parameters: a 0/1 sparse matrix (2 n_obs, 6 n_cams + 3 n_pts).
    Observation k (rows 2k, 2k+1) depends only on the 6 numbers of its camera and the 3 of its point."""
    cam_idx, pt_idx = np.asarray(cam_idx), np.asarray(pt_idx)
    n_obs = len(cam_idx)
    cols = np.hstack([6 * cam_idx[:, None] + np.arange(6), 6 * n_cams + 3 * pt_idx[:, None] + np.arange(3)])
    rows = np.repeat(np.arange(2 * n_obs), 9)
    cols = np.repeat(cols, 2, axis=0).ravel()
    return sp.csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(2 * n_obs, 6 * n_cams + 3 * n_pts))


def bundle_adjust(Rs, ts, pts, cam_idx, pt_idx, obs_xy, K, max_nfev=30):
    """Move all cameras and points together so that the reprojection residuals get as small as possible.

    scipy's least_squares does the work: the sparsity pattern makes it fast, and the Huber loss (errors above 2 px
    count less) keeps a few wrong matches from dragging everything. Each kind of number gets a fixed step size
    (x_scale): 0.01 rad for a rotation, 1% of the scene size for a translation or a point, so one step moves them by
    comparable amounts. BA polishes a good start, it cannot rescue a bad one.
    Returns dict(Rs, ts, pts, rms_before, rms_after, median_before, median_after, nfev, steps):
    rms = sqrt(mean(residual^2)) in px, median = the median reprojection error in px (with the Huber loss the rms can
    grow while most errors shrink: a few bad matches are allowed to stand out), nfev the number of residual
    evaluations (at most max_nfev) and steps how many of them were accepted (0: nothing moved).
    """
    n_cams, n_pts = len(Rs), len(pts)
    args = (n_cams, n_pts, cam_idx, pt_idx, obs_xy, K)
    x0 = pack_params(Rs, ts, pts)
    r0 = reprojection_residuals(x0, *args)
    rms_before = float(np.sqrt(np.mean(r0 ** 2))) if len(r0) else float("nan")
    med_before = float(np.median(np.hypot(r0[0::2], r0[1::2]))) if len(r0) else float("nan")
    if len(r0) == 0:
        return dict(Rs=np.asarray(Rs, float).reshape(-1, 3, 3), ts=np.asarray(ts, float).reshape(-1, 3),
                    pts=np.asarray(pts, float).reshape(-1, 3), rms_before=rms_before, rms_after=rms_before,
                    median_before=med_before, median_after=med_before, nfev=0, steps=0)
    P = np.asarray(pts, float).reshape(-1, 3)
    size = np.ptp(np.percentile(P, [5, 95], axis=0), axis=0).max() if len(P) else 0.0
    size = size if size > 0 else 1.0
    x_scale = np.concatenate([np.tile([1e-2] * 3 + [1e-2 * size] * 3, n_cams), np.full(3 * n_pts, 1e-2 * size)])
    res = least_squares(reprojection_residuals, x0, jac_sparsity=jacobian_sparsity(n_cams, n_pts, cam_idx, pt_idx),
                        loss="huber", f_scale=2.0, x_scale=x_scale, method="trf", max_nfev=max_nfev, args=args,
                        tr_options=dict(atol=1e-10, btol=1e-10))   # solve each step accurately (the default is loose)
    Rs, ts, pts = unpack_params(res.x, n_cams, n_pts)
    r1 = reprojection_residuals(res.x, *args)
    return dict(Rs=Rs, ts=ts, pts=pts, rms_before=rms_before, rms_after=float(np.sqrt(np.mean(r1 ** 2))),
                median_before=med_before, median_after=float(np.median(np.hypot(r1[0::2], r1[1::2]))),
                nfev=int(res.nfev), steps=int(res.njev) - 1)
