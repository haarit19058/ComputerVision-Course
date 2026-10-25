"""Dense clouds: optical flow between placed photos, epipolar check, triangulation, then a clean up.
For our own reconstruction (densify) and for a survey's cameras (densify_survey)."""

import collections
import os
import time

import numpy as np
import cv2

from .data import label_name
from .geometry import (camera_center, epipolar_distance, fundamental_from_poses, pixel_colors, project,
                       projection_matrix, ray_angles, reprojection_error, to_h, triangulate)
from .graph import verify_pairs
from .sfm import scene_image


def _partners(rec, scene, i, n=2, min_angle=3.0, max_angle=30.0):
    """Registered photos that share the most 3D points with photo i and look at them from a useful angle."""
    pts_i = [tid for tid in scene["img_tracks"][i] if tid in rec["points"]]
    votes = collections.Counter(v for tid in pts_i for v in scene["tracks"][tid] if v != i and v in rec["cams"])
    Ci = camera_center(*rec["cams"][i])
    out = []
    for j, c in votes.most_common(12):
        Cj = camera_center(*rec["cams"][j])
        X = np.array([rec["points"][tid] for tid in pts_i if j in scene["tracks"][tid]])
        ang = np.median(ray_angles(X, Ci, Cj))
        if min_angle <= ang <= max_angle:
            out.append(j)
        if len(out) == n:
            break
    return out


def clean_cloud(P, C, src, cell, min_views=2, n_neighbours=8, max_sigma=2.0):
    """Turn a noisy cloud into crisp surfaces.

    P (N,3) points, C (N,3) uint8 colours, src (N,) the photo each point came from, cell: the side of a grid cube.
      1. Voxel grid: all points in one cube become one point, at their mean position with their mean colour.
      2. Keep a cube only when points from at least min_views different photos fell into it (one photo alone can be
         wrong; two that agree rarely are).
      3. Statistical outlier removal: drop points whose mean distance to their n_neighbours nearest points is more
         than max_sigma standard deviations above the average (isolated specks floating in the air).
    Returns (P (M,3) float32, C (M,3) uint8).
    """
    from scipy.spatial import cKDTree
    if not (np.isfinite(cell) and cell > 0):
        raise ValueError(f"clean_cloud: the cube side cell must be a positive number, got {cell}")
    P = np.asarray(P, np.float64).reshape(-1, 3)
    C, src = np.asarray(C).reshape(-1, 3), np.asarray(src, np.int64).reshape(-1)
    finite = np.isfinite(P).all(axis=1)
    P, C, src = P[finite], C[finite], src[finite]
    if len(P) == 0:
        return P.astype(np.float32), C.astype(np.uint8)
    ijk = np.floor((P - P.min(axis=0)) / cell).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    if float(dims[0]) * float(dims[1]) * float(dims[2]) < 2.0 ** 62:     # one number per cube fits in int64
        uniq, inv = np.unique(ijk[:, 0] + dims[0] * (ijk[:, 1] + dims[1] * ijk[:, 2]), return_inverse=True)
    else:                                                                # a huge grid (far strays): rows instead
        uniq, inv = np.unique(ijk, axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv).astype(float)
    Q = np.stack([np.bincount(inv, P[:, k]) for k in range(3)], axis=1) / cnt[:, None]
    D = np.stack([np.bincount(inv, C[:, k].astype(float)) for k in range(3)], axis=1) / cnt[:, None]
    pairs = np.unique(inv.astype(np.int64) * (int(np.max(src)) + 1) + src, return_index=False)
    views = np.bincount(pairs // (int(np.max(src)) + 1), minlength=len(uniq))
    keep = views >= min_views
    Q, D = Q[keep], D[keep]
    if len(Q) > n_neighbours:
        d, _ = cKDTree(Q).query(Q, k=n_neighbours + 1)
        md = d[:, 1:].mean(axis=1)
        ok = md <= md.mean() + max_sigma * md.std()     # <=: when all are alike (std 0), none is an outlier
        Q, D = Q[ok], D[ok]
    return Q.astype(np.float32), np.clip(np.round(D), 0, 255).astype(np.uint8)


def _dense_from_plan(scene, cams, plan, stride=2, min_texture=2.0, max_fb=1.0, max_epi=1.0, max_err=1.0,
                     min_angle=2.0):
    """Steps 0 to 3 of densify for a plan: a list of (photo i, depth of the scene in front of i, partner photos).

    cams {photo: (R, t)}; scene gives K (one per photo) and the photos (scene_image). Returns (pts, colors, src,
    counts): the triangulated points, their colours, the photo each came from, and how many pixels survived each step.
    """
    K = scene["K"]
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    pts, cols, src, counts = [], [], [], np.zeros(5, int)
    for i, depth, partners in plan:
        img_i = scene_image(scene, i)
        gi = cv2.cvtColor(img_i, cv2.COLOR_RGB2GRAY)
        h, w = gi.shape
        Ri, ti = cams[i]
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        p = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
        g = gi.astype(np.float32)
        std = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - cv2.blur(g, (7, 7)) ** 2, 0))
        textured = std[ys, xs].ravel() > min_texture
        for j in partners:
            Rj, tj = cams[j]
            R_rel = Rj @ Ri.T
            t_rel = tj - R_rel @ ti
            H = K[j] @ (R_rel + np.outer(t_rel, [0, 0, 1.0]) / depth) @ np.linalg.inv(K[i])   # pixel of i -> of j
            gj = cv2.cvtColor(scene_image(scene, j), cv2.COLOR_RGB2GRAY)
            wj = cv2.warpPerspective(gj, H, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            fwd, bwd = flow.calc(gi, wj, None), flow.calc(wj, gi, None)
            q = p + fwd[p[:, 1].astype(int), p[:, 0].astype(int)]
            back = cv2.remap(bwd, q[:, 0].reshape(xs.shape).astype(np.float32),
                             q[:, 1].reshape(xs.shape).astype(np.float32), cv2.INTER_LINEAR).reshape(-1, 2)
            qj = to_h(q) @ H.T
            qj = qj[:, :2] / qj[:, 2:]
            hj, wj_ = gj.shape
            ok = textured & (np.linalg.norm(q + back - p, axis=1) < max_fb)
            ok &= np.all((qj >= 0) & (qj <= [wj_ - 1, hj - 1]), axis=1)
            ok &= np.all((q >= 0) & (q <= [w - 1, h - 1]), axis=1)
            counts[:2] += [len(p), textured.sum()]
            k = np.flatnonzero(ok)
            counts[2] += len(k)
            F = fundamental_from_poses(K[i], Ri, ti, K[j], Rj, tj)
            k = k[epipolar_distance(F, p[k], qj[k]) < max_epi]
            counts[3] += len(k)
            if len(k) == 0:
                continue
            X = triangulate(projection_matrix(K[i], Ri, ti), projection_matrix(K[j], Rj, tj), p[k], qj[k])
            good = ((reprojection_error(X, Ri, ti, K[i], p[k]) < max_err) &
                    (reprojection_error(X, Rj, tj, K[j], qj[k]) < max_err) &
                    (ray_angles(X, camera_center(Ri, ti), camera_center(Rj, tj)) > min_angle))
            counts[4] += good.sum()
            pts.append(X[good].astype(np.float32))
            cols.append(pixel_colors(img_i, p[k[good]]))
            src.append(np.full(int(good.sum()), i, np.int32))
    pts = np.vstack(pts) if pts else np.zeros((0, 3), np.float32)
    cols = np.vstack(cols) if cols else np.zeros((0, 3), np.uint8)
    src = np.concatenate(src) if src else np.zeros(0, np.int32)
    return pts, cols, src, counts


def _report(name, n_photos, counts, n_raw, n_clean, seconds):
    print(f"[dense] {name}{n_photos} photos, pixels: {counts[0]:,} looked at, {counts[1]:,} textured, "
          f"{counts[2]:,} flow consistent, {counts[3]:,} on their epipolar line, {counts[4]:,} triangulated well; "
          f"cleaned: {n_raw:,} -> {n_clean:,} points, {seconds:.0f} s")


def densify(rec, scene, n_images=64, n_partners=2, stride=2, min_texture=2.0, max_fb=1.0, max_epi=1.0, max_err=1.0,
            min_angle=2.0, voxel=1 / 500, min_views=2, verbose=True):
    """Dense point cloud from the placed cameras: every stride px of n_images photos testify.

    The photos with the most 3D points are used. For each such photo i and each of its partners j (photos that share
    many points with i, seen at an angle between 3 and 30 degrees):
      0. Warp photo j to look like photo i: the homography of the plane through the median depth of i's points
         (internet photos differ in zoom and tilt; after the warp only parallax is left).
      1. Optical flow (OpenCV's DIS) from i to the warped j and back gives, for each pixel p of i, its match q in j.
      2. Keep p when its patch is not flat, the flow leads back to p within max_fb px, and q lies within max_epi px of
         p's epipolar line (epipolar_distance, with F from the known poses).
      3. triangulate (p, q); keep points within max_err px in both photos and with rays meeting above min_angle degrees.
      4. Clean up (clean_cloud): average the points in each cube of a grid (side voxel x the scene size), keep the
         cubes that min_views different photos agree on, and drop isolated specks.
    Returns dict(pts (M,3) float32, colors (M,3) uint8, n_raw: the number of points before cleaning).
    """
    t0 = time.time()
    n_pts = {i: sum(tid in rec["points"] for tid in scene["img_tracks"][i]) for i in rec["cams"]}
    chosen = [i for i in sorted(rec["cams"], key=lambda i: -n_pts[i]) if n_pts[i] > 0][:n_images]   # with a depth
    plan = []
    for i in chosen:
        Ri, ti = rec["cams"][i]
        Xi = np.array([rec["points"][tid] for tid in scene["img_tracks"][i] if tid in rec["points"]])
        plan.append((i, np.median((Xi @ Ri.T + ti)[:, 2]), _partners(rec, scene, i, n_partners)))
    pts, cols, src, counts = _dense_from_plan(scene, rec["cams"], plan, stride, min_texture, max_fb, max_epi,
                                              max_err, min_angle)
    n_raw = len(pts)
    size = np.ptp(np.percentile(rec["pts"], [5, 95], axis=0), axis=0).max()   # scene size from the sparse points
    pts, cols = clean_cloud(pts, cols, src, voxel * size, min_views=min_views)
    if verbose:
        _report("", len(chosen), counts, n_raw, len(pts), time.time() - t0)
    return dict(pts=pts, colors=cols, n_raw=n_raw)


def survey_scene(rome, group):
    """A scene made from a survey (the ground truth cameras of one landmark, e.g. "st_peters_square"): its photos,
    their K and poses (from the survey, not from our SfM) and the survey's 3D points. Returns (scene, cams), with
    cams {local photo: (R, t)} in the survey's frame, ready for densify_survey."""
    gt = rome["gt"]
    rows = np.flatnonzero((gt["group"] == group) & ~gt.get("secondary", np.zeros(len(gt["index"]), bool)))
    if len(rows) == 0:
        raise ValueError(f"no survey called {group!r}; the crate has {sorted(set(gt['group'].tolist()))}")
    members = gt["index"][rows]
    z = np.load(os.path.join(rome["root"], "gt", f"points_{group}.npz"))
    scene = dict(name=label_name(rome["groups"].index(group), rome) if group in rome["groups"] else group, group=group,
                 root=rome["root"], members=members, paths=rome["images"][members], K=gt["K"][rows],
                 sizes=rome["sizes"][members], gt_points=z["xyz"].astype(float))
    cams = {k: (gt["R"][r], gt["t"][r]) for k, r in enumerate(rows)}
    return scene, cams


def densify_survey(rome, group, n_images=64, n_partners=2, voxel=1 / 600, min_views=1, min_angle=1.0, near_survey=0.01,
                   verbose=True, **flow_args):
    """densify for a survey (survey_scene): the same steps, with the survey's cameras instead of our SfM.

    There are no tracks here, so the plan comes from the survey: a photo "sees" the survey points that project inside
    it, photos are picked greedily to cover the points, and the partners of a photo are nearby photos that look
    the same way (within 45 degrees), see it from an angle between 1 and 25 degrees, and preferably pass the word
    match check (verify_pairs) (survey photos are
    often taken from far away, so neighbouring views differ by only a degree or two). The depth for the warp is the
    median depth of the points it sees. The cleaning cube is a fraction (voxel) of the dense cloud's own size; one
    photo per cube is enough here (min_views=1), the speck filter still removes floating points. Last, only points
    within `near_survey` (a fraction of the survey's size) of a survey point are kept: that removes sky and wrong surfaces.
    Returns dict(pts, colors, n_raw, cams (the poses of the photos used), name).
    """
    t0 = time.time()
    scene, cams = survey_scene(rome, group)
    X = scene["gt_points"]
    X = X[np.random.default_rng(0).choice(len(X), min(len(X), 20000), replace=False)]
    n = len(cams)
    V = np.zeros((n, len(X)), bool)                       # V[i, p]: survey point p projects inside photo i
    depth = np.ones(n)
    for i, (R, t) in cams.items():
        uv, z = project(X, R, t, scene["K"][i])
        w, h = scene["sizes"][i]
        V[i] = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if V[i].any():
            depth[i] = np.median(z[V[i]])
    Vf = V.astype(np.float32)
    C = np.array([camera_center(*cams[i]) for i in range(n)])
    axes = np.array([cams[i][0][2] for i in range(n)])    # viewing directions

    def angle(i, j):
        m = X[V[i] & V[j]].mean(axis=0) if (V[i] & V[j]).any() else X.mean(axis=0)
        a, b = C[i] - m, C[j] - m
        return np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1)))

    # candidates: a coverage ordering of the photos, then for each the 12 most similar views by covisibility
    order, covered = [], np.zeros(len(X))
    for _ in range(min(3 * n_images, n)):
        score = Vf @ (1.0 / (1.0 + covered)).astype(np.float32)
        score[order] = -1
        i = int(np.argmax(score))
        if score[i] <= 0:                                 # the photos left see no survey point (no depth for them)
            break
        order.append(i)
        covered += V[i]
    from scipy.spatial import cKDTree
    near = cKDTree(C).query(C, k=min(n, 41))[1]           # the 40 nearest camera centres of every photo
    cand = {}
    for i in order:
        js = [int(j) for j in near[i][1:] if axes[i] @ axes[j] > np.cos(np.radians(45))
              and 1.0 <= angle(i, int(j)) <= 25.0][:12]
        cand[i] = js
    # partners: first the nearby photos that pass the word match check (verify_pairs), strongest first; when a photo
    # has too few, its nearest photos fill in (the survey poses are trusted, and the flow and epipolar checks reject
    # a poor partner pixel by pixel)
    members = scene["members"]
    key = lambda i, j: (min(members[i], members[j]), max(members[i], members[j]))
    pairs = sorted({key(i, j) for i, js in cand.items() for j in js})
    n_inl, ok = verify_pairs(np.array(pairs, dtype=np.int64).reshape(-1, 2), rome, verbose=False)
    strength = {p: k for p, k, o in zip(pairs, n_inl.tolist(), ok.tolist()) if o}
    plan = []
    for i in order:
        good = sorted([j for j in cand[i] if key(i, j) in strength], key=lambda j: -strength[key(i, j)])
        js = (good + [j for j in cand[i] if j not in good])[:n_partners]
        if js:
            plan.append((i, depth[i], js))
        if len(plan) == n_images:
            break
    pts, cols, src, counts = _dense_from_plan(scene, cams, plan, min_angle=min_angle, **flow_args)
    n_raw = len(pts)
    if n_raw:
        med = np.median(pts, axis=0)
        size = 2 * np.percentile(np.linalg.norm(pts - med, axis=1), 90)
        pts, cols = clean_cloud(pts, cols, src, voxel * size, min_views=min_views)
        # keep what lies on the surveyed surfaces: within near_survey (a fraction of the survey's size) of a survey point;
        # this drops sky and other surfaces the flow got wrong
        from scipy.spatial import cKDTree
        G = scene["gt_points"]
        g_size = np.ptp(np.percentile(G, [5, 95], axis=0), axis=0).max()
        d, _ = cKDTree(G).query(pts)
        pts, cols = pts[d < near_survey * g_size], cols[d < near_survey * g_size]
    if verbose:
        _report(f"{scene['name']}: ", len(plan), counts, n_raw, len(pts), time.time() - t0)
    used = sorted({k for i, _, js in plan for k in [i, *js]})
    return dict(pts=pts, colors=cols, n_raw=n_raw, cams={k: cams[k] for k in used}, name=scene["name"])
