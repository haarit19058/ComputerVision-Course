"""Geometry on one component: SIFT again, tracks, the skeleton SfM, everyone else by PnP, and the two St Peter's
surveys."""

import itertools
import collections
import time

import numpy as np
import cv2

from .data import _gt_points, _read_rgb, label_name, load_image
from .geometry import (bundle_adjust, camera_center, choose_pose, essential_from_F, pixel_colors, pnp_ransac,
                       projection_matrix, ransac_fundamental, ransac_fundamental_cv, ray_angles,
                       refine_relative_pose, reprojection_error, triangulate)
from ._yours import yours
from .graph import UnionFind, skeleton_nodes


def _sift(img, n_features):
    """RootSIFT keypoints (N,2) float32 and descriptors (N,128) float16 of an RGB image (one keypoint per spot)."""
    sift = cv2.SIFT_create(nfeatures=n_features)
    kps, desc = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None)
    pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
    if desc is None:
        return pts, np.zeros((0, 128), np.float16)
    _, first = np.unique(np.round(pts, 1), axis=0, return_index=True)
    first = np.sort(first)
    desc = desc[first] / np.maximum(desc[first].sum(axis=1, keepdims=True), 1e-9)
    return pts[first], np.sqrt(desc).astype(np.float16)                # RootSIFT: sqrt of the L1 normalised SIFT


def sift_features(rome, i, n_features=2000):
    """SIFT on photo i, as landmark_scene computes it: (xy (N,2) float32 pixels, desc (N,128) float16 RootSIFT).
    RootSIFT is SIFT made unit length in L1 then square rooted, so plain L2 distances compare it well."""
    return _sift(load_image(rome, i), n_features)


def landmark_scene(rome, members, edges, weights=None, n_features=2000, per_image=8, verbose=True):
    """Everything SfM needs for one component: SIFT on its photos, matches along its verified edges, and tracks.

    members (n,) photo numbers; edges (E,2) local numbers (from component_graph); weights (E,) their inlier counts.
    Only some edges are matched (each photo's per_image strongest ones), which is plenty to link the tracks.
    Each pair: YOUR match_descriptors (ratio test and mutual check), then RANSAC on F (OpenCV). The surviving
    matches are joined with UnionFind:
    a track is one 3D point followed through several photos. A track that gets two keypoints in one photo loses that
    photo. Returns a scene dict (name, members, K (n,3,3), sizes, feats, pairs, tracks, img_tracks, gt).
    """
    match_descriptors = yours("match_descriptors")            # stops here, not after the SIFT, if it is missing
    match_descriptors(np.eye(2, 128, dtype=np.float32), np.eye(3, 128, dtype=np.float32))
    t0 = time.time()
    members = np.asarray(members, dtype=np.int64).reshape(-1)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    weights = np.ones(len(edges)) if weights is None else np.asarray(weights, float)
    n = len(members)
    lab = rome["labels"][members]
    major = np.bincount(lab[lab >= 0]).argmax() if np.any(lab >= 0) else -1
    feats = []
    for k, m in enumerate(members):
        if verbose and k and k % 200 == 0:
            print(f"  SIFT: {k} of {n} photos ({time.time() - t0:.0f} s)")
        feats.append(_sift(load_image(rome, m), n_features))
    t_sift = time.time() - t0
    if verbose:
        print(f"  SIFT: {n} photos in {t_sift:.0f} s; now matching along the strongest edges")
    chosen = set()                                           # each photo's strongest edges
    order = np.argsort(-weights, kind="stable")
    count = np.zeros(n, int)
    for e in order:
        i, j = edges[e]
        if count[i] < per_image or count[j] < per_image:
            chosen.add(int(e))
            count[i] += 1
            count[j] += 1
    offset = np.concatenate([[0], np.cumsum([len(f[0]) for f in feats])])
    uf = UnionFind(int(offset[-1]))
    pairs, involved = {}, set()
    t_match = time.time()
    for q, e in enumerate(sorted(chosen)):
        if verbose and q and q % 500 == 0:
            print(f"  matched {q} of {len(chosen)} pairs ({time.time() - t0:.0f} s)")
        i, j = (int(v) for v in edges[e])
        m = np.asarray(match_descriptors(feats[i][1].astype(np.float32), feats[j][1].astype(np.float32)),
                       dtype=np.int64).reshape(-1, 2)
        if len(m) < 20:
            continue
        _, inl = ransac_fundamental_cv(feats[i][0][m[:, 0]], feats[j][0][m[:, 1]], thresh=1.5, iters=2000)
        if inl.sum() < 20:
            continue
        pairs[(i, j)] = int(inl.sum())
        for a, b in m[inl].tolist():
            uf.union(int(offset[i] + a), int(offset[j] + b))
            involved.update((int(offset[i] + a), int(offset[j] + b)))
    t_match = time.time() - t_match
    # tracks: the groups of the union find with keypoints from 2 or more photos
    used = np.array(sorted(involved), dtype=np.int64)
    roots = np.array([uf.find(int(v)) for v in used], dtype=np.int64)
    img_of = np.searchsorted(offset, used, side="right") - 1
    tracks, img_tracks = [], {i: {} for i in range(n)}
    sort = np.argsort(roots, kind="stable")
    bounds = np.flatnonzero(np.diff(roots[sort])) + 1
    for grp in np.split(sort, bounds):
        imgs = img_of[grp]
        u, cnt = np.unique(imgs, return_counts=True)
        good = np.isin(imgs, u[cnt == 1])                    # drop photos with two keypoints in this track
        if good.sum() < 2:
            continue
        tid = len(tracks)
        tr = {}
        for v, img in zip(used[grp][good].tolist(), imgs[good].tolist()):
            xy = tuple(feats[img][0][v - offset[img]].tolist())
            tr[img] = xy
            img_tracks[img][tid] = xy
        tracks.append(tr)
    gt = rome["gt"]
    group = rome["groups"][major] if major >= 0 else None
    rows = {int(gt["index"][k]): k for k in range(len(gt["index"])) if gt["group"][k] == group}
    has = np.array([int(m) in rows for m in members])
    gt_R, gt_t = np.zeros((n, 3, 3)), np.zeros((n, 3))
    for li, m in enumerate(members):
        if int(m) in rows:
            gt_R[li], gt_t[li] = gt["R"][rows[int(m)]], gt["t"][rows[int(m)]]
    scene = dict(name=label_name(major, rome), group=group, root=rome["root"], members=members,
                 paths=rome["images"][members], K=rome["K"][members], sizes=rome["sizes"][members], feats=feats,
                 pairs=pairs, tracks=tracks, img_tracks=img_tracks, gt=dict(has=has, R=gt_R, t=gt_t))
    if verbose:
        lens = np.array([len(t) for t in tracks]) if tracks else np.zeros(1)
        print(f"[scene] {scene['name']}: {n} photos, SIFT in {t_sift:.0f} s; {len(pairs)} of {len(chosen)} matched "
              f"pairs kept ({1000 * t_match / max(len(chosen), 1):.0f} ms per pair); {len(tracks):,} tracks "
              f"(mean length {lens.mean():.1f}); {time.time() - t0:.0f} s in all")
    return scene


def scene_image(scene, i):
    """Photo i of a scene (local number) as RGB uint8."""
    return _read_rgb(str(scene["paths"][i]))


def _shared(scene, i, j):
    """Track ids seen in both photos i and j, and their pixels in each."""
    oi, oj = scene["img_tracks"][i], scene["img_tracks"][j]
    tids = [t for t in oi if t in oj]
    return tids, np.array([oi[t] for t in tids]).reshape(-1, 2), np.array([oj[t] for t in tids]).reshape(-1, 2)


def pair_matches(scene, i, j):
    """The matched pixels of photos i and j (local numbers) in a scene: the tracks seen in both.
    Returns (x1 (M,2), x2 (M,2)), row k of both is one track."""
    _, x1, x2 = _shared(scene, i, j)
    return x1.astype(float), x2.astype(float)


def pair_cameras(scene, i, j, thresh=1.5):
    """The two cameras of photos i and j (local numbers) from their matches alone, as projection matrices.

    ransac_fundamental gives F and its inliers, E = K2^T F K1 (essential_from_F), choose_pose picks the pose that
    puts the points in front of both cameras, and refine_relative_pose polishes it. Photo i is at the origin looking
    down +z, P1 = K1 [I | 0], and P2 = K2 [R | t] with |t| = 1: two photos alone cannot tell the true scale.
    Returns (P1 (3,4), P2 (3,4), x1 (M,2), x2 (M,2)): the inlier matches, row k of x1 and x2 one track.
    """
    Ki, Kj = scene["K"][i], scene["K"][j]
    x1, x2 = pair_matches(scene, i, j)
    if len(x1) < 8:
        raise ValueError(f"photos {i} and {j} share only {len(x1)} tracks: pick a pair with more")
    F, inl = ransac_fundamental(x1, x2, thresh=thresh, iters=1000)
    if inl.sum() < 8:
        raise ValueError(f"photos {i} and {j}: no fundamental matrix fits their matches")
    R, t, _, _ = choose_pose(essential_from_F(F, [Ki, Kj]), [Ki, Kj], x1[inl], x2[inl])
    R, t = refine_relative_pose(x1[inl], x2[inl], [Ki, Kj], R, t)
    return (projection_matrix(Ki, np.eye(3), np.zeros(3)), projection_matrix(Kj, R, t / np.linalg.norm(t)),
            x1[inl], x2[inl])


def triangulation_pair(scene, results, min_inliers=50, min_front=0.95, min_angle=2.0, verbose=True):
    """A pair of photos worth triangulating, from the homography test's results (rows i, j, ratio, n_H, n_F, local
    photo numbers). Pairs are tried from the lowest ratio (the most parallax) up; the first that passes every check
    is returned: at least min_inliers matches fit its fundamental matrix, a pose comes out of them (pair_cameras),
    and the toolbox's own triangulation puts at least min_front of the points in front of both cameras, with rays
    meeting at a median angle of at least min_angle degrees. Prints one line per pair tried. Returns (i, j)."""
    for i, j, ratio, *_ in sorted(results, key=lambda r: r[2]):
        i, j = int(i), int(j)
        try:
            P1, P2, x1, x2 = pair_cameras(scene, i, j)
        except ValueError as err:
            why = str(err)
        else:
            X = triangulate(P1, P2, x1, x2)
            Xh = np.column_stack([X, np.ones(len(X))])
            front = np.mean(((Xh @ P1[2]) > 0) & ((Xh @ P2[2]) > 0)) if len(X) else 0.0
            R2, t2 = np.linalg.solve(scene["K"][j], P2)[:, :3], np.linalg.solve(scene["K"][j], P2)[:, 3]
            angle = float(np.median(ray_angles(X, np.zeros(3), camera_center(R2, t2)))) if len(X) else 0.0
            if len(x1) < min_inliers:
                why = f"only {len(x1)} matches fit F (want {min_inliers})"
            elif front < min_front:
                why = f"only {front:.0%} of the points in front of both cameras (want {min_front:.0%})"
            elif angle < min_angle:
                why = f"rays meet at a median {angle:.1f} degrees (want {min_angle:g})"
            else:
                if verbose:
                    print(f"photos {i} and {j}: ratio {ratio:.2f}, {len(x1)} matches fit F, {front:.0%} in front "
                          f"of both cameras, median ray angle {angle:.1f} degrees: chosen")
                return i, j
        if verbose:
            print(f"photos {i} and {j}: ratio {ratio:.2f}, skipped: {why}")
    raise ValueError("no pair passed every check: run the homography test on more pairs")


def model_pair(rec, scene, i=None, j=None, min_angle=3.0):
    """Two placed photos of a model (rec from run_sfm_skeleton or register_rest) and the tracks they both see that the
    model holds a 3D point for. Without i and j: the two placed photos that share the most such tracks among the pairs
    whose rays meet at a median angle of at least min_angle degrees (a real baseline: near copies of one view
    triangulate poorly, as the baseline experiment shows), leaving out the start pair (rec["init_pair"]: its points
    were first triangulated from exactly those two photos); if no pair reaches min_angle, the widest pair. Give both i
    and j to choose the pair yourself. Returns (i, j, P1, P2, x1 (M,2), x2 (M,2), X (M,3) the model's points, refined
    by bundle adjustment over every photo that sees them)."""
    cams, points, K = rec["cams"], rec["points"], scene["K"]
    seen = {img: {t for t in scene["img_tracks"][img] if t in points} for img in cams}
    if (i is None) != (j is None):
        raise ValueError("give both photos (i and j), or neither to let model_pair choose")
    if i is None:
        order, start = sorted(cams), tuple(sorted(rec.get("init_pair") or ()))
        pairs = sorted(((len(seen[a] & seen[b]), a, b) for k, a in enumerate(order) for b in order[k + 1:]
                        if (a, b) != start), key=lambda p: (-p[0], p[1], p[2]))

        def angle(a, b):
            X = np.array([points[t] for t in seen[a] & seen[b]], float).reshape(-1, 3)
            C1, C2 = camera_center(*cams[a]), camera_center(*cams[b])
            return float(np.median(ray_angles(X, C1, C2))) if len(X) else 0.0
        widest = None
        for _, a, b in pairs:                     # the most shared first: usually the first or second qualifies
            ang = angle(a, b)
            if ang >= min_angle:
                i, j = a, b
                break
            if widest is None or ang > widest[0]:
                widest = (ang, a, b)
        else:
            if widest is None:
                raise ValueError("the model has no pair of placed photos to compare other than its start pair")
            i, j = widest[1], widest[2]
    tids = sorted(seen[i] & seen[j])
    x1 = np.array([scene["img_tracks"][i][t] for t in tids], float).reshape(-1, 2)
    x2 = np.array([scene["img_tracks"][j][t] for t in tids], float).reshape(-1, 2)
    X = np.array([points[t] for t in tids], float).reshape(-1, 3)
    return (i, j, projection_matrix(K[i], *cams[i]), projection_matrix(K[j], *cams[j]), x1, x2, X)


def _pair_counts(scene, images=None):
    """collections.Counter {(i, j): number of tracks seen in both}, over the photos in images (default: all)."""
    inside = None if images is None else set(int(i) for i in images)
    counts = collections.Counter()
    for tr in scene["tracks"]:
        v = sorted(k for k in tr if inside is None or k in inside)
        if len(v) >= 2:
            counts.update(itertools.combinations(v, 2))
    return counts


def strongest_pairs(scene, images=None, n=30, min_shared=50):
    """The n pairs of photos (among images, default all) that share the most tracks, at least min_shared.
    Returns an (M,2) int64 array (i < j, local numbers), the strongest first."""
    top = [p for p, c in _pair_counts(scene, images).most_common(n) if c >= min_shared]
    return np.array(top, dtype=np.int64).reshape(-1, 2)


def _two_view(scene, i, j, min_angle=2.0, max_err=4.0):
    """The two view start on photos i and j: ransac_fundamental, choose_pose, refine_relative_pose, triangulate.
    Returns (R, t, tids, X) keeping points in front of both, within max_err px, ray angle above min_angle; or None."""
    Ki, Kj = scene["K"][i], scene["K"][j]
    tids, x1, x2 = _shared(scene, i, j)
    if len(tids) < 30:
        return None
    F, inl = ransac_fundamental(x1, x2, thresh=1.5, iters=500)
    if inl.sum() < 20:
        return None
    R, t, _, _ = choose_pose(essential_from_F(F, [Ki, Kj]), [Ki, Kj], x1[inl], x2[inl])
    R, t = refine_relative_pose(x1[inl], x2[inl], [Ki, Kj], R, t)
    X = triangulate(projection_matrix(Ki, np.eye(3), np.zeros(3)), projection_matrix(Kj, R, t), x1[inl], x2[inl])
    ok = ((reprojection_error(X, np.eye(3), np.zeros(3), Ki, x1[inl]) < max_err) &
          (reprojection_error(X, R, t, Kj, x2[inl]) < max_err) &
          (ray_angles(X, np.zeros(3), camera_center(R, t)) > min_angle))
    return R, t, [tids[k] for k in np.flatnonzero(inl)[ok]], X[ok]


def _choose_init_pair(scene, images, n_pairs=60, n_candidates=8, verbose=True):
    """The start pair among images. Pairs that share many tracks are often near copies (same spot, a zoom or a turn
    of the head): no parallax, no depth. So among the n_pairs pairs sharing most tracks, YOUR homography_ratio
    (H inliers / F inliers, 4 px) scores each one, and pairs with a low ratio (real parallax) are preferred, as COLMAP
    does. The best n_candidates (score c (1 - ratio), then more shared tracks c, then the lower ratio) go through
    _two_view and the pair with the most good points (in front, ray angle above 2 degrees) wins."""
    homography_ratio = yours("homography_ratio")
    scored = []
    for (i, j), c in _pair_counts(scene, images).most_common(n_pairs):
        if c < 50:
            break
        x1, x2 = pair_matches(scene, i, j)
        ratio = float(homography_ratio(x1, x2, thresh=4.0, iters=300, seed=0)[0])
        scored.append((c * max(0.0, 1.0 - ratio), c, ratio, i, j))
    if not scored:
        raise RuntimeError(f"no start pair: no two of the {len(images)} skeleton photos share 50 or more tracks")
    best = None
    for _, c, ratio, i, j in sorted(scored, key=lambda p: (-p[0], -p[1], p[2], p[3], p[4]))[:n_candidates]:
        res = _two_view(scene, i, j)
        good = 0 if res is None else len(res[2])
        if verbose:
            print(f"  start candidate ({i},{j}): {c} shared tracks, homography ratio {ratio:.2f}, {good} good points")
        if best is None or good > best[0]:
            best = (good, i, j)
    if best[0] < 30:
        raise RuntimeError(f"no start pair: the best of the {min(len(scored), n_candidates)} candidate pairs gives only "
                           f"{best[0]} good points (in front of both cameras, ray angle above 2 degrees), 30 needed: too "
                           "little parallax, or too few shared tracks pass the two view checks")
    return best[1], best[2]


def _observations(rec, scene, max_err=np.inf):
    """All observations of triangulated tracks in registered photos, as arrays ready for bundle_adjust."""
    order = sorted(rec["cams"])
    tids = list(rec["points"])
    pos = {tid: k for k, tid in enumerate(tids)}
    cam_idx, pt_idx, obs = [], [], []
    for c, img in enumerate(order):
        for tid, xy in scene["img_tracks"][img].items():
            if tid in pos:
                cam_idx.append(c)
                pt_idx.append(pos[tid])
                obs.append(xy)
    Rs = np.array([rec["cams"][i][0] for i in order]).reshape(-1, 3, 3)
    ts = np.array([rec["cams"][i][1] for i in order]).reshape(-1, 3)
    Ks = scene["K"][order]
    pts = np.array([rec["points"][t] for t in tids]).reshape(-1, 3)
    cam_idx, pt_idx, obs = np.array(cam_idx, int), np.array(pt_idx, int), np.array(obs, float).reshape(-1, 2)
    Xc = np.einsum("nij,nj->ni", Rs[cam_idx], pts[pt_idx]) + ts[cam_idx]
    p = np.einsum("nij,nj->ni", Ks[cam_idx], Xc)
    err = np.linalg.norm(p[:, :2] / np.where(np.abs(p[:, 2:]) < 1e-12, 1e-12, p[:, 2:]) - obs, axis=1)
    err = np.where(Xc[:, 2] > 0, err, np.inf)
    ok = err < max_err
    return order, tids, Rs, ts, Ks, pts, cam_idx[ok], pt_idx[ok], obs[ok], err[ok]


def _prune_and_rms(rec, scene, max_err=4.0):
    """Drop points with fewer than 2 observations under max_err px. Returns the rms (px, per coordinate) of the rest."""
    _, tids, _, _, _, _, _, pt_idx, _, _ = _observations(rec, scene, max_err)
    n_good = np.bincount(pt_idx, minlength=len(tids))
    for k, tid in enumerate(tids):
        if n_good[k] < 2:
            del rec["points"][tid]
    err = _observations(rec, scene, max_err)[-1]
    return float(np.sqrt(np.mean(err ** 2) / 2)) if len(err) else float("nan")


def _run_ba(rec, scene, max_nfev, max_err=4.0, verbose=True):
    """bundle_adjust on every registered camera and point, using the observations within max_err px."""
    order, tids, Rs, ts, Ks, pts, cam_idx, pt_idx, obs, _ = _observations(rec, scene, max_err)
    t0 = time.time()
    out = bundle_adjust(Rs, ts, pts, cam_idx, pt_idx, obs, Ks, max_nfev=max_nfev)
    for c, img in enumerate(order):
        rec["cams"][img] = (np.asarray(out["Rs"][c]), np.asarray(out["ts"][c]).reshape(3))
    for k, tid in enumerate(tids):
        rec["points"][tid] = np.asarray(out["pts"][k])
    rec.setdefault("ba", []).append(dict(n_cams=len(order), seconds=time.time() - t0,
                                         **{k: out[k] for k in ("rms_before", "rms_after", "median_before",
                                                                "median_after", "nfev", "steps")}))
    if verbose:
        print(f"    BA: {len(order)} cameras, {len(tids):,} points, {len(obs):,} observations, median error "
              f"{out['median_before']:.2f} -> {out['median_after']:.2f} px, rms {out['rms_before']:.2f} -> "
              f"{out['rms_after']:.2f} px, {out['steps']} of {out['nfev']} steps accepted ({time.time() - t0:.1f} s)")


def _triangulate_new(rec, scene, m, min_angle=1.5, max_err=4.0):
    """Triangulate the tracks of photo m that have no 3D point yet, each against its widest baseline registered view.
    A point is kept when it reprojects within max_err px in both views and its ray angle is above min_angle degrees."""
    K, cams = scene["K"], rec["cams"]
    Rm, tm = cams[m]
    Cm = camera_center(Rm, tm)
    centres = {v: camera_center(*cams[v]) for v in cams}
    by_view = collections.defaultdict(list)
    for tid in scene["img_tracks"][m]:
        if tid in rec["points"]:
            continue
        views = [v for v in scene["tracks"][tid] if v in cams and v != m]
        if views:
            by_view[max(views, key=lambda v: np.linalg.norm(centres[v] - Cm))].append(tid)
    added = 0
    for v, tids in by_view.items():
        Rv, tv = cams[v]
        xm = np.array([scene["img_tracks"][m][t] for t in tids])
        xv = np.array([scene["img_tracks"][v][t] for t in tids])
        X = triangulate(projection_matrix(K[m], Rm, tm), projection_matrix(K[v], Rv, tv), xm, xv)
        ok = ((reprojection_error(X, Rm, tm, K[m], xm) < max_err) &
              (reprojection_error(X, Rv, tv, K[v], xv) < max_err) &
              (ray_angles(X, Cm, centres[v]) > min_angle))
        for k in np.flatnonzero(ok):
            rec["points"][tids[k]] = X[k]
        added += int(ok.sum())
    return added


def _pnp_register(rec, scene, m, thresh=4.0, min_inliers=12):
    """Pose of photo m from its tracks that already have a 3D point (pnp_ransac). Returns (R, t, n_inl, n) or None."""
    tl = [tid for tid in scene["img_tracks"][m] if tid in rec["points"]]
    if len(tl) < min_inliers:
        return None
    X = np.array([rec["points"][tid] for tid in tl])
    x = np.array([scene["img_tracks"][m][tid] for tid in tl])
    R, t, inl = pnp_ransac(X, x, scene["K"][m], thresh=thresh)
    if inl.sum() < max(min_inliers, 0.25 * len(tl)):
        return None
    return R, t, int(inl.sum()), len(tl)


def _finish(rec, scene, extra=None):
    """Turn the working model into arrays: pts (M,3), colors (M,3) uint8 (from the first photo of each track)."""
    track_ids = np.array(sorted(rec["points"]), dtype=np.int64)
    pts = np.array([rec["points"][tid] for tid in track_ids]).reshape(-1, 3)
    colors = np.zeros((len(track_ids), 3), dtype=np.uint8)
    by_img = collections.defaultdict(list)
    for k, tid in enumerate(track_ids.tolist()):
        img = min(v for v in scene["tracks"][tid] if v in rec["cams"])
        by_img[img].append(k)
    for img, ks in by_img.items():
        xy = np.array([scene["tracks"][track_ids[k]][img] for k in ks])
        colors[ks] = pixel_colors(scene_image(scene, img), xy)
    out = dict(cams=dict(rec["cams"]), points=dict(rec["points"]), pts=pts, colors=colors, track_ids=track_ids,
               history=list(rec.get("history", [])), skeleton=list(rec.get("skeleton", sorted(rec["cams"]))),
               ba=list(rec.get("ba", [])))
    out.update(extra or {})
    return out


def run_sfm_skeleton(scene, tree_edges, *, size=100, ba_every=4, verbose=True):
    """Incremental SfM, on the skeleton photos only.

    The skeleton is the spanning tree with its leaves peeled off (skeleton_nodes, at least size photos). Start from a
    strong pair with real parallax (YOUR homography_ratio picks it; then ransac_fundamental, choose_pose,
    triangulate), then again and again: register the skeleton photo with
    the most 2D-3D matches (pnp_ransac), triangulate its new tracks, and every ba_every photos run bundle_adjust.
    Returns rec = dict(cams {local photo: (R, t)}, pts (M,3), colors (M,3) uint8, track_ids, points, skeleton,
    history (one row per placed photo), ba (one row per bundle adjustment: rms before and after, steps accepted)).
    The tree weights are not needed (the tree already holds the strongest links): pass the tree edges only.
    """
    t_start = time.time()
    n = len(scene["members"])
    skel = skeleton_nodes(n, tree_edges, size)
    if verbose:
        print(f"[skeleton] {len(skel)} of {n} photos")
    i, j = _choose_init_pair(scene, skel, verbose=verbose)
    R, t, tids, X = _two_view(scene, i, j)
    rec = dict(cams={i: (np.eye(3), np.zeros(3)), j: (R, t)}, points=dict(zip(tids, X)), history=[],
               skeleton=[int(s) for s in skel])
    _run_ba(rec, scene, max_nfev=30, verbose=False)
    rms = _prune_and_rms(rec, scene)
    if verbose:
        ba = rec["ba"][-1]
        print(f"  start: photos ({i},{j}), {len(rec['points'])} points, rms {rms:.2f} px (BA: {ba['steps']} of "
              f"{ba['nfev']} steps accepted)")
    failed, step, todo = {}, 0, set(int(s) for s in skel) - {i, j}
    while True:
        cands = [m for m in todo if m not in rec["cams"] and failed.get(m, -1) < len(rec["cams"])]
        if not cands:
            break
        n_2d3d = {m: sum(tid in rec["points"] for tid in scene["img_tracks"][m]) for m in cands}
        m = max(cands, key=lambda q: n_2d3d[q])
        if n_2d3d[m] < 12:
            break
        res = _pnp_register(rec, scene, m)
        if res is None:
            failed[m] = len(rec["cams"])
            continue
        step += 1
        rec["cams"][m] = res[:2]
        n_new = _triangulate_new(rec, scene, m)
        if step % ba_every == 0:
            _run_ba(rec, scene, max_nfev=15, verbose=False)
        rms = _prune_and_rms(rec, scene)
        rec["history"].append(dict(step=step, image=m, n_cams=len(rec["cams"]), n_points=len(rec["points"]), rms=rms))
        if verbose and (step % 5 == 0 or step < 3):
            print(f"  step {step}: photo {m} ({res[2]} of {res[3]} inliers), +{n_new} points, "
                  f"{len(rec['cams'])} cameras, {len(rec['points']):,} points, rms {rms:.2f} px")
    if verbose and len(rec["ba"]) > 1:
        bas = rec["ba"][1:]
        print(f"  {len(bas)} bundle adjustments on the way: {sum(b['steps'] > 0 for b in bas)} moved the model, "
              f"median error {np.mean([b['median_before'] for b in bas]):.2f} -> "
              f"{np.mean([b['median_after'] for b in bas]):.2f} px on average, {sum(b['seconds'] for b in bas):.0f} s "
              "in all")
    _run_ba(rec, scene, max_nfev=30, verbose=verbose)
    rms = _prune_and_rms(rec, scene)
    out = _finish(rec, scene, dict(init_pair=(i, j), rms=rms))
    if verbose:
        print(f"[skeleton] {len(out['cams'])} of {len(skel)} skeleton photos placed, {len(out['pts']):,} points, "
              f"rms {rms:.2f} px, {time.time() - t_start:.0f} s")
    return out


def register_rest(rec, scene, thresh=4.0, verbose=True):
    """The Rome trick: every photo of the component that is not placed yet gets its pose from the skeleton's 3D points
    alone (pnp_ransac), all in one step. Then their new tracks are triangulated and one bundle adjustment polishes
    everything. thresh (px) is the inlier threshold of PnP, of the bundle adjustment and of the final pruning.
    Returns a new rec (same keys as run_sfm_skeleton)."""
    t0 = time.time()
    work = dict(cams=dict(rec["cams"]), points=dict(rec["points"]), history=list(rec["history"]),
                skeleton=list(rec["skeleton"]), ba=list(rec.get("ba", [])))
    n = len(scene["members"])
    new = {}
    for m in range(n):
        if m in work["cams"]:
            continue
        res = _pnp_register(work, scene, m, thresh=thresh)
        if res is not None:
            new[m] = res[:2]
    work["cams"].update(new)
    added = sum(_triangulate_new(work, scene, m) for m in new)
    t_pnp = time.time() - t0
    _prune_and_rms(work, scene, max_err=2 * thresh)
    _run_ba(work, scene, max_nfev=20, max_err=thresh, verbose=verbose)
    rms = _prune_and_rms(work, scene, max_err=thresh)
    out = _finish(work, scene, dict(init_pair=rec.get("init_pair"), rms=rms))
    if verbose:
        print(f"[rest] {len(new)} more photos placed by PnP in {t_pnp:.0f} s (+{added:,} points); now "
              f"{len(out['cams'])} of {n} photos, {len(out['pts']):,} points, rms {rms:.2f} px, "
              f"{time.time() - t0:.0f} s in all")
    return out


def camera_centres_vs_gt(rec, scene):
    """Camera centres of the placed photos that have a survey camera: (C_est (L,3) in our frame, C_gt (L,3) in the
    survey frame), row k of both is the same photo."""
    keys = [k for k in sorted(rec["cams"]) if scene["gt"]["has"][k]]
    C_est = np.array([camera_center(*rec["cams"][k]) for k in keys]).reshape(-1, 3)
    C_gt = np.array([camera_center(scene["gt"]["R"][k], scene["gt"]["t"][k]) for k in keys]).reshape(-1, 3)
    return C_est, C_gt


def st_peters_maps(rome, max_points=100000):
    """The two survey maps of St Peter's (Square and Basilica), each in its own frame, and the anchors they share:
    photos placed in both surveys (their camera centres in each frame).

    Returns dict(square=dict(pts, colors, centres), basilica=dict(...), links_square (L,3), links_basilica (L,3)):
    row k of the two links arrays is the same anchor in the two frames.
    """
    gt = rome["gt"]
    names = rome["groups"]
    sq = next((g for g in names if "square" in g.lower() and "peter" in g.lower()), None)
    ba = next((g for g in names if "basilica" in g.lower() and "peter" in g.lower()), None)
    if sq is None or ba is None:
        raise ValueError("this crate has no St Peter's Square and Basilica pair (the small crate?)")
    out = {}
    for key, g in (("square", sq), ("basilica", ba)):
        rows = np.flatnonzero(gt["group"] == g)
        centres = np.array([camera_center(gt["R"][k], gt["t"][k]) for k in rows])
        pts, rgb = _gt_points(rome["root"], g, max_points)
        out[key] = dict(pts=pts, colors=rgb, centres=centres, Rs=gt["R"][rows], index=gt["index"][rows])
    common = np.intersect1d(out["square"]["index"], out["basilica"]["index"])
    pick = lambda d: d["centres"][[int(np.flatnonzero(d["index"] == i)[0]) for i in common]].reshape(-1, 3)
    out["links_square"], out["links_basilica"] = pick(out["square"]), pick(out["basilica"])
    if len(common) < 3:
        raise ValueError(f"only {len(common)} photos are placed in both St Peter's surveys, umeyama needs 3")
    return out
