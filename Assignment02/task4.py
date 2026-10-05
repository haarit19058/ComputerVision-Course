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
    plt.imshow(canvas)
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
    desc_a = np.asarray(desc_a, dtype=np.float64)
    desc_b = np.asarray(desc_b, dtype=np.float64)

    if desc_a.shape[0] == 0 or desc_b.shape[0] == 0:
        return np.zeros((desc_a.shape[0], desc_b.shape[0]))
    
    sq_a = np.sum(desc_a**2, axis=1, keepdims=True)
    sq_b = np.sum(desc_b**2, axis=1)
    dot = np.dot(desc_a, desc_b.T)
    dist_sq = sq_a + sq_b - 2 * dot
    
    return np.sqrt(np.maximum(dist_sq, 0))


def match_ratio_test(desc_scene: np.ndarray, desc_template: np.ndarray,
                     ratio: float = 0.9) -> np.ndarray:
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
    distances = descriptor_distances(desc_scene, desc_template)
    
    if distances.shape[0] == 0 or distances.shape[1] < 2:
        return np.zeros((0, 2), dtype=int)
    
    # Sort along the template axis (axis=1) for each scene keypoint
    sorted_idx = np.argsort(distances, axis=1)
    best_idx = sorted_idx[:, 0]
    second_best_idx = sorted_idx[:, 1]
    
    best_dist = distances[np.arange(distances.shape[0]), best_idx]
    second_dist = distances[np.arange(distances.shape[0]), second_best_idx]
    
    mask = best_dist < (ratio * second_dist)
    
    scene_indices = np.where(mask)[0]
    template_indices = best_idx[mask]
    
    return np.column_stack((template_indices, scene_indices))


def predict_centres(kp_template: np.ndarray, kp_scene: np.ndarray,
                    matches: np.ndarray, ori_template: np.ndarray,
                    ori_scene: np.ndarray, template_shape: tuple) -> np.ndarray:
    """Turn every single match into a full guess at where Waldo is.

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

    y_t, x_t, sigma_t = kp_template[t_idx].T
    y_s, x_s, sigma_s = kp_scene[s_idx].T

    theta_t = ori_template[t_idx]
    theta_s = ori_scene[s_idx]

    s = sigma_s / sigma_t
    dtheta = np.radians(theta_s - theta_t)

    H_t, W_t = template_shape
    ox = W_t / 2.0 - x_t
    oy = H_t / 2.0 - y_t

    cos_d = np.cos(dtheta)
    sin_d = np.sin(dtheta)

    cx = x_s + s * (cos_d * ox - sin_d * oy)
    cy = y_s + s * (sin_d * ox + cos_d * oy)

    return np.column_stack((cx, cy, s))


def consensus_centre(predictions: np.ndarray, tol: float = 12.0,
                     min_votes: int = 8) -> tuple:
    """Find where the guesses pile up -- the DENSEST cluster, not the average.

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
    if len(predictions) < min_votes:
        return None, None, np.zeros(len(predictions), dtype=bool)

    centers = predictions[:, :2]
    
    # Broadcast differences to find all-pairs distances (M, M)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    counts = np.sum(dist < tol, axis=1)
    seed_idx = np.argmax(counts)

    if counts[seed_idx] < min_votes:
        return None, None, np.zeros(len(predictions), dtype=bool)

    mask = dist[seed_idx] < tol
    cluster = predictions[mask]

    med_cx, med_cy = np.median(cluster[:, :2], axis=0)
    med_s = np.median(cluster[:, 2])

    return (med_cx, med_cy), med_s, mask


def find_waldo(scene: np.ndarray, template: np.ndarray,
               upsample: float = 1.0) -> dict:
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
    h_orig, w_orig = template.shape
    th = max(1, int(round(h_orig * upsample)))
    tw = max(1, int(round(w_orig * upsample)))
    
    # Enlarge the template before detecting
    t_up = cv2.resize(template, (tw, th), interpolation=cv2.INTER_LINEAR)

    kp_t = with_unit_scale(dog_keypoints(t_up))
    kp_s = with_unit_scale(dog_keypoints(scene))

    if len(kp_t) == 0 or len(kp_s) == 0:
        return {"box": None, "n_matches": 0, "n_votes": 0,
                "matches": np.zeros((0, 2), int), "votes": np.zeros(0, bool),
                "kp_template": kp_t, "kp_scene": kp_s}


    ori_t = np.array([dominant_orientation(t_up, y, x, s) for y, x, s in kp_t])
    ori_s = np.array([dominant_orientation(scene, y, x, s) for y, x, s in kp_s])

    desc_t = describe_keypoints(t_up, kp_t, ori_t)
    desc_s = describe_keypoints(scene, kp_s, ori_s)

    matches = match_ratio_test(desc_s, desc_t)
    preds = predict_centres(kp_t, kp_s, matches, ori_t, ori_s, (th, tw))

    center, scale_s, votes_mask = consensus_centre(preds)

    # Scale the template keypoints back down to original size for draw_matches
    kp_t_orig = kp_t.copy()
    kp_t_orig[:, :2] /= upsample
    kp_t_orig[:, 2] /= upsample

    if center is None:
        return {"box": None, "n_matches": len(matches), "n_votes": 0,
                "matches": matches, "votes": votes_mask,
                "kp_template": kp_t_orig, "kp_scene": kp_s}

    box = box_from_centre(center[0], center[1], scale_s * upsample, (h_orig, w_orig))

    return {"box": box, "n_matches": len(matches), "n_votes": int(np.sum(votes_mask)),
            "matches": matches, "votes": votes_mask,
            "kp_template": kp_t_orig, "kp_scene": kp_s}


if __name__ == "__main__":

    template_rgb = load_image_as_rgb("imgs/waldo_template.png")
    template = to_grayscale(template_rgb)
    scene_names = ["scene_easy", "scene_medium", "scene_rotated", "scene_hard"]

    print("{:<16}{:>9}{:>8}{:>9}{:>10}".format(
        "scene", "matches", "votes", "IoU", "agree?"))
    for name in scene_names:
        scene_rgb = load_image_as_rgb("imgs/{}.png".format(name))
        scene = to_grayscale(scene_rgb)
        truth = load_ground_truth(name)

        found = find_waldo(scene, template)
        correlation = detect_multiscale(scene_rgb, template_rgb)

        feature_box = found["box"]
        corr_box = None
        if correlation:
            best = correlation[0]
            corr_box = (best["x"], best["y"], best["w"], best["h"])

        overlap = iou(feature_box, truth) if feature_box else 0.0
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


    # Question 3.8

    scene_easy = to_grayscale(load_image_as_rgb("imgs/scene_easy.png"))

    for upsample in [1.5, 2.0, 3.0, 4.0]:
        res = find_waldo(scene_easy, template, upsample=upsample)
        n_kp = len(res["kp_template"])
        n_matches = res["n_matches"]
        print(f"upsample={upsample:.1f} | Template Keypoints: {n_kp} | Matches: {n_matches}")