"""Task 3, the provided steps the notebook runs.  PROVIDED.

Adds an iterative curve to the Task 2 error sweep, then compares one-step,
iterative and SIFT-guided tracking on three pairs of the descent: 375 to 376
(small motion), 300 to 301 (large motion) and 409 to 410 (near touchdown).
Figures are shown below the cell and saved, with the numbers, in outputs/task3/.
"""

import numpy as np

from provided.io import load_frame
from provided.plotting import plot_series, plot_tracker_comparison, plot_tracks
from provided.runner import FRAMES, attempt, save_json, task_output
from provided.sift import sift_matches, track_with_sift
from provided.task2.steps import POINTS, SCALE, SHIFTS, TRACK_RADIUS, sweep, sweep_setup

ITERATIONS = 10
PAIRS = {"small_motion": (375, 376), "large_motion": (300, 301), "touchdown": (409, 410)}


def tracking_statistics(points0, points1, valid):
    """The numbers Task 3.4 asks you to report for every frame pair.

    Displacement is the distance between a point and its tracked position,
    measured over valid tracks only. Returns a dict with the keys
    "mean_displacement", "median_displacement", "max_displacement" and
    "failed_track_rate" (fraction of tracks that failed, 0 to 1).
    """
    valid = np.asarray(valid, dtype=bool)
    d = np.linalg.norm(np.asarray(points1) - np.asarray(points0), axis=1)[valid]
    nan = float("nan")
    return {"mean_displacement": float(d.mean()) if len(d) else nan,
            "median_displacement": float(np.median(d)) if len(d) else nan,
            "max_displacement": float(d.max()) if len(d) else nan,
            "failed_track_rate": float(1 - valid.mean()) if len(valid) else nan}


def error_sweep(select_features, spatial_gradients, one_step_flow, track_point):
    """Task 3.1: the Task 2.3 error sweep, with your iterative tracker added."""
    def iterative_estimator(image0, image1, point, grad0=None, window_radius=TRACK_RADIUS):
        result = attempt(track_point, image0, image1, point, window_radius, ITERATIONS, 1e-3,
                         grad0)
        if result is None or not result[1]:
            return None
        return tuple(np.asarray(result[0]) - np.asarray(point))

    image, points, grad0 = sweep_setup(select_features, spatial_gradients)
    if points is None or grad0 is None:
        return None
    curves = sweep({"one-step": one_step_flow, "iterative": iterative_estimator}, image, points,
                   grad0, window_radius=TRACK_RADIUS)
    save_json(task_output(3) / "error_sweep.json",
              {name: values.tolist() for name, (_, values) in curves.items()})
    plot_series(curves, task_output(3) / "error_sweep.png", "true shift (px)", "mean error (px)",
                "one-step vs iterative flow error against shift size")
    for k, shift in enumerate(SHIFTS):
        print(f"    shift {shift:5.2f} px: one-step {curves['one-step'][1][k]:.3f} px, "
              f"iterative {curves['iterative'][1][k]:.3f} px")
    return curves


def compare_pair(label, select_features, track_points):
    """Task 3.4: track every selected point of one pair three ways and compare."""
    a, b = PAIRS[label]
    i0, i1 = load_frame(FRAMES, a, SCALE), load_frame(FRAMES, b, SCALE)
    points = attempt(select_features, i0, POINTS["max_points"], POINTS["window_radius"],
                     POINTS["min_distance"])
    if points is None:
        return None
    matches = sift_matches(i0, i1)
    runs = {"one_step": lambda: attempt(track_points, i0, i1, points, TRACK_RADIUS, 1),
            "iterative": lambda: attempt(track_points, i0, i1, points, TRACK_RADIUS, ITERATIONS),
            "sift_guided": lambda: attempt(track_with_sift, track_points, i0, i1, points, matches,
                                           TRACK_RADIUS, max_iterations=ITERATIONS)}
    results, tracks = {}, {}
    for name, run in runs.items():
        tracked = run()
        if tracked is None:
            continue
        tracks[name.replace("_", "-")] = tracked
        results[name] = tracking_statistics(points, *tracked)
        s = results[name]
        print(f"  {label} ({a} to {b}) {name.replace('_', '-'):>11}: mean "
              f"{s['mean_displacement']:.2f} px, failed-track rate {s['failed_track_rate']:.2f}")
        plot_tracks(i0, points, tracked[0], tracked[1], task_output(3) / f"{label}_{name}.png",
                    title=f"{name.replace('_', ' ')}, frames {a} to {b}", show=False)
    save_json(task_output(3) / f"{label}_statistics.json", results)
    if len(tracks) == len(runs):
        plot_tracker_comparison(i0, points, tracks, task_output(3) / f"{label}_compare.png",
                                title=f"one-step vs iterative vs SIFT-guided, frames {a} to {b}")
    return results
