"""Task 2, the provided steps the notebook runs.  PROVIDED.

The error sweep shifts one frame by known amounts from 0.1 to 8 px and
measures how far your one-step flow is off; the real pairs run it on two
pairs of the descent. Figures are shown below the cell and saved, with the
numbers, in outputs/task2/. Task 3 reuses sweep() for its iterative curve.
"""

import numpy as np

from provided.io import load_frame
from provided.plotting import plot_series, plot_tracks
from provided.runner import FRAMES, attempt, save_json, task_output
from provided.sampling import translate

SHIFTS = np.geomspace(0.1, 8.0, 12)
SCALE = 0.25
POINTS = dict(max_points=60, window_radius=3, min_distance=10)
TRACK_RADIUS = 7


def flow_error(tracker, image, grad0, points, dx, dy, **kw):
    """Mean error, over `points`, of a tracker's (u, v) estimate of a
    synthetic shift (dx, dy) applied to `image`.

    tracker : a function (image0, image1, point, **kw) -> (u, v) or None,
              such as your one_step_flow.
    """
    shifted = translate(image, dx, dy)
    errors = []
    for p in points:
        result = attempt(tracker, image, shifted, p, grad0=grad0, **kw)
        if result is None:
            continue
        u, v = result
        errors.append(np.hypot(u - dx, v - dy))
    return float(np.mean(errors)) if errors else float("nan")


def sweep(trackers, image, points, grad0, shifts=SHIFTS, **kw):
    """{name: (shifts, mean error)} for every (name, tracker) in `trackers`."""
    return {name: (shifts, np.array([flow_error(tracker, image, grad0, points, s, 0.0, **kw)
                                     for s in shifts]))
            for name, tracker in trackers.items()}


def sweep_setup(select_features, spatial_gradients, frame=300):
    """The frame, points and gradients the error sweeps of Tasks 2 and 3 use."""
    image = load_frame(FRAMES, frame, SCALE)
    points = attempt(select_features, image, POINTS["max_points"], POINTS["window_radius"],
                     POINTS["min_distance"])
    grad0 = attempt(spatial_gradients, image)
    return image, points, grad0


def error_sweep(select_features, spatial_gradients, one_step_flow):
    """Task 2.3: one-step flow error against the size of a synthetic shift."""
    image, points, grad0 = sweep_setup(select_features, spatial_gradients)
    if points is None or grad0 is None:
        return None
    curves = sweep({"one-step": one_step_flow}, image, points, grad0, window_radius=TRACK_RADIUS)
    save_json(task_output(2) / "error_sweep.json",
              {name: values.tolist() for name, (_, values) in curves.items()})
    plot_series(curves, task_output(2) / "error_sweep.png", "true shift (px)", "mean error (px)",
                "one-step flow error against shift size")
    for shift, error in zip(SHIFTS, curves["one-step"][1]):
        print(f"    shift {shift:5.2f} px: mean error {error:.3f} px")
    return curves


def real_pair(label, a, b, select_features, spatial_gradients, one_step_flow):
    """Task 2.3: one-step flow of every selected point of frame a, on the pair a -> b."""
    i0, i1 = load_frame(FRAMES, a, SCALE), load_frame(FRAMES, b, SCALE)
    points = attempt(select_features, i0, POINTS["max_points"], POINTS["window_radius"],
                     POINTS["min_distance"])
    grad0 = attempt(spatial_gradients, i0)
    if points is None or grad0 is None:
        return None
    tracked, ok = [], []
    for p in points:
        result = attempt(one_step_flow, i0, i1, p, TRACK_RADIUS, grad0)
        tracked.append(p if result is None else p + np.array(result))
        ok.append(result is not None)
    tracked, ok = np.array(tracked), np.array(ok)
    step = float(np.mean(np.linalg.norm(tracked[ok] - points[ok], axis=1))) if ok.any() \
        else float("nan")
    print(f"  {label} ({a} to {b}): {int(ok.sum())} of {len(ok)} windows solved, "
          f"mean step {step:.2f} px")
    plot_tracks(i0, points, tracked, ok, task_output(2) / f"one_step_{label}.png",
                title=f"one-step flow, frames {a} to {b}")
    return step
