"""Task 4.4 · Compare the three trajectories.  PROVIDED: you do not need to
change this file.

The notebook (task4.ipynb) calls compare() with your Task 4.1 functions. From
a terminal, the same comparison runs on the functions in your saved notebooks:

    python -m provided.task4.compare_trackers              # frames 270 to 413, scale 0.25
    python -m provided.task4.compare_trackers --retrack    # ignore the saved tracks

The whole pipeline runs three times on the same trackable points, once per
tracker (one-step, iterative, SIFT-guided). Each set of tracks goes through
the provided camera-motion fit and flight code (provided/task4/run.py), and the three
results are drawn together in outputs/task4/compare/:

    ground_track.png    the three ground tracks, seen from above (X to the right, Y up)
    summary.csv         one row of numbers per tracker

Tracking is the slow part (several minutes), so the tracks are saved in
outputs/task4/compare/ and reused while your code stays the same; any change
to your functions makes the next run track again.
"""

import argparse
import csv
import hashlib
import inspect
import json
import time

import numpy as np

from provided.io import load_frame, load_tracks, save_tracks
from provided.plotting import plot_ground_tracks
from provided.runner import FRAMES, attempt, output_dir, task_output
from provided.task4.run import (SCALE, fit_all_pairs, fly, frame_pairs, load_constants, motion_arrays,
                       summarise, trackers_from, warn_if_slow)


def compare_dir():
    return output_dir(task_output(4) / "compare")


def code_key(trackers, track_pair, scale):
    """A hash of your functions' source, so saved tracks are reused only while it is unchanged."""
    from provided.notebook import FUNCTIONS
    functions = {id(f): f for f in [track_pair, *trackers.values()]}
    for f in list(functions.values()):
        for name in FUNCTIONS:
            g = getattr(f, "__globals__", {}).get(name)
            if callable(g):
                functions[id(g)] = g
    digest = hashlib.sha256(f"scale={scale}".encode())
    for f in sorted(functions.values(), key=lambda f: f.__name__):
        try:
            digest.update(inspect.getsource(f).encode("utf-8"))
        except (OSError, TypeError):
            digest.update(getattr(getattr(f, "__code__", None), "co_code", b""))
    return digest.hexdigest()


def track_all(trackers, track_pair, scale=SCALE):
    """Track every consecutive pair with every tracker, from the same points.

    Returns ({tracker: {(a, b): (points0, points1)}}, (height, width)), or
    None while your code is still a TODO.
    """
    pairs = frame_pairs()
    print(f"tracking {len(pairs)} pairs with {len(trackers)} trackers (several minutes)")
    tracks = {name: {} for name in trackers}
    image1 = load_frame(FRAMES, pairs[0][0], scale)
    for k, (a, b) in enumerate(pairs):
        image0, image1 = image1, load_frame(FRAMES, b, scale)
        counts = []
        started = time.perf_counter()
        for name, tracker in trackers.items():
            result = attempt(track_pair, image0, image1, tracker)
            if result is None:
                return None
            tracks[name][(a, b)] = result
            counts.append(f"{name} {len(result[0])}")
        if k == 0:
            warn_if_slow((time.perf_counter() - started) / len(trackers), len(pairs), len(trackers))
        if k % 20 == 0 or k == len(pairs) - 1:
            print(f"  {a} to {b}: tracks kept: {', '.join(counts)}")
    return tracks, image1.shape


def load_or_track(trackers, track_pair, scale=SCALE, retrack=False):
    out = compare_dir()
    key = code_key(trackers, track_pair, scale)
    paths = {name: out / f"tracks_{name}.npz" for name in trackers}
    if not retrack and all(p.exists() for p in paths.values()):
        loaded = {name: load_tracks(p) for name, p in paths.items()}
        info = next(iter(loaded.values()))[1]
        if all(str(i.get("code_key", "")) == key for _, i in loaded.values()):
            print("reusing the saved tracks (your code has not changed since they were made)")
            return {name: t for name, (t, _) in loaded.items()}, info
        print("your code changed since the saved tracks were made: tracking again")
    result = track_all(trackers, track_pair, scale)
    if result is None:
        return None
    tracks, (h, w) = result
    info = {"scale": scale, "width": w, "height": h, "code_key": key}
    for name, path in paths.items():
        save_tracks(path, tracks[name], **info)
    return tracks, info


def fill_gaps(rows, first, last):
    """A motion-table row for every pair first -> last; pairs that could not be
    fitted count as no motion (zoom 1), so the chain of frames stays complete."""
    by_frame = {r["frame"]: r for r in rows}
    missing = [a for a in range(first, last) if a not in by_frame]
    for a in missing:
        by_frame[a] = {"frame": a, "next": a + 1, "tracks": 0, "zoom": 1.0, "rotation_deg": 0.0,
                       "shift_x": 0.0, "shift_y": 0.0, "inlier_fraction": 0.0,
                       "median_residual": np.nan}
    return [by_frame[a] for a in range(first, last)], missing


def compare(trackers, track_pair, scale=SCALE, retrack=False, constants=None):
    """Task 4.4: the whole pipeline once per tracker, and the three results side by side.

    Writes outputs/task4/compare/ground_track.png and summary.csv. Returns the
    summary rows, or None while your code is a TODO.
    """
    constants = load_constants() if constants is None else constants
    loaded = load_or_track(trackers, track_pair, scale, retrack)
    if loaded is None:
        return None
    tracks, info = loaded
    centre = np.array([(float(info["width"]) - 1) / 2, (float(info["height"]) - 1) / 2])
    flights, table = {}, []
    for name, pairs in tracks.items():
        rows, fits = fit_all_pairs(pairs, centre, verbose=False)
        first, last = min(pairs)[0], max(pairs)[1]
        rows, missing = fill_gaps(rows, first, last)
        if missing:
            print(f"  {name}: {len(missing)} pairs had too few tracks and count as no motion")
        flight = fly(*motion_arrays(rows), constants, float(info["scale"]), verbose=not flights)
        flights[name] = flight
        fitted_rows = [r for r in rows if r["tracks"] > 0]
        table.append({"tracker": name, **summarise(flight), "pairs_without_fit": len(missing),
                      "median_tracks": float(np.median([r["tracks"] for r in rows])),
                      "median_inlier_fraction": float(np.median(
                          [r["inlier_fraction"] for r in fitted_rows])),
                      "median_residual_px": float(np.median(
                          [r["median_residual"] for r in fitted_rows]))})
    out = compare_dir()
    length = next(iter(flights.values()))["length_unit"]
    positions = {name: f["positions"] for name, f in flights.items()}
    if all(f["horizontal"] for f in flights.values()):
        plot_ground_tracks(positions, out / "ground_track.png", "ground track, by tracker",
                           units=length)
    else:
        print("  [info] X and Y need the focal length: ground-track plot skipped")
    columns = ["tracker", "median_tracks", "median_inlier_fraction", "median_residual_px",
               "pairs_without_fit", "start_Z", "end_Z", "duration", "horizontal_drift",
               "max_descent_speed", "length_unit", "time_unit"]
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    print("  saved outputs/task4/compare/summary.csv")
    for row in table:
        print(f"  {row['tracker']:>12}: end Z {row['end_Z']:.4g} {length}, drift "
              f"{row['horizontal_drift']:.4g} {length}, median residual "
              f"{row['median_residual_px']:.2f} px")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description="Task 4.4: compare the three trajectories")
    parser.add_argument("--scale", type=float, default=SCALE)
    parser.add_argument("--retrack", action="store_true", help="ignore the saved tracks")
    args = parser.parse_args(argv)
    from provided.notebook import load
    fn = load(4)
    for problem in fn._problems:
        print(f"  problem in your notebooks: {problem}")
    compare(trackers_from(fn), fn.track_pair, args.scale, args.retrack)
    print("done")


if __name__ == "__main__":
    main()
