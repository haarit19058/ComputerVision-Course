"""Task 4, the provided pipeline: track the descent, fit the motion, reconstruct the flight.
PROVIDED: you do not need to change this file.

The notebook (task4.ipynb) calls read_motion() for Task 4.2 and
reconstruct_flight() for Task 4.3 with your Task 4.1 functions. From a
terminal, the same pipeline runs on the functions in your saved notebooks:

    python -m provided.task4.run                        # SIFT-guided tracking, frames 270 to 413
    python -m provided.task4.run --tracker one-step      # or iterative

It tracks every consecutive frame pair with one of your trackers, fits the
camera motion of every pair (provided/task4/motion_model.py), and turns the
result into altitude, velocity and the 3D trajectory (provided/task4/trajectory.py). It
writes outputs/task4/motion.csv, outputs/task4/trajectory.csv, summary.json
and the figures.
"""

import argparse
import csv
import json
import time

import numpy as np

from provided.io import list_frames, load_frame, save_tracks
from provided.plotting import plot_curves, plot_ground_tracks, plot_trajectory_3d, plot_velocities
from provided.runner import CONSTANTS, FRAMES, attempt, save_json, task_output
from provided.task4.motion_model import robust_fit_similarity
from provided.task4.trajectory import (frame_times, ground_displacements, metric_altitudes,
                              relative_altitudes, step_multiples, time_to_touchdown, trajectory,
                              velocities)

SCALE = 0.25
TRACKER_NAMES = ("one-step", "iterative", "sift-guided")
COLUMNS = ["frame", "next", "tracks", "zoom", "rotation_deg", "shift_x", "shift_y",
           "inlier_fraction", "median_residual"]


def load_constants():
    """data/mission_constants.json as a dict ({} if it is missing)."""
    return json.loads(CONSTANTS.read_text(encoding="utf-8")) if CONSTANTS.exists() else {}


def frame_pairs():
    frames = sorted(list_frames(FRAMES))
    return [(a, a + 1) for a in frames if a + 1 in frames]


def warn_if_slow(seconds, pairs, trackers=1):
    """After the first pair: say how long the whole run will take, if it is slow."""
    total = seconds * pairs * trackers
    if seconds * trackers > 3:
        print(f"  the first pair took {seconds * trackers:.1f} s, so all {pairs} pairs take about "
              f"{total / 60:.0f} minutes. If your select_features loops over every pixel, the "
              "speed tip in its docstring makes it much faster.")


def track_descent(track_pair, tracker, name="sift-guided", scale=SCALE):
    """Your track_pair, with one of your trackers, on every consecutive frame pair.

    Returns ({(a, b): (points0, points1)}, (height, width)), or None while
    track_pair or the tracker is still a TODO.
    """
    pairs = frame_pairs()
    print(f"tracking {len(pairs)} pairs with the {name} tracker (a few minutes)")
    tracks = {}
    image1 = load_frame(FRAMES, pairs[0][0], scale)
    for k, (a, b) in enumerate(pairs):
        image0, image1 = image1, load_frame(FRAMES, b, scale)
        started = time.perf_counter()
        result = attempt(track_pair, image0, image1, tracker)
        if result is None:
            return None
        if k == 0:
            warn_if_slow(time.perf_counter() - started, len(pairs))
        tracks[(a, b)] = result
        if k % 20 == 0 or k == len(pairs) - 1:
            print(f"  {a} to {b}: {len(result[0])} tracks")
    h, w = image1.shape
    save_tracks(task_output(4) / "tracks.npz", tracks, scale=scale, width=w, height=h)
    return tracks, (h, w)


def fit_all_pairs(tracks, centre, verbose=True):
    """The provided robust similarity fit for every frame pair of a tracks dict.

    Returns (rows, fits): one motion-table row per fitted pair (the columns of
    motion.csv) and {frame: (points0, residuals, inliers)}. Pairs with fewer
    than 3 tracks are skipped.
    """
    rows, fits = [], {}
    for (a, b), (p0, p1) in sorted(tracks.items()):
        if len(p0) < 3:
            if verbose:
                print(f"  {a} -> {b}: only {len(p0)} tracks, skipped")
            continue
        model, inliers, residuals = robust_fit_similarity(p0, p1, centre)
        fits[a] = (p0, residuals, inliers)
        rows.append({"frame": a, "next": b, "tracks": len(p0), "zoom": model["zoom"],
                     "rotation_deg": np.degrees(model["rotation"]),
                     "shift_x": model["shift"][0], "shift_y": model["shift"][1],
                     "inlier_fraction": float(np.mean(inliers)),
                     "median_residual": float(np.median(residuals[inliers]))})
    return rows, fits


def motion_arrays(rows):
    """(frames, zooms, rotations in radians, shifts) from motion-table rows."""
    frames = [int(r["frame"]) for r in rows] + [int(rows[-1]["next"])]
    get = lambda key: np.array([float(r[key]) for r in rows])
    return (np.array(frames), get("zoom"), np.deg2rad(get("rotation_deg")),
            np.c_[get("shift_x"), get("shift_y")])


def fly(frames, zooms, rotations, shifts, constants, scale, verbose=True):
    """The provided Task 4 flight code, in order, on one motion table.

    Returns a dict with the times, positions [X, Y, Z], velocities
    [v_X, v_Y, v_Z], time to touchdown and units.
    """
    say = print if verbose else (lambda *a: None)
    focal = constants.get("focal_length_px")
    interval = constants.get("frame_interval_s")
    anchor_frame, anchor_altitude = constants.get("anchor_frame"), constants.get("anchor_altitude_m")

    altitude = relative_altitudes(zooms)
    length_unit = "relative"
    if anchor_frame is not None and anchor_altitude is not None and anchor_frame in frames:
        altitude = metric_altitudes(altitude, int(np.where(frames == anchor_frame)[0][0]),
                                    float(anchor_altitude))
        length_unit = "m"
    else:
        say("  [info] no altitude reading in mission_constants.json: lengths stay relative")
    steps = step_multiples(zooms)
    times = frame_times(steps, float(interval) if interval else 1.0)
    time_unit = "s" if interval else "frame intervals"

    positions = None
    if focal:
        d = ground_displacements(shifts, rotations, altitude, float(focal) * scale)
        positions = trajectory(d, altitude)
    else:
        say("  [info] no focal length in mission_constants.json: X and Y skipped")
    horizontal = positions is not None
    if positions is None:
        positions = np.c_[np.zeros((len(altitude), 2)), altitude]
    v = velocities(positions, times)
    tau = time_to_touchdown(altitude, times)
    speed_unit = "m/s" if (length_unit, time_unit) == ("m", "s") else \
        f"{length_unit} per {'s' if time_unit == 's' else 'frame interval'}"
    return {"frames": frames, "times": times, "steps": steps, "positions": positions,
            "horizontal": horizontal, "velocities": v, "tau": tau,
            "length_unit": length_unit, "time_unit": time_unit, "speed_unit": speed_unit}


def summarise(flight):
    """A few numbers that describe one flight."""
    p, v = flight["positions"], flight["velocities"]
    return {"frames": [int(flight["frames"][0]), int(flight["frames"][-1])],
            "length_unit": flight["length_unit"], "time_unit": flight["time_unit"],
            "start_Z": float(p[0, 2]), "end_Z": float(p[-1, 2]),
            "duration": float(flight["times"][-1]),
            "horizontal_drift": float(np.hypot(*(p[-1, :2] - p[0, :2]))),
            "max_descent_speed": float(np.max(-v[:, 2]))}


def read_motion(track_pair, tracker, name="sift-guided", scale=SCALE):
    """Task 4.2: track the descent, fit the camera motion of every pair, plot it.

    Writes outputs/task4/motion.csv and zoom.png, rotation.png, shift.png.
    Returns the motion-table rows, or None while your code is a TODO.
    """
    tracked = track_descent(track_pair, tracker, name, scale)
    if tracked is None:
        return None
    tracks, (h, w) = tracked
    centre = np.array([(w - 1) / 2, (h - 1) / 2])
    rows, _ = fit_all_pairs(tracks, centre)
    if not rows:
        print("no frame pairs were fitted")
        return None
    out = task_output(4)
    with open(out / "motion.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved outputs/task4/motion.csv ({len(rows)} pairs)")
    frame = [r["frame"] for r in rows]
    plot_curves(frame, {"zoom - 1": [100 * (r["zoom"] - 1) for r in rows]},
                out / "zoom.png", "frame", "zoom - 1 (%)", "image zoom per frame pair")
    plot_curves(frame, {"rotation": [r["rotation_deg"] for r in rows]},
                out / "rotation.png", "frame", "rotation (degrees)",
                "image rotation per frame pair")
    plot_curves(frame, {"shift x": [r["shift_x"] for r in rows],
                        "shift y": [r["shift_y"] for r in rows]},
                out / "shift.png", "frame", "shift (px)", "image shift per frame pair")
    return {"rows": rows, "scale": scale}


def reconstruct_flight(motion, constants=None):
    """Task 4.3: altitude, velocity, time to touchdown and the 3D path, from the motion.

    Writes outputs/task4/trajectory.csv, summary.json and the flight figures.
    Returns (flight, summary).
    """
    if motion is None:
        print("no motion yet: finish Task 4.1 and run the Task 4.2 cell first")
        return None, None
    constants = load_constants() if constants is None else constants
    out = task_output(4)
    flight = fly(*motion_arrays(motion["rows"]), constants, motion["scale"])
    times, positions, v, tau = (flight[k] for k in ("times", "positions", "velocities", "tau"))
    length, t_unit = flight["length_unit"], flight["time_unit"]
    plot_curves(times, {"Z": positions[:, 2]}, out / "altitude.png", f"time ({t_unit})",
                f"Z, altitude ({length})", "altitude of the lander")
    mid = (times[:-1] + times[1:]) / 2
    plot_velocities({"lander": (mid, v)}, out / "velocity.png", t_unit, flight["speed_unit"],
                    "velocity of the lander")
    finite = np.isfinite(tau)
    plot_curves(times[1:][finite], {"time to touchdown": tau[finite]},
                out / "time_to_touchdown.png", f"time ({t_unit})", f"tau ({t_unit})",
                "time to touchdown")
    if flight["horizontal"]:
        plot_trajectory_3d(positions, out / "trajectory_3d.png", "3D trajectory of the lander",
                           units=length)
        plot_ground_tracks({"lander": positions}, out / "ground_track.png", units=length)
    with open(out / "trajectory.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "X", "Y", "Z", "v_X", "v_Y", "v_Z", "tau"])
        for k in range(len(flight["frames"])):
            step = [""] * 4 if k == 0 else [*v[k - 1], tau[k - 1]]
            writer.writerow([flight["frames"][k], times[k], *positions[k], *step])
    summary = summarise(flight)
    save_json(out / "summary.json", summary)
    print("  summary: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                                    for k, v in summary.items()))
    return flight, summary


def trackers_from(fn):
    """{name: tracker} from your Task 4.1 functions."""
    return {"one-step": fn.one_step, "iterative": fn.iterative, "sift-guided": fn.sift_guided}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Task 4: reconstruct the flight")
    parser.add_argument("--tracker", choices=TRACKER_NAMES, default="sift-guided")
    parser.add_argument("--scale", type=float, default=SCALE)
    args = parser.parse_args(argv)
    from provided.notebook import load
    fn = load(4)
    for problem in fn._problems:
        print(f"  problem in your notebooks: {problem}")
    motion = read_motion(fn.track_pair, trackers_from(fn)[args.tracker], args.tracker, args.scale)
    if motion is not None:
        reconstruct_flight(motion)
    print("done")


if __name__ == "__main__":
    main()
