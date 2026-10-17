"""Plotting helpers for the provided steps.  PROVIDED.

Every function saves a PNG in outputs/ and closes the figure. In a notebook
the saved PNG is then shown below the cell, so what you see is exactly the
file the report uses; outside a notebook nothing needs a display.
"""

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:      # scripts and tests: no display needed
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _show_in_notebook(path):
    if "ipykernel" not in sys.modules:
        return
    try:
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    except Exception:                    # never let a display problem stop a step
        pass


def save(fig, path, show=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    try:
        shown = path.resolve().relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"  saved {shown}")
    if show:
        _show_in_notebook(path)
    return path


def show_images(images, path, title=None):
    """Side-by-side images from a {title: image} dict.

    Images with negative values (e.g. Ix, Iy) get a diverging colour map
    centred on zero; the others are shown in grayscale.
    """
    fig, axes = plt.subplots(1, len(images), figsize=(4.2 * len(images), 3.6), squeeze=False)
    for ax, (name, img) in zip(axes[0], images.items()):
        img = np.asarray(img)
        if img.min() < 0:
            lim = float(np.abs(img).max()) or 1.0
            shown = ax.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim)
        else:
            shown = ax.imshow(img, cmap="gray")
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(name)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    return save(fig, path)


def plot_points(image, points, path, title="points"):
    """Points marked over an image (e.g. trackable points).  PROVIDED."""
    points = np.asarray(points)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap="gray")
    if len(points):
        ax.plot(points[:, 0], points[:, 1], "o", color="orange", markersize=4)
    ax.set_title(f"{title}  ({len(points)} points)")
    ax.axis("off")
    return save(fig, path)


def plot_histograms(images, path, bins=64):
    """Intensity histograms from a {label: image} dict."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, img in images.items():
        ax.hist(np.ravel(img), bins=bins, range=(0, 1), histtype="step", label=name)
    ax.set_xlabel("intensity")
    ax.set_ylabel("pixel count")
    ax.legend()
    return save(fig, path)


def _arrow_scale(magnitudes, target_px=45.0, low=1.0, high=300.0):
    """A multiplier that makes typical displacements about `target_px` long.

    Real per-frame displacements range from well under a pixel (one-step on
    an easy pair) to tens of pixels (SIFT-guided on a hard pair), so drawing
    them at their true length makes most figures unreadable. This picks one
    multiplier from the 90th percentile of the magnitudes actually in the
    figure, so arrows are visible without changing the vectors' directions
    or their length relative to each other.
    """
    magnitudes = np.asarray(magnitudes)
    typical = np.percentile(magnitudes, 90) if magnitudes.size else 0.0
    if typical < 1e-6:
        return high
    return float(np.clip(target_px / typical, low, high))


def plot_tracks(image, points0, points1, valid, path, title="tracked points", show=True):
    """Displacement vectors from points0 to points1 over an image."""
    points0, points1, valid = np.asarray(points0), np.asarray(points1), np.asarray(valid, bool)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap="gray")
    d = points1 - points0
    scale = _arrow_scale(np.linalg.norm(d[valid], axis=1)) if valid.any() else 1.0
    ax.quiver(points0[valid, 0], points0[valid, 1], d[valid, 0] * scale, d[valid, 1] * scale,
              color="orange", angles="xy", scale_units="xy", scale=1, width=0.003)
    ax.plot(points0[~valid, 0], points0[~valid, 1], "rx", markersize=5, label="failed")
    ax.set_title(f"{title}  ({valid.sum()} ok, {(~valid).sum()} failed, "
                 f"arrows magnified {scale:.0f}x)")
    ax.axis("off")
    return save(fig, path, show)


def plot_tracker_comparison(image, points0, results, path, title="tracker comparison"):
    """One image, one set of starting points, several trackers overlaid.  PROVIDED.

    results : {name: (tracked, valid)} for the same points0, e.g. the output of
              one_step, iterative and sift_guided on the same pair. Each tracker's
              displacement vectors are drawn in its own colour, so the three can be
              compared at a glance instead of on three separate figures.
    """
    points0 = np.asarray(points0)
    deltas = {}
    all_magnitudes = []
    for name, (tracked, valid) in results.items():
        tracked, valid = np.asarray(tracked), np.asarray(valid, dtype=bool)
        d = tracked - points0
        deltas[name] = (d, valid)
        if valid.any():
            all_magnitudes.append(np.linalg.norm(d[valid], axis=1))
    scale = _arrow_scale(np.concatenate(all_magnitudes)) if all_magnitudes else 1.0

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.imshow(image, cmap="gray")
    for (name, (d, valid)), colour in zip(deltas.items(), COLOURS):
        ax.quiver(points0[valid, 0], points0[valid, 1], d[valid, 0] * scale, d[valid, 1] * scale,
                  color=colour, angles="xy", scale_units="xy", scale=1, width=0.0028,
                  label=f"{name} ({valid.sum()} ok, {(~valid).sum()} failed)")
    ax.set_title(f"{title}  (arrows magnified {scale:.0f}x, same factor for every tracker)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.axis("off")
    return save(fig, path)


def plot_flow(image, flow, path, step=16, title="optical flow"):
    """Dense flow field (H, W, 2) drawn as arrows every `step` pixels."""
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    u, v = flow[ys, xs, 0], flow[ys, xs, 1]
    scale = _arrow_scale(np.hypot(u, v))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap="gray")
    ax.quiver(xs, ys, u * scale, v * scale, color="orange", angles="xy",
              scale_units="xy", scale=1, width=0.003)
    ax.set_title(f"{title}  (arrows magnified {scale:.0f}x)")
    ax.axis("off")
    return save(fig, path)


# The world frame of Task 4 (see the axes figure in the handout): right-handed,
# fixed to the FIRST image. X points to the right of that image, Y toward its
# top and Z up (the altitude). Image y points down, world Y points up.
AXIS_NAMES = ("X, right in the first image", "Y, up in the first image", "Z, altitude")
COLOURS = ("tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple")


def plot_trajectories_3d(trajectories, path, title="3D trajectory", units="m"):
    """One or more 3D paths from a {label: (N, 3) array of [X, Y, Z]} dict.

    Each path is drawn with its ground track (the path dropped to Z = 0) as a
    dashed line of the same colour. A circle marks the first frame and a
    triangle the last.
    """
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection="3d")
    everything = np.vstack([np.asarray(p, dtype=float) for p in trajectories.values()])
    for (name, p), colour in zip(trajectories.items(), COLOURS):
        p = np.asarray(p, dtype=float)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=colour, lw=2, label=name)
        ax.plot(p[:, 0], p[:, 1], np.zeros(len(p)), color=colour, lw=0.8, ls="--", alpha=0.7)
        ax.scatter(*p[0], color=colour, marker="o", s=24)
        ax.scatter(*p[-1], color=colour, marker="^", s=30)
    # the same scale for X and Y, so the horizontal path is not distorted
    centre = (everything[:, :2].max(axis=0) + everything[:, :2].min(axis=0)) / 2
    half = max(float(np.ptp(everything[:, :2], axis=0).max()) / 2, 1e-9) * 1.05
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(0, float(everything[:, 2].max()) * 1.05 or 1.0)
    for axis, name in zip("xyz", AXIS_NAMES):
        getattr(ax, f"set_{axis}label")(f"{name} ({units})")
    ax.set_title(title + "\n(o first frame, ^ last frame, dashed: ground track; "
                 "X and Y share one scale)", fontsize=10)
    ax.legend(loc="upper left")
    return save(fig, path)


def plot_trajectory_3d(positions, path, title="3D trajectory", units="m"):
    """3D path (N, 3) of [X, Y, Z] with its ground track."""
    return plot_trajectories_3d({"lander": positions}, path, title, units)


def plot_ground_tracks(trajectories, path, title="ground track", units="m"):
    """The paths seen from above: X to the right, Y up, equal scales."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for (name, p), colour in zip(trajectories.items(), COLOURS):
        p = np.asarray(p, dtype=float)
        ax.plot(p[:, 0], p[:, 1], color=colour, lw=1.5, label=name)
        ax.plot(*p[0, :2], "o", color=colour)
        ax.plot(*p[-1, :2], "^", color=colour)
    ax.set_xlabel(f"{AXIS_NAMES[0]} ({units})")
    ax.set_ylabel(f"{AXIS_NAMES[1]} ({units})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title + " (seen from above)")
    ax.grid(alpha=0.3)
    ax.legend()
    return save(fig, path)


def plot_velocities(velocities, path, time_unit="s", speed_unit="m/s", title="velocity"):
    """v_X, v_Y and v_Z against time, from a {label: (times, (N, 3) velocities)} dict."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    for (name, (t, v)), colour in zip(velocities.items(), COLOURS):
        v = np.asarray(v, dtype=float)
        for k, ax in enumerate(axes):
            ax.plot(t, v[:, k], color=colour, marker=".", lw=1, label=name)
    for ax, component in zip(axes, ("$v_X$", "$v_Y$", "$v_Z$")):
        ax.set_ylabel(component)
        ax.axhline(0, color="gray", lw=0.6)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"{title} ({speed_unit})")
    axes[0].legend()
    axes[-1].set_xlabel(f"time ({time_unit})")
    return save(fig, path)


def plot_series(series, path, xlabel, ylabel, title=""):
    """Line plot of several {label: (x, y)} curves, each with its own x."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for (name, (x, y)), colour in zip(series.items(), COLOURS):
        ax.plot(x, y, color=colour, marker=".", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return save(fig, path)


def plot_curves(x, curves, path, xlabel, ylabel, title=""):
    """Line plot of several {label: y-values} curves against x."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for name, y in curves.items():
        ax.plot(x, y, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return save(fig, path)
