"""Task 4 · Reconstruct the Flight: the flight code.
PROVIDED: you do not need to change this file.

This module turns the camera motion of task4/motion_model.py into the
lander's altitude, time axis, 3D trajectory and velocity. task4/run.py calls
it for you; read it to see how your optical flow becomes a trajectory.

Inputs: for every frame pair k -> k+1 a zoom s_k, a rotation theta_k
(radians) and a shift t_k (pixels), and from data/mission_constants.json
(focal length, frame interval, one altitude).

Conventions (see the axes figure in the handout)
  * N frame pairs describe N + 1 frames; per-frame arrays have N + 1
    entries, per-step arrays N
  * image coordinates are pixels: x to the right, y DOWN (along the rows)
  * the world frame is right-handed, fixed to the FIRST image and in metres:
        X  to the right of the first image
        Y  toward the TOP of the first image (the opposite of image y)
        Z  the altitude, upward
    the lander starts above (X, Y) = (0, 0)
  * positions are [X, Y, Z]; velocities are [v_X, v_Y, v_Z], and v_Z < 0
    while the lander descends
  * heading: psi_0 = 0 and psi_{k+1} = psi_k + theta_k (the lander's turn
    about Z; it turns the image by theta_k)
"""

import numpy as np


def relative_altitudes(zooms):
    """Altitude of every frame relative to the first (Task 4.2).

    From equation (4.2), s_k = Z_k / Z_{k+1}: Z_0 = 1 and Z_{k+1} = Z_k / s_k.  Returns (N + 1,).
    """
    return np.concatenate([[1.0], 1.0 / np.cumprod(np.asarray(zooms, dtype=float))])


def step_multiples(zooms, window=9, min_rate=0.005):
    """How many frame intervals each step spans, judged from the zooms.

    Each step's log zoom is compared with the typical log zoom of the steps
    around it (within `window` steps); log zooms add up over time. Where the
    motion is too small to judge (typical log zoom below `min_rate`), the step
    counts as 1. Returns an (N,) integer array of values >= 1.
    """
    rates = np.log(np.asarray(zooms, dtype=float))          # log zooms add up over time
    half = window // 2
    steps = np.ones(len(rates), dtype=int)
    for k in range(len(rates)):
        around = np.r_[rates[max(0, k - half):k], rates[k + 1:k + half + 1]]
        typical = np.median(around) if len(around) else 0.0
        if typical >= min_rate:                              # enough motion to judge
            steps[k] = max(1, int(round(rates[k] / typical)))
    return steps


def frame_times(steps, frame_interval):
    """Time of every frame, starting at 0 (Task 4.3).

    t_0 = 0 and t_{k+1} = t_k + steps_k * frame_interval.  Returns (N + 1,).
    """
    return np.concatenate([[0.0], np.cumsum(np.asarray(steps, dtype=float) * frame_interval)])


def metric_altitudes(relative, anchor_index, anchor_altitude):
    """Scale relative altitudes so frame `anchor_index` is at `anchor_altitude` metres."""
    relative = np.asarray(relative, dtype=float)
    return relative * (anchor_altitude / relative[anchor_index])


def ground_displacements(shifts, rotations, altitudes, focal_length):
    """Horizontal displacement of the lander for every step (Task 4.3).

    shifts       : (N, 2) image shifts t_k in pixels
    rotations    : (N,) image rotations theta_k in radians
    altitudes    : (N + 1,) altitudes in metres
    focal_length : focal length in pixels, at the same scale as the shifts

    Returns (N, 2) displacements [dX, dY] in metres, in the world frame (see
    the module conventions: image y points down, world Y points up). The
    relation follows from the camera model at the start of Task 4 (see the
    comment below); the tests check it on a simulated descent.
    """
    # With D = diag(1, -1) (image y down, world Y up), a ground point G appears
    # at x_k = (f / Z_k) D R(-psi_k)(G - C_k), so between frames
    # t_k = -(f / Z_{k+1}) D R(-psi_{k+1})(C_{k+1} - C_k), i.e.
    # C_{k+1} - C_k = -(Z_{k+1} / f) R(psi_{k+1}) D t_k
    headings = np.cumsum(np.asarray(rotations, dtype=float))      # psi_1 ... psi_N
    flipped = np.asarray(shifts, dtype=float) * np.array([1.0, -1.0])   # D t_k
    out = np.zeros((len(headings), 2))
    for k, (t, psi, z) in enumerate(zip(flipped, headings, altitudes[1:])):
        c, s = np.cos(psi), np.sin(psi)
        out[k] = -(z / focal_length) * np.array([[c, -s], [s, c]]) @ t
    return out


def trajectory(displacements, altitudes):
    """3D positions of every frame: (N + 1, 3) of [X, Y, Z].

    X and Y start at 0 and add up the displacements; Z is the altitude.
    """
    xy = np.vstack([[0.0, 0.0], np.cumsum(np.asarray(displacements, dtype=float), axis=0)])
    return np.c_[xy, np.asarray(altitudes, dtype=float)]


def velocities(positions, times):
    """Velocity [v_X, v_Y, v_Z] of every step, (P_{k+1} - P_k) / (t_{k+1} - t_k).

    Returns (N, 3).
    """
    positions = np.asarray(positions, dtype=float)
    return np.diff(positions, axis=0) / np.diff(np.asarray(times, dtype=float))[:, None]


def time_to_touchdown(altitudes, times):
    """Time to touchdown at the end of every step (Task 4.3).

    Equation (4.3): tau_k = Z_{k+1} / (-v_Z), with the vertical velocity
    v_Z = (Z_{k+1} - Z_k) / (t_{k+1} - t_k); np.inf where v_Z >= 0.
    Returns (N,).
    """
    z = np.asarray(altitudes, dtype=float)
    descent = (z[:-1] - z[1:]) / np.diff(np.asarray(times, dtype=float))
    tau = np.full(len(descent), np.inf)
    down = descent > 0
    tau[down] = z[1:][down] / descent[down]
    return tau
