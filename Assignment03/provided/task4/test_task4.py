"""Task 4 tests.

    python -m pytest provided/task4

task4/motion_model.py and task4/trajectory.py are provided, so their tests
(the first two sections below) pass from the start; they check the provided
code and show its conventions at work. The last section tests your Task 4.1
functions in task4.ipynb (the `fn` fixture) on a synthetic textured pair with a
known shift; in the notebook, check("4.1") runs them.
"""

import numpy as np

from provided.sift import gaussian_blur
from provided.task4.motion_model import (apply_similarity, fit_similarity, robust_fit_similarity,
                                similarity_system)
from provided.testing import part
from provided.task4.trajectory import (frame_times, ground_displacements, metric_altitudes,
                              relative_altitudes, time_to_touchdown, trajectory, velocities)

CENTRE = np.array([293.5, 215.5])


# ---- task4/motion_model.py (provided) --------------------------------------

def true_motion(points, zoom, rotation, shift, centre=CENTRE):
    """Reference model used only by these tests."""
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    return (zoom * (points - centre) @ R.T) + shift + centre


def tracks(n=80, seed=0, zoom=1.04, rotation=np.deg2rad(1.5), shift=(4.0, -3.0), noise=0.0):
    rng = np.random.default_rng(seed)
    p0 = rng.uniform([20, 20], [568, 412], (n, 2))
    p1 = true_motion(p0, zoom, rotation, np.array(shift)) + rng.normal(0, noise, (n, 2))
    return p0, p1


def test_similarity_system_holds_for_the_true_parameters():
    p0, p1 = tracks(10)
    A, b = similarity_system(p0, p1, CENTRE)
    assert A.shape == (20, 4) and b.shape == (20,)
    p, q = 1.04 * np.cos(np.deg2rad(1.5)), 1.04 * np.sin(np.deg2rad(1.5))
    assert np.allclose(A @ np.array([p, q, 4.0, -3.0]), b)


def test_fit_recovers_zoom_rotation_and_shift():
    model = fit_similarity(*tracks(), CENTRE)
    assert np.isclose(model["zoom"], 1.04)
    assert np.isclose(model["rotation"], np.deg2rad(1.5))
    assert np.allclose(model["shift"], [4.0, -3.0])


def test_apply_similarity_reproduces_the_tracks():
    p0, p1 = tracks()
    model = {"zoom": 1.04, "rotation": np.deg2rad(1.5), "shift": np.array([4.0, -3.0])}
    assert np.allclose(apply_similarity(model, p0, CENTRE), p1)


def test_robust_fit_ignores_wrong_tracks():
    p0, p1 = tracks(n=100, seed=3, noise=0.3)
    wrong = np.random.default_rng(4).choice(100, 20, replace=False)
    p1[wrong] += np.random.default_rng(5).uniform(-40, 40, (20, 2))
    model, inliers, residuals = robust_fit_similarity(p0, p1, CENTRE)
    assert np.isclose(model["zoom"], 1.04, atol=2e-3)
    assert inliers.shape == residuals.shape == (100,)
    assert not inliers[wrong].any()
    assert inliers.sum() >= 75


# ---- task4/trajectory.py (provided) ----------------------------------------

FOCAL = 400.0                                   # pixels
FLIP = np.array([1.0, -1.0])                    # image y points down, world Y points up


def rotation(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def project(ground, position, altitude, heading):
    return FOCAL * FLIP * ((ground - position) @ rotation(-heading).T) / altitude


def simulated_descent(n=30):
    k = np.arange(n + 1)
    altitude = 500.0 * 0.95 ** k
    position = np.c_[3.0 * k + 0.05 * k ** 2, -2.0 * k]
    heading = np.deg2rad(0.4) * np.sin(k / 4)
    ground = np.array([[37.0, -12.0], [-60.0, 45.0], [120.0, 80.0]])
    zooms = altitude[:-1] / altitude[1:]
    rotations = heading[1:] - heading[:-1]
    shifts = []
    for i in range(n):
        x0 = project(ground, position[i], altitude[i], heading[i])
        x1 = project(ground, position[i + 1], altitude[i + 1], heading[i + 1])
        moved = x0 @ (zooms[i] * rotation(rotations[i])).T
        shifts.append(x1[0] - moved[0])
    return zooms, rotations, np.array(shifts), altitude, position


def test_relative_altitudes_chain_the_zooms():
    assert np.allclose(relative_altitudes(np.array([2.0, 1.25, 1.0])), [1.0, 0.5, 0.4, 0.4])


def test_the_simulated_descent_is_recovered():
    zooms, rotations, shifts, altitude, position = simulated_descent()
    z = metric_altitudes(relative_altitudes(zooms), 10, altitude[10])
    assert np.allclose(z, altitude)
    d = ground_displacements(shifts, rotations, z, FOCAL)
    p = trajectory(d, z)
    assert np.allclose(p[:, :2], position - position[0], atol=1e-6)
    assert np.allclose(p[:, 2], altitude)


def test_axes_follow_the_handout():
    shifts = np.array([[-4.0, 0.0], [0.0, 4.0]])
    d = ground_displacements(shifts, np.zeros(2), np.array([100.0, 100.0, 100.0]), FOCAL)
    assert np.allclose(d, [[1.0, 0.0], [0.0, 1.0]])


def test_velocities_and_time_to_touchdown():
    p = np.array([[0.0, 0.0, 10.0], [1.0, 2.0, 8.0], [3.0, 2.0, 7.0]])
    v = velocities(p, np.array([0.0, 0.5, 1.5]))
    assert np.allclose(v, [[2.0, 4.0, -4.0], [2.0, 0.0, -1.0]])
    tau = time_to_touchdown(np.array([10.0, 8.0, 8.0, 9.0]), np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.isclose(tau[0], 4.0)
    assert np.isinf(tau[1]) and np.isinf(tau[2])


def test_frame_times_follow_the_steps():
    assert np.allclose(frame_times(np.array([1, 1, 1]), 0.5), [0.0, 0.5, 1.0, 1.5])


# ---- your Task 4.1 functions (task4.ipynb) ----------------------------------------

def big_texture(n=160, seed=0):
    """A larger, SIFT-friendly texture, with enough keypoints for reliable
    matching under a large shift."""
    image = gaussian_blur(np.random.default_rng(seed).random((n, n)), 1.2)
    return (image - image.min()) / (image.max() - image.min())


@part("4.1")
def test_one_step_tracks_a_small_shift(fn, make_texture, shift):
    image0 = make_texture()
    image1 = shift(image0, 0.4, -0.3)
    points = np.array([[20.0, 20.0], [32.0, 40.0], [40.0, 25.0]])
    tracked, valid = fn.one_step(image0, image1, points)
    assert valid.all()
    assert np.allclose(tracked, points + [0.4, -0.3], atol=0.1)


@part("4.1")
def test_iterative_tracks_a_larger_shift(fn, make_texture, shift):
    image0 = make_texture()
    image1 = shift(image0, 1.8, 1.1)
    points = np.array([[20.0, 20.0], [32.0, 40.0], [40.0, 25.0]])
    tracked, valid = fn.iterative(image0, image1, points)
    assert valid.all()
    assert np.allclose(tracked, points + [1.8, 1.1], atol=0.1)


@part("4.1")
def test_sift_guided_beats_one_step_on_large_motion(fn):
    from provided.sampling import translate
    image0 = big_texture()
    image1 = translate(image0, 20.0, 0.0)          # far beyond a plain window's reach
    points = np.array([[60.0, 60.0], [90.0, 80.0], [110.0, 40.0]])
    truth = points + [20.0, 0.0]
    step_tracked, _ = fn.one_step(image0, image1, points)
    guided_tracked, guided_valid = fn.sift_guided(image0, image1, points)
    assert guided_valid.all()
    assert np.allclose(guided_tracked, truth, atol=0.5)
    step_error = np.linalg.norm(step_tracked - truth, axis=1)
    guided_error = np.linalg.norm(guided_tracked - truth, axis=1)
    assert (guided_error < step_error).all()


@part("4.1")
def test_track_pair_selects_and_tracks_points(fn, make_texture, shift):
    image0 = make_texture()
    image1 = shift(image0, 2.0, -1.0)
    p0, p1 = fn.track_pair(image0, image1, fn.iterative)
    assert len(p0) >= 10
    assert p0.shape == p1.shape
    assert np.allclose(p1 - p0, [2.0, -1.0], atol=0.3)
