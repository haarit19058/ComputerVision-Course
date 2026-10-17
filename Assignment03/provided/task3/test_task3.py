"""Task 3 tests (synthetic textures, no dataset needed).

    python -m pytest provided/task3

They test the functions in your saved notebooks (the `fn` fixture); in the
notebook, check("3.1") and so on run the tests of one subtask.
"""

import numpy as np

from provided.sift import gaussian_blur, sift_matches, track_with_sift
from provided.testing import part
from provided.task3.steps import tracking_statistics


# 3.1 Iterative Lucas-Kanade --------------------------------------------------

@part("3.1")
def test_track_point_recovers_a_subpixel_shift(fn, make_texture, shift):
    image0 = make_texture()
    image1 = shift(image0, 1.3, -0.7)
    new_point, ok = fn.track_point(image0, image1, np.array([32.0, 32.0]), window_radius=7)
    assert ok
    assert np.allclose(new_point, [33.3, 31.3], atol=0.08)


@part("3.1")
def test_iterations_do_not_hurt(fn, make_texture, shift):
    image0 = make_texture()
    image1 = shift(image0, 1.8, 1.1)
    start, truth = np.array([32.0, 32.0]), np.array([33.8, 33.1])
    one, _ = fn.track_point(image0, image1, start, window_radius=7, max_iterations=1)
    many, ok = fn.track_point(image0, image1, start, window_radius=7, max_iterations=20)
    assert ok
    assert np.linalg.norm(many - truth) < 0.1
    assert np.linalg.norm(many - truth) <= np.linalg.norm(one - truth) + 1e-6


@part("3.1")
def test_track_point_fails_when_the_window_leaves_the_image(fn, make_texture, shift):
    image0 = make_texture()
    _, ok = fn.track_point(image0, shift(image0, 0.5, 0.5), np.array([2.0, 2.0]), window_radius=7)
    assert not ok


@part("3.1")
def test_track_point_starts_from_the_initial_guess(fn, make_texture, shift):
    image0 = make_texture()
    image1 = shift(image0, 6.0, -4.0)            # large motion: start near it, as SIFT would
    new_point, ok = fn.track_point(image0, image1, np.array([32.0, 32.0]), window_radius=7,
                                max_iterations=20, initial=np.array([5.4, -3.5]))
    assert ok
    assert np.allclose(new_point, [38.0, 28.0], atol=0.1)


# 3.2 Track many points -------------------------------------------------------

@part("3.2")
def test_track_points_accepts_one_initial_guess_per_point(fn, make_texture, shift):
    image0 = make_texture()
    image1 = shift(image0, 6.0, -4.0)
    points = np.array([[26.0, 30.0], [34.0, 36.0]])
    initial = np.array([[5.5, -3.6], [6.4, -4.5]])
    tracked, valid = fn.track_points(image0, image1, points, max_iterations=20, initial=initial)
    assert valid.all()
    assert np.allclose(tracked, points + [6.0, -4.0], atol=0.1)


@part("3.2")
def test_track_points_returns_positions_and_validity(fn, make_texture, shift):
    image0 = make_texture()
    points = np.array([[20.0, 20.0], [32.0, 40.0], [2.0, 2.0]])
    tracked, valid = fn.track_points(image0, shift(image0, 1.0, 0.0), points)
    assert tracked.shape == (3, 2) and valid.shape == (3,)
    assert valid.tolist() == [True, True, False]


# 3.3 Use SIFT to handle large motion ------------------------------------------

def big_texture(n=160, seed=0):
    """A larger texture: SIFT needs more content than the 64x64 make_texture."""
    image = gaussian_blur(np.random.default_rng(seed).random((n, n)), 1.2)
    return (image - image.min()) / (image.max() - image.min())


@part("3.3")
def test_track_with_sift_recovers_motion_too_large_for_zero_start(fn, shift):
    image0 = big_texture()
    image1 = shift(image0, 20.0, 0.0)                  # far beyond a window_radius=7 reach
    points = np.array([[60.0, 60.0], [90.0, 80.0], [110.0, 40.0]])
    matches = sift_matches(image0, image1)
    tracked, valid = track_with_sift(fn.track_points, image0, image1, points, matches)
    assert valid.all()
    assert np.allclose(tracked, points + [20.0, 0.0], atol=0.5)


# 3.4 Compare the three trackers: tracking_statistics -------------------------

def test_tracking_statistics():
    p0 = np.zeros((3, 2))
    p1 = np.array([[3.0, 4.0], [0.0, 1.0], [9.0, 9.0]])
    s = tracking_statistics(p0, p1, np.array([True, True, False]))
    assert np.isclose(s["mean_displacement"], 3.0)
    assert np.isclose(s["median_displacement"], 3.0)
    assert np.isclose(s["max_displacement"], 5.0)
    assert np.isclose(s["failed_track_rate"], 1 / 3)
