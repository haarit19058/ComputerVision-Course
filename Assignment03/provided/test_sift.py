"""Tests for the provided SIFT code (they pass before you write anything)."""

import numpy as np

from provided.sampling import bilinear_sample
from provided.sift import (detect_and_describe, gaussian_blur, guess_displacements,
                         match_descriptors, sift_matches, track_with_sift, window_error)


def texture(n=160, seed=0):
    image = gaussian_blur(np.random.default_rng(seed).random((n, n)), 1.2)
    return (image - image.min()) / (image.max() - image.min())


def similarity(image, angle_deg, scale):
    """Rotate and scale an image about its centre; returns the image and the 2x2 map."""
    n = image.shape[0]
    t = np.deg2rad(angle_deg)
    M = scale * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    y, x = np.mgrid[0:n, 0:n].astype(float)
    c = (n - 1) / 2
    src = np.einsum("ij,jhw->ihw", np.linalg.inv(M), np.stack([x - c, y - c])) + c
    return bilinear_sample(image, src[0], src[1])[0], M


def test_descriptors_have_128_unit_length_entries():
    points, descriptors = detect_and_describe(texture())
    assert len(points) > 30
    assert descriptors.shape == (len(points), 128)
    assert np.allclose(np.linalg.norm(descriptors, axis=1), 1.0)
    assert (descriptors >= 0).all()


def test_matches_follow_a_large_translation():
    image0 = texture()
    y, x = np.mgrid[0:160, 0:160].astype(float)
    image1, _ = bilinear_sample(image0, x - 9.4, y + 6.2)          # content moves (9.4, -6.2)
    p0, p1 = sift_matches(image0, image1)
    assert len(p0) >= 20
    error = np.linalg.norm((p1 - p0) - [9.4, -6.2], axis=1)
    assert np.mean(error < 1.0) >= 0.9


def test_matches_survive_rotation_and_scale():
    image0 = texture(seed=1)
    image1, M = similarity(image0, 25, 1.15)
    p0, p1 = sift_matches(image0, image1)
    assert len(p0) >= 20
    c = (image0.shape[0] - 1) / 2
    error = np.linalg.norm(p1 - ((p0 - c) @ M.T + c), axis=1)
    assert np.mean(error < 1.5) >= 0.8


def test_ratio_test_rejects_ambiguous_matches():
    rng = np.random.default_rng(2)
    d = rng.random((5, 128))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    assert match_descriptors(d, d).tolist() == [[i, i] for i in range(5)]
    twins = np.vstack([d, d[:1]])                               # descriptor 0 appears twice
    assert 0 not in match_descriptors(d, twins)[:, 0]


def test_guess_displacements_uses_nearby_matches():
    rng = np.random.default_rng(3)
    left = rng.uniform([0, 0], [50, 100], (30, 2))
    right = rng.uniform([150, 0], [200, 100], (30, 2))
    m0 = np.vstack([left, right])
    m1 = np.vstack([left + [5, 0], right + [0, -8]])
    m1[0] += [40, 40]                                            # one wrong match
    guess = guess_displacements(np.array([[20.0, 50.0], [180.0, 50.0]]), m0, m1)
    assert np.allclose(guess, [[5, 0], [0, -8]])
    assert np.allclose(guess_displacements(np.zeros((3, 2)), np.zeros((0, 2)), np.zeros((0, 2))), 0)


def test_window_error_is_zero_for_the_right_shift():
    image0 = texture()
    y, x = np.mgrid[0:160, 0:160].astype(float)
    image1, _ = bilinear_sample(image0, x - 6, y)
    p = np.array([[60.0, 60.0], [90.0, 80.0]])
    assert np.allclose(window_error(image0, image1, p, p + [6, 0]), 0)
    assert (window_error(image0, image1, p, p) > 1e-4).all()


def test_track_with_sift_keeps_the_better_start():
    image0 = texture()
    y, x = np.mgrid[0:160, 0:160].astype(float)
    image1, _ = bilinear_sample(image0, x - 6, y)                 # true motion (6, 0)

    def stand_in_tracker(im0, im1, points, radius, initial=None, **options):
        """Pretends to track: returns each start position unchanged."""
        start = np.zeros_like(points) if initial is None else initial
        return points + start, np.ones(len(points), bool)

    points = np.array([[60.0, 60.0], [100.0, 90.0]])
    grid = np.stack(np.meshgrid(np.arange(20, 140, 10.0), np.arange(20, 140, 10.0)), -1).reshape(-1, 2)
    good = (grid, grid + [6, 0])                                  # matches that tell the truth
    tracked, _ = track_with_sift(stand_in_tracker, image0, image1, points, good)
    assert np.allclose(tracked, points + [6, 0])
    slow, _ = bilinear_sample(image0, x - 1, y)                   # true motion only (1, 0)
    bad = (grid, grid + [30, 0])                                  # misleading matches
    tracked, _ = track_with_sift(stand_in_tracker, image0, slow, points, bad)
    assert np.allclose(tracked, points)                           # (0, 0) was the better start
