"""Task 1 tests (synthetic images, no dataset needed).

    python -m pytest provided/task1

They test the functions in your saved task1.ipynb (the `fn` fixture); in the
notebook, check("1.1") and so on run the tests of one subtask.
"""

import numpy as np

from provided.sampling import translate
from provided.testing import part


# 1.1 Spatial gradients --------------------------------------------------------

def ramp(h=12, w=16):
    """I(x, y) = x + 2y, so Ix = 1 and Iy = 2 everywhere inside."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return x + 2 * y


@part("1.1")
def test_spatial_gradients_of_a_ramp(fn):
    ix, iy = fn.spatial_gradients(ramp())
    assert ix.shape == iy.shape == (12, 16)
    assert np.allclose(ix[1:-1, 1:-1], 1.0)
    assert np.allclose(iy[1:-1, 1:-1], 2.0)


@part("1.1")
def test_spatial_gradients_border_is_zero(fn):
    ix, iy = fn.spatial_gradients(ramp())
    for g in (ix, iy):
        assert np.allclose(g[0, :], 0) and np.allclose(g[-1, :], 0)
        assert np.allclose(g[:, 0], 0) and np.allclose(g[:, -1], 0)


@part("1.1")
def test_x_is_columns_and_y_is_rows(fn):
    image = np.zeros((5, 5))
    image[:, 3:] = 1.0                     # vertical edge: changes along x only
    ix, iy = fn.spatial_gradients(image)
    assert np.abs(ix).max() > 0
    assert np.allclose(iy, 0)


# 1.2 Temporal change and motion ---------------------------------------------

@part("1.2")
def test_temporal_difference_of_identical_images_is_zero(fn):
    image = ramp()
    assert np.allclose(fn.temporal_difference(image, image), 0)


@part("1.2")
def test_temporal_difference_is_image1_minus_image0(fn):
    a, b = np.zeros((4, 4)), np.zeros((4, 4))
    a[1, 1], b[1, 1] = 2.0, 5.0
    assert np.isclose(fn.temporal_difference(a, b)[1, 1], 3.0)
    assert np.isclose(fn.temporal_difference(b, a)[1, 1], -3.0)


@part("1.2")
def test_motion_constraint_holds_for_a_known_shift(fn, make_texture):
    # Brightness constancy: I(x + u, y + v, t + 1) = I(x, y, t). Linearising
    # in (u, v) gives Ix u + Iy v + It ~= 0. Check it away from the border,
    # where spatial_gradients is exact and the shift is small enough that
    # the linear approximation is good.
    image0 = make_texture()
    u, v = 0.4, -0.3
    image1 = translate(image0, u, v)
    ix, iy = fn.spatial_gradients(image0)
    it = fn.temporal_difference(image0, image1)
    residual = ix * u + iy * v + it
    assert np.abs(residual[8:-8, 8:-8]).max() < 0.05


# 1.3 Choosing good points to track ------------------------------------------

@part("1.3")
def test_structure_tensor_of_constant_gradients(fn):
    ix = np.ones((9, 9))
    iy = 2 * np.ones((9, 9))
    G = fn.structure_tensor(ix, iy, 4, 4, 1)                     # 3x3 window: 9 pixels
    assert np.allclose(G, 9 * np.array([[1, 2], [2, 4]]))


@part("1.3")
def test_structure_tensor_window_is_centred_on_x_y(fn):
    ix = np.zeros((9, 9))
    ix[4, 6] = 1.0                                            # pixel x=6, y=4
    iy = np.zeros((9, 9))
    assert np.isclose(fn.structure_tensor(ix, iy, 5, 4, 1)[0, 0], 1.0)   # x in 4..6
    assert np.isclose(fn.structure_tensor(ix, iy, 3, 4, 1)[0, 0], 0.0)   # x in 2..4


@part("1.3")
def test_feature_score_is_the_smaller_eigenvalue(fn):
    assert np.isclose(fn.feature_score(np.diag([5.0, 2.0])), 2.0)
    assert np.isclose(fn.feature_score(np.array([[4.0, 0.0], [0.0, 0.0]])), 0.0)   # an edge
    assert np.isclose(fn.feature_score(np.array([[2.0, 1.0], [1.0, 2.0]])), 1.0)


@part("1.3")
def test_select_features_finds_the_corners_of_a_square(fn):
    image = np.zeros((60, 60))
    image[20:40, 20:40] = 1.0
    points = fn.select_features(image, max_points=4, window_radius=2, min_distance=8)
    assert points.shape == (4, 2)
    corners = np.array([[20, 20], [39, 20], [20, 39], [39, 39]], dtype=float)
    for p in points:
        assert np.min(np.linalg.norm(corners - p, axis=1)) <= 3.0


@part("1.3")
def test_select_features_keeps_points_apart(fn, make_texture):
    points = fn.select_features(make_texture(), max_points=30, window_radius=2, min_distance=6)
    assert 0 < len(points) <= 30
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    assert d[np.triu_indices(len(points), 1)].min() >= 6


@part("1.3")
def test_select_features_threshold_can_reject_everything(fn, make_texture):
    points = fn.select_features(make_texture(), max_points=30, threshold=1e9)
    assert points.shape == (0, 2)
