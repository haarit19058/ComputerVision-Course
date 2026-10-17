"""Task 2 tests (synthetic textures, no dataset needed).

    python -m pytest provided/task2

They test the functions in your saved notebooks (the `fn` fixture); in the
notebook, check("2.1") and so on run the tests of one subtask.
"""

import numpy as np

from provided.testing import part


# 2.1 Lucas-Kanade as least squares -------------------------------------------

@part("2.1")
def test_lucas_kanade_update_solves_exact_motion(fn):
    rng = np.random.default_rng(1)
    ix, iy = rng.normal(size=(2, 5, 5))
    it = -(0.3 * ix - 0.2 * iy)                    # brightness constancy with (u, v) = (0.3, -0.2)
    u, v = fn.lucas_kanade_update(ix, iy, it)
    assert np.allclose([u, v], [0.3, -0.2])


@part("2.1")
def test_lucas_kanade_update_rejects_a_flat_window(fn):
    zeros = np.zeros((5, 5))
    assert fn.lucas_kanade_update(zeros, zeros, zeros) is None


# 2.2 Estimate one-step motion ------------------------------------------------

@part("2.2")
def test_one_step_flow_recovers_a_small_shift(fn, make_texture, shift):
    image0 = make_texture()
    image1 = shift(image0, 0.4, -0.3)
    result = fn.one_step_flow(image0, image1, np.array([32.0, 32.0]), window_radius=7)
    assert result is not None
    assert np.allclose(result, [0.4, -0.3], atol=0.05)


@part("2.2")
def test_one_step_flow_fails_when_the_window_leaves_the_image(fn, make_texture, shift):
    image0 = make_texture()
    image1 = shift(image0, 0.5, 0.5)
    assert fn.one_step_flow(image0, image1, np.array([2.0, 2.0]), window_radius=7) is None


@part("2.2")
def test_one_step_flow_can_reuse_precomputed_gradients(fn, make_texture, shift):
    image0 = make_texture()
    image1 = shift(image0, 0.4, -0.3)
    grad0 = fn.spatial_gradients(image0)
    result = fn.one_step_flow(image0, image1, np.array([32.0, 32.0]), grad0=grad0)
    assert np.allclose(result, [0.4, -0.3], atol=0.05)


# 2.3 When one step is not enough ---------------------------------------------

@part("2.3")
def test_one_step_flow_gets_worse_as_the_shift_grows(fn, make_texture, shift):
    image0 = make_texture()
    point = np.array([32.0, 32.0])
    small_error = np.linalg.norm(
        np.array(fn.one_step_flow(image0, shift(image0, 0.4, -0.3), point)) - [0.4, -0.3])
    large_error = np.linalg.norm(
        np.array(fn.one_step_flow(image0, shift(image0, 6.0, -4.0), point)) - [6.0, -4.0])
    assert large_error > small_error
