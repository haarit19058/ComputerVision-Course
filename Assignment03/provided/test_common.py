"""Tests for the provided helpers and for the assignment rules.

These should pass before you write any code. The rule tests keep passing only
if you stay within the allowed libraries.
"""

import ast
from pathlib import Path

import numpy as np

from provided.io import frame_index, list_frames, load_frame, save_image
from provided.sampling import bilinear_sample

ROOT = Path(__file__).resolve().parents[1]
PROHIBITED = {"cv2", "skimage", "scipy", "torch", "tensorflow", "jax", "kornia", "sklearn"}


def test_frame_index_reads_numbers_from_names():
    assert frame_index("12.png") == 12
    assert frame_index("frame_0012.png") == 12


def test_save_and_load_roundtrip(tmp_path):
    image = np.linspace(0, 1, 20 * 30, dtype=np.float32).reshape(20, 30)
    save_image(tmp_path / "7.png", image)
    assert list(list_frames(tmp_path)) == [7]
    loaded = load_frame(tmp_path, 7)
    assert loaded.shape == (20, 30)
    assert np.abs(loaded - image).max() < 1 / 255 + 1e-6


def test_bilinear_sample_is_exact_on_pixels_and_linear_between():
    image = np.arange(12, dtype=float).reshape(3, 4)          # value = 4*y + x
    values, valid = bilinear_sample(image, np.array([1.0, 1.5, 2.25]), np.array([1.0, 0.5, 1.5]))
    assert valid.all()
    assert np.allclose(values, [5.0, 3.5, 8.25])


def test_bilinear_sample_marks_outside_points_invalid():
    image = np.ones((3, 4))
    _, valid = bilinear_sample(image, np.array([-0.5, 3.5, 1.0]), np.array([1.0, 1.0, 2.5]))
    assert not valid.any()


def _python_files():
    return [p for p in ROOT.rglob("*.py") if ".venv" not in p.parts and "venv" not in p.parts]


def _code_to_check():
    """(where, source, is_your_code) for every provided .py file and every code
    cell of the four notebooks (Jupyter-only lines blanked out)."""
    from provided.notebook import (NOTEBOOKS, _without_magics, cell_source, read_notebook,
                                 your_code_label)
    for path in _python_files():
        yield str(path.relative_to(ROOT)), path.read_text(encoding="utf-8"), False
    for name in NOTEBOOKS.values():
        path = ROOT / name
        if not path.exists():
            continue
        for k, cell in enumerate(read_notebook(path).get("cells", [])):
            if cell.get("cell_type") == "code":
                yield (f"{name}, cell {k + 1}", _without_magics(cell_source(cell)),
                       bool(your_code_label(cell)))


def _parsed(source, where):
    try:
        return ast.parse(source, filename=where)
    except SyntaxError:
        return None                      # a cell with a syntax error: reported by its checks


def test_no_prohibited_libraries_are_imported():
    offenders = []
    for where, source, _ in _code_to_check():
        tree = _parsed(source, where)
        for node in ast.walk(tree) if tree else []:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [f"{where}: {n}" for n in names if n.split(".")[0] in PROHIBITED]
    assert not offenders, "prohibited imports:\n" + "\n".join(offenders)


def test_np_gradient_is_not_used():
    offenders = []
    for where, source, your_code in _code_to_check():
        if not your_code and not where.endswith(".py"):
            continue                     # only your YOUR CODE cells, and the provided code
        if Path(where).name.startswith("test_"):
            continue
        tree = _parsed(source, where)
        for node in ast.walk(tree) if tree else []:
            if isinstance(node, ast.Attribute) and node.attr == "gradient" \
                    and isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy"):
                offenders.append(f"{where}, line {node.lineno}")
    assert not offenders, "np.gradient is not allowed (Task 1): " + ", ".join(offenders)
