"""Shared pytest fixtures.  PROVIDED.

`fn` holds your functions, loaded from the *saved* notebooks (task1.ipynb up
to the task a test file belongs to), exactly as the grader loads them.
Running `python -m pytest` from the repository root also puts the root on the
import path, so tests can import `common`, `task1`, `task2`, ...
"""

import re

import pytest

from provided.notebook import load
from provided.testing import smooth_texture
from provided.sampling import translate

_loaded = {}


@pytest.fixture
def fn(request):
    module = request.module.__name__                 # e.g. "provided.task2.test_task2"
    task = int(next(p for p in module.split(".") if re.fullmatch(r"task\d", p))[4:])
    if task not in _loaded:
        _loaded[task] = load(task)
    return _loaded[task]


@pytest.fixture
def make_texture():
    return smooth_texture


@pytest.fixture
def shift():
    return translate
