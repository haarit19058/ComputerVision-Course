"""Small helpers for the provided steps.  PROVIDED."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_reported = set()


def attempt(function, *args, **kwargs):
    """Call one of your functions.

    If it (or anything it calls) is still a TODO, print which function is
    missing (once per run) and return None, so the run script can skip that
    step instead of crashing.
    """
    try:
        return function(*args, **kwargs)
    except NotImplementedError as exc:
        tb = exc.__traceback__
        while tb.tb_next is not None:
            tb = tb.tb_next
        code = tb.tb_frame.f_code
        from provided.notebook import FUNCTIONS
        part = FUNCTIONS.get(code.co_name)
        where = f" (Task {part}, task{part[0]}.ipynb)" if part else ""
        message = f"  [TODO] {code.co_name}(){where} is not implemented yet"
        if code.co_name != getattr(function, "__name__", code.co_name):
            message += f", needed by {function.__name__}()"
        if message not in _reported:
            _reported.add(message)
            print(message)
        return None


def output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, data):
    """Save numbers/arrays as readable JSON (arrays become lists)."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        raise TypeError(type(value))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=convert), encoding="utf-8")
    try:
        shown = path.resolve().relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"  saved {shown}")
    return path


def task_output(task):
    """outputs/task<N>/ under the repository root, created if needed."""
    return output_dir(ROOT / "outputs" / f"task{task}")


FRAMES = ROOT / "data" / "frames"
if not FRAMES.is_dir():                  # a copy of this folder kept next to the original
    FRAMES = ROOT.parent / "lunar-velocity-mission" / "data" / "frames"
CONSTANTS = ROOT / "data" / "mission_constants.json"


def parse_pair(text):
    """'10:11' -> (10, 11)"""
    a, b = text.split(":")
    return int(a), int(b)
