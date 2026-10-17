"""Your notebooks as code.  PROVIDED: you do not need to change this file.

What is graded is always the *saved* notebook file, read fresh from disk:

* a code cell whose first line starts with  # YOUR CODE  holds your functions.
  You may add more such cells (for helper functions): all of them are used, in
  order, from provided.task1.ipynb up to the task being loaded;
* every other cell (the provided cells, and any scratch or note cells you add)
  is never treated as your code.

From a YOUR CODE cell only definitions are kept: imports, functions, classes
and assignments. Anything else in it (a plot, a print, a test call) is
skipped, so an experiment left in the cell cannot break loading. Do your
experiments in cells of your own.
"""

import ast
import hashlib
import importlib
import inspect
import json
import linecache
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "# YOUR CODE"
NOTEBOOKS = {1: "task1.ipynb", 2: "task2.ipynb", 3: "task3.ipynb", 4: "task4.ipynb"}

# what every YOUR CODE cell can use without importing it; the notebooks' setup
# cells import the same names
PREAMBLE = ("import numpy as np\n"
            "from provided.sampling import bilinear_sample\n"
            "from provided.sift import sift_matches, track_with_sift\n")

# the functions you write, and the subtask that asks for each
FUNCTIONS = {
    "spatial_gradients": "1.1", "temporal_difference": "1.2",
    "structure_tensor": "1.3", "feature_score": "1.3", "select_features": "1.3",
    "lucas_kanade_update": "2.1", "one_step_flow": "2.2",
    "track_point": "3.1", "track_points": "3.2",
    "one_step": "4.1", "iterative": "4.1", "sift_guided": "4.1", "track_pair": "4.1",
}

_DEFINITIONS = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                ast.Assign, ast.AnnAssign, ast.AugAssign)


# ---- cells ------------------------------------------------------------------

def read_notebook(path):
    """A notebook file as a dict (the .ipynb format is JSON)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def provided_info(cell):
    """The hidden tag of a provided cell, e.g. {"role": "question", "part": "1.2"}, else None."""
    info = cell.get("metadata", {}).get("assignment")
    return info if isinstance(info, dict) and info.get("provided") else None


def your_code_label(cell):
    """"Task 1.1" for a YOUR CODE cell (from the text after the marker), else None."""
    if cell.get("cell_type") != "code":
        return None
    for line in cell_source(cell).splitlines():
        if line.strip():
            text = line.strip()
            if text.upper().startswith(MARKER):
                return text[len(MARKER):].strip(" :") or "your code"
            return None
    return None


def fingerprint(notebook):
    """A hash of the code that decides your results: the provided code cells and
    your YOUR CODE cells. Notes and scratch cells you add do not change it."""
    digest = hashlib.sha256()
    for cell in notebook.get("cells", []):
        info = provided_info(cell)
        if cell.get("cell_type") == "code" and (your_code_label(cell) or info):
            digest.update(cell_source(cell).encode("utf-8") + b"\0")
    return digest.hexdigest()


# ---- loading your functions -------------------------------------------------

class YourFunctions(types.SimpleNamespace):
    """Your functions by name; a missing one says where it was looked for."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        part = FUNCTIONS.get(name)
        where = f" (Task {part}, task{part[0]}.ipynb)" if part else ""
        problems = self.__dict__.get("_problems", [])
        hint = ("" if not problems else
                "\n  problems while loading:\n    " + "\n    ".join(problems))
        raise AttributeError(
            f"{name}(){where} was not found in any '{MARKER}' cell of the saved notebooks. "
            f"Check that the cell's first line starts with '{MARKER}' and that the notebook "
            f"is saved.{hint}")


def _without_magics(source):
    """Jupyter-only lines (%magic, !shell) are not Python: blank them out."""
    return "\n".join("" if line.lstrip().startswith(("%", "!")) else line
                     for line in source.splitlines())


def _run_definitions(source, name, namespace, problems, skipped):
    code = _without_magics(source)
    linecache.cache[name] = (len(code), None, code.splitlines(True), name)   # for tracebacks
    try:
        tree = ast.parse(code, filename=name)
    except SyntaxError as exc:
        problems.append(f"{name}: syntax error on line {exc.lineno} ({exc.msg}), "
                        "so the functions of this cell are missing")
        return
    for statement in tree.body:
        if not isinstance(statement, _DEFINITIONS):
            skipped.append(f"{name}, line {statement.lineno}")
            continue
        try:
            exec(compile(ast.Module(body=[statement], type_ignores=[]), name, "exec"), namespace)
        except Exception as exc:            # e.g. an assignment that uses a scratch variable
            problems.append(f"{name}, line {statement.lineno}: {type(exc).__name__}: {exc} "
                            "(skipped)")


def load(upto, root=None):
    """Your functions from provided.task1.ipynb up to task<upto>.ipynb, read from the saved files.

    Returns a YourFunctions namespace (fn.spatial_gradients, ...), with the
    loading problems, if any, in fn._problems.
    """
    root = Path(root) if root else ROOT
    namespace = {"__name__": "your_notebook_code"}
    exec(PREAMBLE, namespace)
    problems, skipped = [], []
    for task in range(1, upto + 1):
        path = root / NOTEBOOKS[task]
        try:
            notebook = read_notebook(path)
        except FileNotFoundError:
            problems.append(f"{path.name} not found")
            continue
        except (ValueError, UnicodeDecodeError) as exc:
            problems.append(f"{path.name} could not be read as a notebook ({exc})")
            continue
        for cell in notebook.get("cells", []):
            label = your_code_label(cell)
            if label:
                _run_definitions(cell_source(cell), f"{path.name} [{label}]", namespace,
                                 problems, skipped)
    loaded = YourFunctions(**{k: v for k, v in namespace.items() if not k.startswith("__")})
    loaded._problems, loaded._skipped = problems, skipped
    return loaded


def use_my_functions(upto, namespace=None):
    """Bring your functions of task1 ... task<upto> into this notebook, from the saved files.

    Later tasks build on your earlier ones (Task 2 uses your Task 1 gradients),
    so a notebook's setup cell calls this. Save the earlier notebooks first.
    """
    if namespace is None:
        namespace = inspect.currentframe().f_back.f_globals
    loaded = load(upto)
    wanted = [n for n, p in FUNCTIONS.items() if int(p[0]) <= upto]
    found = [n for n in wanted if n in loaded.__dict__]
    for name in found:
        namespace[name] = loaded.__dict__[name]
    sources = ", ".join(NOTEBOOKS[t] for t in range(1, upto + 1))
    print(f"your functions from {sources} (the saved files): {', '.join(found) or 'none'}")
    missing = [n for n in wanted if n not in found]
    if missing:
        print(f"  not found yet: {', '.join(missing)}")
    for problem in loaded._problems:
        print(f"  problem: {problem}")


# ---- checking your functions in a notebook ---------------------------------

def _innermost(exc):
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    return tb


def _in_provided_code(filename):
    try:
        Path(filename).resolve().relative_to(ROOT)
        return True
    except (ValueError, OSError):
        return False


def _your_line(exc):
    """'line 7: ix = ...' for the deepest frame of the error that is in your code."""
    tb, found = exc.__traceback__, None
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        if not _in_provided_code(filename):
            found = tb
        tb = tb.tb_next
    if found is None:
        return ""
    line = linecache.getline(found.tb_frame.f_code.co_filename, found.tb_lineno).strip()
    return f" (in {found.tb_frame.f_code.co_name}, line {found.tb_lineno}: {line})" if line else ""


def _failed_assert(exc):
    """The assert line of the test that failed, when the assertion has no message."""
    if str(exc):
        return str(exc).splitlines()[0]
    tb = _innermost(exc)
    return linecache.getline(tb.tb_frame.f_code.co_filename, tb.tb_lineno).strip() if tb else ""


def check(part, namespace=None):
    """Run the tests of one subtask, e.g. check("1.1"), on the functions in this notebook.

    It checks the functions as they are now in the notebook. The grader runs
    the same tests on your saved notebook, so save it (Ctrl+S) before you
    submit.
    """
    from provided.testing import FIXTURES
    if namespace is None:
        namespace = inspect.currentframe().f_back.f_globals
    fn = YourFunctions(**{k: v for k, v in namespace.items() if not k.startswith("__")})
    task = int(str(part).split(".")[0])
    module = importlib.import_module(f"provided.task{task}.test_task{task}")
    tests = [t for name, t in vars(module).items()
             if name.startswith("test_") and callable(t) and getattr(t, "part", None) == str(part)]
    passed = 0
    print(f"Task {part} checks")
    for test in tests:
        kwargs = {}
        for name in inspect.signature(test).parameters:
            kwargs[name] = fn if name == "fn" else FIXTURES[name]()
        title = test.__name__[len("test_"):].replace("_", " ")
        try:
            test(**kwargs)
            status, why = "PASS", ""
            passed += 1
        except NotImplementedError as exc:
            tb = _innermost(exc)
            name = tb.tb_frame.f_code.co_name if tb else "a function"
            status, why = "TODO", f"{name}() is not implemented yet"
        except AssertionError as exc:
            status, why = "FAIL", _failed_assert(exc)
        except Exception as exc:
            status, why = "ERROR", f"{type(exc).__name__}: {exc}{_your_line(exc)}"
        print(f"  {status:5} {title}" + (f": {why}" if why else ""))
    print(f"{passed} of {len(tests)} passed" +
          ("" if passed == len(tests) else " (the grader runs these on your saved notebook)"))
    return passed == len(tests)
