"""Package your submission into one zip file with only what is required.
PROVIDED: you do not need to change this file.

Run from this folder, after `python run_notebooks.py`:

    python make_submission.py                 # writes submission.zip
    python make_submission.py my_name.zip      # a different output name

It builds report.html from your saved notebooks and the files in outputs/
(it never relies on the outputs stored inside a notebook), then zips your
four notebooks, report.html and the outputs/ files used for grading. Nothing
from data/ and no caches.

The zip is always written, and you can submit it even if parts are
unfinished. It then warns you about:

  * checks that fail on your saved notebooks (the grader runs the same ones);
  * required outputs/ files that are missing (unfinished tasks lose those marks);
  * notebooks whose code changed after the last run of run_notebooks.py;
  * questions still marked TODO (unanswered questions get no marks).

Only your name and roll number are needed: without them your submission
cannot be matched to you.
"""

import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from provided.notebook import NOTEBOOKS, fingerprint, read_notebook  # noqa: E402
from provided.report import FIGURES, build_report, identity  # noqa: E402

NUMBERS = [
    "outputs/task1/constraint_check.json",
    "outputs/task2/error_sweep.json",
    "outputs/task3/error_sweep.json",
    "outputs/task3/small_motion_statistics.json", "outputs/task3/large_motion_statistics.json",
    "outputs/task3/touchdown_statistics.json",
    "outputs/task4/summary.json",
    "outputs/task4/compare/summary.csv",
]
OUTPUTS = NUMBERS + [f"outputs/{rel}" for figures in FIGURES.values() for rel, _ in figures]
FILES = list(NOTEBOOKS.values()) + ["report.html"] + OUTPUTS
OPTIONAL = ["outputs/run_stamp.json"]


def failing_checks():
    """{task: "3 of 6 fail"} from pytest on the saved notebooks, or None without pytest."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        return None
    result = {}
    for task in NOTEBOOKS:
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=no", "-p",
                              "no:cacheprovider", f"provided/task{task}"],
                             cwd=ROOT, capture_output=True, text=True, timeout=600)
        tail = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
        failed = sum(int(n) for n in re.findall(r"(\d+) (?:failed|error)", tail))
        passed = sum(int(n) for n in re.findall(r"(\d+) passed", tail))
        if failed:
            result[task] = f"{failed} of {failed + passed} fail"
    return result


def stale_notebooks():
    """Notebooks whose code changed after the last run_notebooks.py (or never ran there)."""
    try:
        stamp = json.loads((ROOT / "outputs" / "run_stamp.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        stamp = {}
    stale, errors = [], []
    for task, name in NOTEBOOKS.items():
        entry = stamp.get(f"task{task}")
        try:
            current = fingerprint(read_notebook(ROOT / name))
        except (FileNotFoundError, ValueError, UnicodeDecodeError):
            continue
        if not entry or entry.get("fingerprint") != current:
            stale.append(name)
        elif entry.get("errors"):
            errors.append(f"{name} ({entry['errors']} cell(s) with errors)")
    return stale, errors


def print_list(items, indent="  ", width=78):
    line = indent
    for item in items:
        if len(line) + len(item) + 2 > width:
            print(line.rstrip())
            line = indent
        line += f"{item}, "
    print(line.rstrip(", "))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out_path = ROOT / (argv[0] if argv else "submission.zip")

    _, todo = build_report(ROOT)
    wanted = FILES + [f for f in OPTIONAL if (ROOT / f).is_file()]
    present = [f for f in wanted if (ROOT / f).is_file()]
    missing = [f for f in FILES if f not in present]
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in present:
            zf.write(ROOT / f, arcname=f)
    print(f"wrote {out_path.name}: {len(present)} files, with report.html built from your "
          "saved notebooks")

    warnings = 0
    checks = failing_checks()
    if checks is None:
        print("\n(pytest is not installed, so your saved notebooks were not checked)")
    elif checks:
        warnings += 1
        print("\nwarning: checks that fail on your saved notebooks (the grader runs these):")
        for task, text in checks.items():
            print(f"  Task {task}: {text}  (python -m pytest provided/task{task} shows which)")
    if missing:
        warnings += 1
        print("\nwarning: missing, so not in the zip (run the notebooks to make them):")
        for f in missing:
            print(f"  {f}")
    stale, errors = stale_notebooks()
    if stale:
        warnings += 1
        print("\nwarning: code changed after the last clean run, or never run that way: "
              f"{', '.join(stale)}\n  run python run_notebooks.py so outputs/ matches your code.")
    if errors:
        warnings += 1
        print("\nwarning: the last clean run had errors in: " + ", ".join(errors))
    if todo:
        warnings += 1
        print(f"\nwarning: {len(todo)} questions still TODO (you can still submit; unanswered "
              "questions get no marks):")
        print_list(todo)

    name, roll = identity(ROOT)
    unfilled = [f for f, v in (("name", name), ("roll number", roll)) if not v]
    if unfilled:
        print(f"\nplease fill in your {' and '.join(unfilled)} at the top of task1.ipynb and "
              "run this again: without them your submission cannot be matched to you.")
    elif warnings:
        print(f"\n{out_path.name} is ready to submit as it is; fix the warnings above and run "
              "this again if you want to include more.")
    else:
        print(f"\n{out_path.name} is ready to submit: every check passes, all required files "
              "are included and every question is answered.")


if __name__ == "__main__":
    main()
