"""Run the notebooks from a clean start.  PROVIDED: you do not need to change this file.

Run from this folder (any system, no bash needed):

    python run_notebooks.py                # task1 to task4, in order (about 10 minutes)
    python run_notebooks.py task3 task4    # only some of them

Each notebook runs in a fresh kernel, top to bottom, like "restart the kernel
and run all cells", so your results cannot depend on cells you ran earlier in
another order. Your notebook files are only read, never changed, so this is
safe while they are open in Jupyter; save them first, since the saved files
are what runs. The run writes:

    outputs/                    the figures and numbers, as the notebook cells make them
    outputs/notebooks/          executed copies of the notebooks, with every output
    outputs/run_stamp.json      what ran, so make_submission.py can tell whether
                                your code changed after the last run

It keeps going past errors and lists every cell that failed at the end,
including cells you added yourself.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIMEOUT_S = 3600          # per cell: tracking the whole descent takes minutes


def describe(cell, index):
    from provided.notebook import cell_source, provided_info, your_code_label
    kind = ("your code" if your_code_label(cell) else
            "provided" if provided_info(cell) else "a cell you added")
    lines = [l for l in cell_source(cell).splitlines() if l.strip()]
    first = (lines[0].strip() if lines else "")[:60]
    return f"cell {index} ({kind}: {first})"


def run_one(name):
    import nbformat
    from nbclient import NotebookClient
    from provided.notebook import fingerprint, read_notebook

    path = ROOT / f"{name}.ipynb"
    if not path.exists():
        return {"missing": True}, [f"{path.name} not found"]
    saved = read_notebook(path)
    notebook = nbformat.reads(json.dumps(saved), as_version=4)
    client = NotebookClient(notebook, timeout=TIMEOUT_S, kernel_name="python3",
                            allow_errors=True, resources={"metadata": {"path": str(ROOT)}},
                            extra_arguments=["--IPKernelApp.log_level=ERROR"])
    started = time.time()
    problems = []
    try:
        client.execute()
    except Exception as exc:             # the kernel died or a cell ran out of time
        problems.append(f"the run stopped early: {type(exc).__name__}: {str(exc)[:200]}")
    for index, cell in enumerate(notebook.cells, 1):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                problems.append(f"{describe(cell, index)}: {output.get('ename')}: "
                                f"{str(output.get('evalue', ''))[:160]}")
    copies = ROOT / "outputs" / "notebooks"
    copies.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, str(copies / path.name))
    return {"fingerprint": fingerprint(saved), "errors": len(problems),
            "seconds": round(time.time() - started), "finished": time.strftime("%Y-%m-%d %H:%M"),
            }, problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        import nbclient  # noqa: F401
        import ipykernel  # noqa: F401
    except ImportError:
        print("run_notebooks.py needs Jupyter: pip install -r requirements.txt")
        return 1
    sys.path.insert(0, str(ROOT))
    names = [a.removesuffix(".ipynb") for a in argv] or ["task1", "task2", "task3", "task4"]
    stamp_path = ROOT / "outputs" / "run_stamp.json"
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        stamp = {}
    all_problems = {}
    for name in names:
        print(f"running {name}.ipynb from a clean kernel ...", flush=True)
        entry, problems = run_one(name)
        if not entry.get("missing"):
            stamp[name] = entry
            print(f"  done in {entry['seconds']} s, {len(problems)} cell(s) with errors")
        all_problems[name] = problems
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    failed = {n: p for n, p in all_problems.items() if p}
    if failed:
        print("\ncells that failed (the rest of each notebook still ran):")
        for name, problems in failed.items():
            for problem in problems:
                print(f"  {name}.ipynb, {problem}")
    print("\nexecuted copies are in outputs/notebooks/, figures and numbers in outputs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
