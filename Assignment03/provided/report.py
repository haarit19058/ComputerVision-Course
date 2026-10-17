"""Build report.html from your saved notebooks and the files in outputs/.  PROVIDED.

The report never uses the outputs stored inside a notebook, so it is complete
even if those were cleared or do not display on another computer:

* figures come from the files the provided cells save in outputs/;
* answers come from the markdown cells of the saved notebooks: everything you
  write below a question, up to the next provided cell, belongs to it (extra
  cells you add there are included too);
* your code comes from your YOUR CODE cells.

Provided cells are found by a hidden tag, not by their position, so cells you
add, move or delete elsewhere cannot break the report. A note you add away
from any question is kept, under "Other notes".
"""

import base64
import html
import json
import time
from pathlib import Path

from provided.notebook import (NOTEBOOKS, ROOT, cell_source, fingerprint, provided_info,
                             read_notebook, your_code_label)

FIGURES = {
    "1.1": [("task1/gradients.png", "Frame 276 with its gradients Ix, Iy and the gradient magnitude")],
    "1.2": [("task1/constraint.png", "The motion constraint on frame 276 shifted by a known (u, v)")],
    "1.3": [("task1/features.png", "The points selected for tracking on frame 276")],
    "2.3": [("task2/error_sweep.png", "One-step error against the size of a synthetic shift"),
            ("task2/one_step_small_motion.png", "One-step tracks on pair A, frames 375 to 376"),
            ("task2/one_step_large_motion.png", "One-step tracks on pair B, frames 300 to 301")],
    "3.1": [("task3/error_sweep.png", "One-step and iterative error against the shift size")],
    "3.4": [("task3/small_motion_compare.png", "The three trackers on frames 375 to 376"),
            ("task3/large_motion_compare.png", "The three trackers on frames 300 to 301"),
            ("task3/touchdown_compare.png", "The three trackers on frames 409 to 410")],
    "4.2": [("task4/zoom.png", "Zoom between consecutive frames"),
            ("task4/rotation.png", "Rotation between consecutive frames"),
            ("task4/shift.png", "Shift between consecutive frames")],
    "4.3": [("task4/altitude.png", "Altitude Z over the descent"),
            ("task4/velocity.png", "Velocity v_X, v_Y and v_Z"),
            ("task4/time_to_touchdown.png", "Time to touchdown"),
            ("task4/trajectory_3d.png", "The reconstructed 3D trajectory"),
            ("task4/ground_track.png", "The ground track, seen from above")],
    "4.4": [("task4/compare/ground_track.png", "The ground tracks of the three trackers")],
}
NUMBERS = {"1.2": ["task1/constraint_check.json"],
           "3.4": ["task3/small_motion_statistics.json", "task3/large_motion_statistics.json",
                   "task3/touchdown_statistics.json"],
           "4.3": ["task4/summary.json"], "4.4": ["task4/compare/summary.csv"]}


# ---- reading the notebooks --------------------------------------------------

def _is_todo(text):
    return not text.strip() or "TODO" in text


def identity(root=ROOT):
    """(name, roll number) from the top of task1.ipynb, "" where still TODO."""
    found = {"Name": "", "Roll number": ""}
    try:
        cells = read_notebook(Path(root) / NOTEBOOKS[1]).get("cells", [])
    except (FileNotFoundError, ValueError, UnicodeDecodeError):
        return "", ""
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        for line in cell_source(cell).splitlines():
            for field in found:
                if line.strip().startswith(f"**{field}:**") and not found[field]:
                    value = line.split(":**", 1)[1].strip()
                    found[field] = "" if "TODO" in value else value.strip("* ")
    return found["Name"], found["Roll number"]


def read_task(task, root=ROOT):
    """The report blocks of one notebook, in order, and its answers.

    blocks : list of (kind, data); kind is "heading", "figures", "code",
             "question", "answer", "extra" or "notes"
    answers: {"1.2 (a)": text}
    """
    path = Path(root) / NOTEBOOKS[task]
    try:
        cells = read_notebook(path).get("cells", [])
    except FileNotFoundError:
        return [("missing", f"{path.name} was not found")], {}
    except (ValueError, UnicodeDecodeError) as exc:
        return [("missing", f"{path.name} could not be read ({exc})")], {}
    blocks, answers, current, shown = [], {}, None, set()
    for cell in cells:
        info = provided_info(cell)
        source = cell_source(cell)
        label = your_code_label(cell)
        if info:
            role, part = info.get("role"), info.get("part")
            current = None
            if role in ("title", "heading"):
                blocks.append(("heading", source))
                if part in FIGURES and part not in shown:
                    blocks.append(("figures", part))
                    shown.add(part)
            elif role == "question":
                key = f"{part} {info.get('label', '')}".strip() if part not in (
                    "bonus", "references") else info.get("label", part)
                blocks.append(("question", source))
                current = key
                answers.setdefault(key, "")
            elif role == "your-code":
                blocks.append(("code", (label or f"Task {part}", source)))
            continue
        if label:                                  # a YOUR CODE cell you added
            blocks.append(("code", (label, source)))
            continue
        if cell.get("cell_type") == "markdown":
            if "**Name:**" in source or "**Roll number:**" in source:
                continue                           # shown at the top of the report
            if current is not None:
                answers[current] = (answers[current] + "\n\n" + source).strip()
                blocks.append(("answer", source))
            elif source.strip():
                blocks.append(("notes", source))
        elif cell.get("cell_type") == "code" and source.strip():
            images = [o["data"]["image/png"] for o in cell.get("outputs", [])
                      if isinstance(o.get("data"), dict) and "image/png" in o["data"]]
            blocks.append(("extra", (source, images)))
    for part in FIGURES:                           # a figure heading that was deleted
        if part.startswith(str(task)) and part not in shown:
            blocks.append(("figures", part))
    return blocks, answers


def unanswered(root=ROOT):
    """Every question whose answer is still empty or TODO, like "1.2 (b)" or "Q3"."""
    todo = []
    for task in NOTEBOOKS:
        _, answers = read_task(task, root)
        todo += [key for key, text in answers.items() if _is_todo(text)]
    return todo


# ---- writing the report -----------------------------------------------------

CSS = """
:root { --ink:#1b2430; --muted:#5b6674; --rule:#d9dee5; --tint:#f4f6f9; --accent:#a5680e;
        --bad:#9b2c2c; }
body { margin:0; background:#fff; color:var(--ink);
       font:16px/1.55 Georgia, 'Times New Roman', serif; }
main { max-width:900px; margin:0 auto; padding:24px 20px 80px; }
h1, h2, h3 { font-family:system-ui, 'Segoe UI', Roboto, sans-serif; line-height:1.25; }
h1 { font-size:26px; margin:34px 0 8px; } h2 { font-size:20px; margin:28px 0 6px; }
.meta { font-family:system-ui, sans-serif; color:var(--muted); font-size:14px; }
.meta b { color:var(--ink); }
.question { margin:22px 0 4px; font-weight:bold; }
.answer { border-left:3px solid var(--accent); background:var(--tint);
          padding:6px 14px; margin:4px 0 12px; }
.answer.todo { border-left-color:var(--bad); }
.notes { border-left:3px solid var(--rule); padding:4px 14px; color:var(--muted); }
figure { margin:14px 0; } figure img { max-width:100%; border:1px solid var(--rule); }
figcaption { font:13px system-ui, sans-serif; color:var(--muted); }
.missing { font:14px system-ui, sans-serif; color:var(--bad); }
details { margin:8px 0; font:14px system-ui, sans-serif; }
pre { background:var(--tint); padding:10px; overflow-x:auto; font-size:13px; }
table { border-collapse:collapse; } td, th { border:1px solid var(--rule); padding:3px 8px; }
.raw { white-space:pre-wrap; font:14px/1.5 ui-monospace, monospace; background:none; padding:0; }
"""

SCRIPTS = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/markdown-it@14.1.0/dist/markdown-it.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/markdown-it-texmath@1.0.0/texmath.min.js"></script>
"""

RENDER = """
<script>
window.addEventListener("load", function () {
  if (!window.markdownit) return;          /* offline: the plain text stays, still readable */
  var md = window.markdownit({html: false, linkify: true});
  if (window.texmath && window.katex) {
    md.use(window.texmath, {engine: window.katex, delimiters: "dollars",
                            katexOptions: {throwOnError: false}});
  }
  document.querySelectorAll(".md").forEach(function (el) {
    el.innerHTML = md.render(el.getAttribute("data-md"));
  });
});
</script>
"""


def _md(text, css=""):
    escaped = html.escape(text)
    return (f'<div class="md {css}" data-md="{html.escape(text, quote=True)}">'
            f'<pre class="raw">{escaped}</pre></div>')


def _figures(part, outputs):
    out = []
    for rel, caption in FIGURES[part]:
        path = outputs / rel
        if path.is_file():
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            out.append(f'<figure><img alt="{html.escape(caption)}" '
                       f'src="data:image/png;base64,{data}">'
                       f'<figcaption>{html.escape(caption)} (outputs/{rel})</figcaption></figure>')
        else:
            out.append(f'<p class="missing">figure not found: outputs/{rel} (run the notebook)</p>')
    for rel in NUMBERS.get(part, []):
        path = outputs / rel
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            out.append(f"<details><summary>numbers: outputs/{rel}</summary>"
                       f"<pre>{html.escape(text)}</pre></details>")
    return "\n".join(out)


def build_report(root=ROOT, out=None):
    """Write report.html for the notebooks in `root`; returns (path, unanswered list)."""
    root = Path(root)
    out = Path(out) if out else root / "report.html"
    outputs = root / "outputs"
    name, roll = identity(root)
    try:
        stamp = json.loads((outputs / "run_stamp.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        stamp = {}
    parts, todo = [], []
    for task in NOTEBOOKS:
        blocks, answers = read_task(task, root)
        todo += [k for k, v in answers.items() if _is_todo(v)]
        try:
            fresh = stamp.get(f"task{task}", {}).get("fingerprint") == fingerprint(
                read_notebook(root / NOTEBOOKS[task]))
        except (FileNotFoundError, ValueError, UnicodeDecodeError):
            fresh = False
        parts.append(f'<p class="meta">{NOTEBOOKS[task]}: ' +
                     ("figures made by a clean run of this code (run_notebooks.py)" if fresh else
                      "no clean run of this exact code recorded; figures are from the last "
                      "time the cells ran") + "</p>")
        pending_answer = False
        for kind, data in blocks:
            if kind == "missing":
                parts.append(f'<p class="missing">{html.escape(data)}</p>')
            elif kind == "heading":
                parts.append(_md(data))
            elif kind == "figures":
                parts.append(_figures(data, outputs))
            elif kind == "code":
                label, source = data
                parts.append(f"<details><summary>code: {html.escape(label)}</summary>"
                             f"<pre>{html.escape(source)}</pre></details>")
            elif kind == "question":
                if pending_answer:
                    parts.append('<div class="answer todo"><p class="missing">no answer</p></div>')
                parts.append(f'<div class="question">{_md(data)}</div>')
                pending_answer = True
                continue
            elif kind == "answer":
                parts.append(_md(data, "answer todo" if _is_todo(data) else "answer"))
            elif kind == "extra":
                source, images = data
                imgs = "".join(f'<img src="data:image/png;base64,{i}">' for i in images)
                parts.append(f"<details><summary>a code cell you added</summary>"
                             f"<pre>{html.escape(source)}</pre>{imgs}</details>")
            elif kind == "notes":
                parts.append('<p class="meta">Other notes</p>' + _md(data, "notes"))
            pending_answer = False if kind in ("answer",) else pending_answer
        if pending_answer:
            parts.append('<div class="answer todo"><p class="missing">no answer</p></div>')
    title = f"ES666 A3 report{': ' + name if name else ''}"
    header = (f'<p class="meta"><b>Name:</b> {html.escape(name) or "not filled in"} &nbsp; '
              f'<b>Roll number:</b> {html.escape(roll) or "not filled in"} &nbsp; '
              f'built {time.strftime("%Y-%m-%d %H:%M")} &nbsp; '
              f'{len(todo)} question(s) unanswered</p>')
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style>{SCRIPTS}</head>"
            f"<body><main><p class=\"meta\">Lunar Landing: The Final Descent · ES666 Assignment 3"
            f"</p>{header}{''.join(parts)}</main>{RENDER}</body></html>")
    out.write_text(page, encoding="utf-8")
    return out, todo
