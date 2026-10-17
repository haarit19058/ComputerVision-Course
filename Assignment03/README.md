# Lunar Landing: The Final Descent · Notebooks

The lander's inertial log for the final descent is lost. Only the descent
camera survived: 144 consecutive images from frame 270 to touchdown. You will
recover the lander's **velocity and 3D trajectory** from the images, with
classical computer vision that you implement yourself.

The assignment text is in the handout, `ES666_A3_Handout.pdf`, and in
`tasks.md`. You work in four Jupyter notebooks, one per task. Everything that
is not computer vision is provided (image loading, SIFT matching, plotting,
the checks, and all of Task 4's trajectory pipeline), so you can focus on
optical flow: gradients, then Lucas-Kanade, then its iterative and SIFT-guided
versions, then connecting those functions to the provided pipeline.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab                          # then open task1.ipynb
```

VS Code with its Jupyter extension works too. Always open the notebooks from
this folder, so they find `provided/`, `data/` and each other.

## What is where

| Path | What it is |
|------|------------|
| `task1.ipynb` ... `task4.ipynb` | the four tasks: your code, the checks, the figures and your answers |
| `provided/` | all the provided code and its tests, which the notebooks use; you do not need to open it: image I/O, bilinear sampling, SIFT matching, plotting, the checks and the report builder, the steps each notebook runs (`task1/` to `task3/`), and Task 4's camera-motion fit, flight code, pipeline, tracker comparison and 3D animation (`task4/`) |
| `data/` | `frames/` (270 to 413) and `mission_constants.json` |
| `run_notebooks.py` | runs all four notebooks from a clean start |
| `make_submission.py` | builds `report.html` and `submission.zip` |

## How the notebooks work

* **Your code** goes in the cells whose first line starts with `# YOUR CODE`.
  Only those cells are graded, and they are read from the **saved** notebook
  file. Keep function names and arguments as given: the checks and the
  provided cells call them. A helper function you want graded goes in a
  `# YOUR CODE` cell too (you may add new ones).
* **Your own cells** (experiments, extra plots, notes) can go anywhere. They
  are never graded and cannot break the checks or the report.
* **Checks:** the `check("1.1")` cell after your code runs that subtask's
  tests on your functions and prints PASS or FAIL. `python -m pytest provided/task1`
  runs the same tests on the saved notebook, the way the grader does.
* **Figures and numbers:** the provided cells show each figure below the cell
  and save it, with the numbers, in `outputs/`. The report uses those files,
  not what is stored in the notebook, so nothing is lost if outputs are
  cleared.
* **Answers:** write each answer in the cell right below its question, in one
  or two sentences, replacing *TODO*. Extra cells you add below a question
  count as part of its answer.
* **Later tasks use your earlier ones:** the setup cell of `task2.ipynb` loads
  your Task 1 functions from the saved `task1.ipynb`, and so on. Save a
  notebook (Ctrl+S) before you move on to the next.

## Running everything from a clean start

```bash
python run_notebooks.py              # all four, in order (about 10 minutes)
python run_notebooks.py task3        # just one
```

It runs each notebook in a fresh kernel, top to bottom, so your results do
not depend on cells you ran earlier in another order. It only reads your
notebooks, so it is safe while they are open (save them first). It keeps
going past errors and lists every cell that failed at the end. Tasks 4.2 and
4.4 track all 143 frame pairs and take a few minutes; the comparison saves
its tracks and reuses them while your code stays the same.

## Large motion: SIFT (provided)

Lucas-Kanade only converges when it starts near the answer, and near
touchdown the ground moves further than that between frames. `provided/sift.py`
is a from-scratch NumPy SIFT that you use but do not write:

```python
matches = sift_matches(image0, image1)          # (points0, points1)
tracked, valid = track_with_sift(track_points, image0, image1, points, matches)
```

`track_with_sift` tracks each point from (0, 0) and from its SIFT guess (the
`initial` argument of your `track_point`) and keeps the better fit.

## Conventions

* **Images:** float32 arrays of shape (H, W) in [0, 1], indexed `image[y, x]`.
* **Points:** (N, 2) arrays of `[x, y]` in pixels.
* **Motion model (Task 4, provided):** `{"zoom": s, "rotation": theta,
  "shift": [tx, ty]}`, measured about the image centre (see
  `provided/task4/motion_model.py`).
* **Trajectory axes (world frame, Task 4, provided):** right-handed, fixed to
  the first image, in metres. `X` points to the right of that image, `Y`
  toward its **top** (image `y` points down, so `Y` is the opposite of image
  `y`), and `Z` is the altitude, upward. Positions are `[X, Y, Z]`, velocities
  `[v_X, v_Y, v_Z]`, and `v_Z < 0` while the lander descends. The handout has
  a figure of these axes; see also `provided/task4/trajectory.py`.

  ```text
            Z (altitude, up, toward the camera)
            |
            |    Y (toward the top of the first image)
            |   /
            |  /
            | /
            +-----------> X (to the right of the first image)
  ```
* **Speed:** the provided steps work on frames shrunk to a quarter of their size.

## Rules

* **Allowed:** Python 3, NumPy (including `np.linalg.svd`, `lstsq`, `eig`,
  `norm`), Pillow, Matplotlib and the Python standard library. Jupyter,
  `pytest` and `nbclient` are only for running the notebooks and checks.
* **Not allowed:** OpenCV, scikit-image, SciPy, PyTorch, TensorFlow, JAX,
  Kornia, pretrained models, ready-made motion estimation, and `np.gradient`.
* `plotly` (in `requirements.txt`) is used only by the provided 3D animation;
  do not use it in your own code.
* `provided/test_common.py` checks the imports in every notebook cell, and
  `np.gradient` in your `# YOUR CODE` cells, for you.

## What to submit

Fill in your **name and roll number** at the top of `task1.ipynb`, run
`python run_notebooks.py`, then

```bash
python make_submission.py
```

It builds `report.html` (your answers, your code and the figures, in one file
that opens in any browser) from the saved notebooks and `outputs/`, and
writes `submission.zip` with the four notebooks, `report.html` and the
`outputs/` files used for grading (nothing from `data/`). You can submit the
zip as it is; unfinished parts only lose their own marks. It warns about:

* checks that fail on your saved notebooks;
* required `outputs/` files that are missing;
* notebooks whose code changed after your last `run_notebooks.py`;
* questions still marked *TODO*.

The one thing you must fix is a missing name or roll number: without them
your submission cannot be matched to you.

Follow any extra submission instructions from the course staff.

## Questions

Ask on Google Classroom first, so the rest of the class sees the answer too;
email shekhar.hement@iitgn.ac.in if it needs to stay private.

---

🤖 Yes, AI can solve this assignment in one prompt. Your trajectory will be reconstructed. Your understanding will not.
