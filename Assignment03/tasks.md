# Lunar Landing: The Final Descent

## ES666 Assignment 3: Velocity and 3D Trajectory from a Descent Camera

> **T+00:23:10.** The lander has pitched over and is looking straight down
> at the surface. Its inertial log for the final descent is corrupted:
> mission control no longer knows how fast it came down, where it drifted,
> or when it stopped to hover. Only the descent camera survived, 144
> consecutive images from frame 270 to touchdown at frame 413.
>
> Your team has one objective: **recover the lander's velocity and its 3D
> trajectory from the images alone, using classical computer vision.** No
> neural networks, no OpenCV, no black-box trackers. Track the surface,
> read the camera's motion from how the ground moves in the image, and
> turn it into altitude, speed and a path in 3D.

## Mission Architecture

``` text
TASK 1               TASK 2            TASK 3               TASK 4
Find Where the   ->  Track One     ->  Track Fast       ->  Reconstruct
Image Changes        Point             Motion               the Flight
gradients, I_t       Lucas-Kanade      iterative LK         your trackers in
trackable points     one step          SIFT-guided LK       the provided pipeline
```

### Mission data

-   `data/frames/`: frames 270 to 413 of the descent camera (PNG, 2352 x
    1728 pixels), consecutive.
-   `data/mission_constants.json`: provided by mission control: the camera
    focal length in pixels (at full resolution), the frame
    interval in seconds, and the altitude of the lander at one frame.

The frames are real images from the Chang'e-5 landing camera, taken on
1 December 2020 during the last three minutes before touchdown (level 2A
data from the Ground Research and Application System of China's Lunar and
Planetary Exploration Program, https://moon.bao.ac.cn). Cite this source in
your reports.

### Allowed libraries

-   Python 3
-   NumPy
-   Pillow (PIL)
-   Matplotlib
-   Python standard library

NumPy linear algebra such as `np.linalg.svd`, `np.linalg.lstsq`,
`np.linalg.eig`, and `np.linalg.norm` is allowed.

The starter code provides a from-scratch SIFT matcher (`provided/sift.py`).
You do not implement SIFT; use it to give Lucas-Kanade a starting guess
when the motion between frames is large (see Task 3).

`plotly` (in `requirements.txt`) is used only by the provided 3D animation
(`task4.animate_trajectory`); do not use it in your own code.

### Prohibited

Do not use OpenCV, scikit-image, SciPy, PyTorch/TensorFlow/JAX/Kornia,
pretrained optical-flow models, pretrained feature
detectors/descriptors, or ready-made motion-estimation routines.

------------------------------------------------------------------------

# TASK 1: FIND WHERE THE IMAGE CHANGES (20 marks)

### Objective

Learn how motion is written into image intensities: compute image
gradients, connect them to motion, and find the points that can be
tracked.

### Story

> "The ground is still there in every frame."\
> "Then learn to read it. Every change of brightness is a clue to how we
> moved."

### Required work

#### 1. Spatial gradients (6 marks)

Implement, without `np.gradient`:

$$
I_x(x,y)=\frac{I(x+1,y)-I(x-1,y)}{2},\qquad I_y(x,y)=\frac{I(x,y+1)-I(x,y-1)}{2}.
$$

Show $I_x$, $I_y$ and the gradient magnitude of frame 276.

**Answer in the notebook:** (a) Where are the gradients strongest, and where are they
weakest?

#### 2. Temporal change and motion (5 marks)

Implement $I_t = I_1 - I_0$ and check $I_x u + I_y v + I_t \approx 0$
numerically on a synthetic shift of known $(u, v)$.

**Answer in the notebook:** (a) What mean residual does the check report
(saved in `constraint_check.json`)? (b) Why can one pixel not give both $u$
and $v$?

#### 3. Choosing good points to track (9 marks)

For each local window build $G = A^TA$, with one row $[I_x, I_y]$ of
$A$ per pixel, and use $q(x,y)=\min(\lambda_1,\lambda_2)$ as a
trackability score. Select strong points with a minimum separation and
show them on frame 276. The same matrix $G$ is what Lucas-Kanade
solves in Task 2.

**Answer in the notebook:** (a) What does the minimum distance between selected points
prevent? (b) What do the eigenvalues of $G$ look like on a
flat patch, an edge and a corner? (c) Why does a large $q$ mean a
point is easy to track?

### Deliverables

-   `task1.ipynb`: your Task 1 code, the check cells run, and your answers
-   what it saves in `outputs/task1/`: `gradients.png`, `constraint.png`,
    `features.png` and `constraint_check.json`

------------------------------------------------------------------------

# TASK 2: TRACK ONE POINT (30 marks)

### Objective

Estimate the motion of one point with Lucas-Kanade, and find out when a
single step is enough.

### Story

> "Pick one rock and follow it."\
> "One step at a time."

### Required work

#### 1. Lucas-Kanade as least squares (12 marks)

Assume constant motion in a local window and solve, with NumPy linear
algebra,

$$
A\begin{bmatrix}u\\ v\end{bmatrix}\approx b,\qquad b=-\begin{bmatrix}I_{t1}\\ \vdots\\ I_{tN}\end{bmatrix}.
$$

**Answer in the notebook:** (a) Write the least-squares solution for $(u, v)$ using
$G = A^TA$. (b) When can it not be solved? Relate this to $q$ from
Task 1.3.

#### 2. Estimate one-step motion (9 marks)

Sample the gradient and temporal-difference windows at a point (with the
provided bilinear sampling) and solve once. The provided tests and the
Task 2.3 sweep check it on synthetic sub-pixel shifts.

**Answer in the notebook:** (a) How large is the error of `one_step_flow` on
sub-pixel shifts? See the smallest shifts of the Task 2.3 sweep
(`error_sweep.json`).

#### 3. When one step is not enough (9 marks)

Run the provided sweep of synthetic shifts from 0.1 to 8 px, and
one-step flow on two real pairs: pair A, 375 to 376, where the motion is
small, and pair B, 300 to 301, where it is much larger.

**Answer in the notebook:** (a) Above what shift does the one-step error grow quickly?
(b) Why does one step come close on pair A (375 to 376) but badly
undershoot pair B (300 to 301)?

### Deliverables

-   `task2.ipynb`: your Task 2 code, the check cells run, and your answers
-   what it saves in `outputs/task2/`: `error_sweep.png`, `error_sweep.json`,
    `one_step_small_motion.png` and `one_step_large_motion.png`

------------------------------------------------------------------------

# TASK 3: TRACK FAST MOTION (30 marks)

### Objective

Make Lucas-Kanade follow large motion: iterate it, track many points
reliably, and start it from SIFT matches when the motion is too large.

### Story

> "Near touchdown some rocks jump further than a tracker can reach."\
> "The archive recognises them by their signature (SIFT)."

### Required work

#### 1. Iterative Lucas-Kanade (9 marks)

Start from $(u,v)=(0,0)$ or from a given initial guess, warp the second
image with bilinear sampling, and repeat the Task 2 update until the
increment is small or a maximum iteration count is reached. The provided
cell adds your iterative curve to the error sweep of Task 2.3.

**Answer in the notebook:** (a) Up to what shift does the iterative tracker stay accurate?

#### 2. Track many points (6 marks)

Track many points at once and mark a track as failed when its window
leaves either image or the least-squares system cannot be solved.

**Answer in the notebook:** (a) Can a track that is marked valid still be wrong? Give an
example from your results.

#### 3. Use SIFT to handle large motion (7 marks)

Use the provided `provided/sift.py`: `track_with_sift` runs your tracker
from $(0,0)$ and from a SIFT starting guess and keeps the better fit.

**Answer in the notebook:** (a) Why do iterative and SIFT-guided tracking give
almost the same result on 375 to 376, but very different results on 300
to 301? Use the numbers of Task 3.4.

#### 4. Compare the three trackers (8 marks)

Compare one-step, iterative and SIFT-guided tracking on the two pairs of
Task 2.3 and on a third pair near touchdown (409 to 410); the provided
cells overlay all three trackers in one figure per pair.

**Answer in the notebook:** fill in the table of mean displacements and
failed-track rates (0 to 1). (a) Which tracker is right on each pair? (b) Why does
SIFT-guided tracking lose many points at 409 to 410?

### Deliverables

-   `task3.ipynb`: your Task 3 code, the check cells run, and your answers
-   what it saves in `outputs/task3/`: `error_sweep.png`, `error_sweep.json`,
    and for each of the three pairs the `*_compare.png` figure and the
    `*_statistics.json` numbers

------------------------------------------------------------------------

# TASK 4: RECONSTRUCT THE FLIGHT (20 marks)

### Objective

Connect your trackers to the provided template code and recover the
lander's velocity and 3D trajectory, then find out how the quality of
your optical flow decides the result.

### Story

> "Mission control has one altitude reading and the camera's focal
> length."\
> "Then we can put the whole descent in metres. Fly it back."

### Provided code

Everything in Task 4 is provided (the frame loop, the camera-motion fit
in `provided/task4/motion_model.py`, the flight code in `provided/task4/trajectory.py`, the
plots and the tracker comparison) except the four short functions of
Task 4.1 in `task4.ipynb`.

The camera points straight down at flat ground. As the lander moves, the
tracked ground features between two consecutive frames undergo a
similarity transformation: they can scale, rotate and translate.

$$
\mathbf{x}_{k+1} = s_k R(\theta_k)\,\mathbf{x}_k + \mathbf{t}_k.
$$

Using the tracked points from Tasks 1 to 3, `provided/task4/motion_model.py`
estimates the scale $s_k$, rotation $\theta_k$ and translation
$\mathbf{t}_k$ by least squares. The scale is related to the lander's
altitude:

$$
s_k = \frac{Z_k}{Z_{k+1}}.
$$

A robust fitting step removes incorrectly tracked points and refits the
motion.

Finally, `provided/task4/trajectory.py` converts the estimated motion over all
frames into altitude, position, velocity and the lander's 3D trajectory,
in metres and seconds. Neither file is yours to write, but you should be
able to explain them: they are built entirely from your Task 1 to 3
tracks. Read them, run them through the notebook, and interpret what they
produce.

Your goal: understand how simple 2D image motion can be used to recover
the lander's motion through 3D space.

The trajectory uses a fixed, right-handed world frame attached to
frame 270 (see the axes figure in the handout):

| Axis | Direction | Velocity |
|------|-----------|----------|
| $X$ | to the right of frame 270 (along image $x$) | $v_X$ |
| $Y$ | toward the top of frame 270 (opposite to image $y$, which points down) | $v_Y$ |
| $Z$ | the altitude, upward, toward the camera | $v_Z$, negative while descending |

### Required work

#### 1. Connect your trackers to the pipeline (4 marks)

Fill in the Task 4.1 cell of `task4.ipynb`: the adapters `one_step`,
`iterative` and `sift_guided`, which call your Task 2 and Task 3 code,
and `track_pair`, which selects features (Task 1), runs a tracker and
keeps the valid tracks. The tests check the wiring on synthetic data.

**Answer in the notebook:** (a) Why must `track_pair` drop the failed tracks
before the motion fit?

#### 2. Read the motion from the images (4 marks)

The provided cell tracks the whole descent with your SIFT-guided tracker;
look at the zoom, rotation and shift plots.

**Answer in the notebook:** (a) Why is the zoom $s_k = Z_k / Z_{k+1}$? Use the pinhole
model. (b) What are the phases of the descent, and at about which frames
do they change? (c) Which pattern in these plots is not caused by the
lander, and how can you tell?

#### 3. Reconstruct the flight (4 marks)

Look at the altitude, the velocity $(v_X, v_Y, v_Z)$, the time to
touchdown $\tau = Z / (-v_Z)$, the 3D trajectory and the ground track
(seen from above, $X$ to the right and $Y$ up).

**Answer in the notebook:** fill in the summary table. (a) Which result do you trust
least, and why?

#### 4. Compare the three trajectories (5 marks)

The provided cell of Task 4.4 flies the pipeline once per
tracker (one-step, iterative, SIFT-guided) on the same points and draws
the three ground tracks together, seen from above.

**Answer in the notebook:** fill in the table from `compare/summary.csv`. (a) Which
trajectory do you trust, and which Task 3 result supports it? (b) Does a
low median residual mean a tracker is right? Use your numbers.

An optional provided cell turns the same three
flights into an interactive 3D animation,
`outputs/task4/trajectory_animation.html`, that you open in a web browser:
play or scrub through the descent, rotate and zoom the scene, and hide
trackers from the legend. It is the same comparison as the ground track,
seen in 3D. It is not graded; it exists so you can watch, not just read,
how far a tracker that cannot follow large motion drifts from the
others.

#### 5. Where does uncertainty come from? (3 marks)

**Answer in the notebook:** (a) Why does a small error in each zoom grow along the
descent? (b) What changes if the anchor altitude is 10 % too high? And if
the focal length is?

### Deliverables

-   `task4.ipynb`: your Task 4.1 code, the check cell run, your answers, and
    the Final Mission Report at its end
-   what it saves in `outputs/task4/`: the motion and flight figures,
    `summary.json`, `compare/ground_track.png` and `compare/summary.csv`

------------------------------------------------------------------------

# Final Mission Report: What Did We Learn? (bonus, 5 marks)

Answer at the end of `task4.ipynb`, in one or two
sentences each, with one piece of evidence from your results.

1.  Where in the descent does "all pixels in a window move together" fail
    most, and what in your results shows it?
2.  Would a pyramidal (coarse-to-fine) KLT tracker fix 300 to 301? Would
    it fix 409 to 410? Why?
3.  If you ignored the pattern from Task 4.2, which Task 4.3 results would
    be wrong, and by how much?
4.  Where does the "flat ground, camera straight down" model fail, and
    what error does it cause?
5.  Name one independent measurement that would check your trajectory,
    and what it would check.

# Deliverables and Marks

| Task | Topic | Marks |
|------|-------|------:|
| 1 | Gradients, motion constraint, trackable points | 20 |
| 2 | Lucas-Kanade, one step | 30 |
| 3 | Iterative + SIFT-guided tracking | 30 |
| 4 | Tracker wiring + trajectory interpretation | 20 |
| **Total** | | **100** |
| Bonus | Final Mission Report (five questions) | 5 |

## Working Instructions

1.  Read `README.md` for setup and how the notebooks work, and this file
    (or the handout) for what each task asks you to implement and
    interpret.
2.  Work in `task1.ipynb` to `task4.ipynb`, in order. Write your code only
    in the cells whose first line starts with `# YOUR CODE`: only those are
    graded, read from the saved notebook. You may add your own cells for
    experiments and notes. Everything in `provided/` is provided and must
    not be changed.
3.  Verify as you go: each `check(...)` cell tests your functions
    (`python -m pytest provided/taskN` runs the same tests on the saved notebook),
    and the provided cells show your results on the real descent frames and
    save them in `outputs/`.
4.  Before you submit, run everything from a clean start with
    `python run_notebooks.py` (or restart the kernel and run all cells of
    each notebook), and check that it finishes without errors.
5.  Fill in your name and roll number at the top of `task1.ipynb`, then run
    `python make_submission.py`: it builds `report.html` and writes
    `submission.zip` (the notebooks, the report and the `outputs/` files
    used for grading, nothing from `data/`) and warns about anything still
    missing or TODO; you can still submit a partial report.
6.  The 3D animation cell at the end of Task 4.4 is optional and not
    graded: it writes `outputs/task4/trajectory_animation.html`, to open in
    a browser. Watch it for intuition; do not write it up.
7.  Questions: ask on Google Classroom first, so the rest of the class sees
    the answer too; email shekhar.hement@iitgn.ac.in if it needs to stay
    private.
