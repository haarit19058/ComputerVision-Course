# ES666 Computer Vision: End-Semester Study Guide

**Syllabus:** projective geometry through GANs and diffusion.
Gradient descent and backprop are skipped because you already know them.

**Sources used**
- **Handwritten notes p90–191:** geometry and optical flow, in Hartley & Zisserman notation.
- **Course slides:** chapters 9–34 of *Foundations of Computer Vision* (Torralba, Isola, Freeman), free at https://visionbook.mit.edu
- **Understanding Deep Learning** (Prince), free at https://udlbook.github.io/udlbook/ for GAN (Ch 15), VAE (Ch 17), diffusion (Ch 18) and transformers (Ch 12).
- **Stanford CS231A course notes** for epipolar geometry and SfM.
- **Szeliski, *Computer Vision: Algorithms and Applications* 2e** (https://szeliski.org/Book): Ch 2, 8, 11, 12.
- **Hartley & Zisserman, *Multiple View Geometry*:** the reference your instructor's notation follows.

## Plan

| # | Session | Notes / book | Assignment link |
|---|---|---|---|
| 1 | Projective geometry, transformations, DLT, RANSAC | notes p90–129, FoCV 38, 41 | A4 `homography_dlt`, `homography_ratio`; A1 SVD |
| 2 | The camera: P, K, resectioning, affine cameras, homography cases | notes p130–155, FoCV 39, 42 | A3 (f + altitude), A4 (K from EXIF) |
| 3 | Epipolar geometry, F, E, stereo, triangulation, SfM | notes p156–174, FoCV 40, 44 | A4 (RANSAC on F, `triangulate_points`, `umeyama`) |
| 4 | Optical flow: Horn–Schunck, Lucas–Kanade | notes p175–191, FoCV 46–48 | A3 (all tasks) |
| 5 | Generalisation, CNNs, transformers, ViT | slides 11, 24, 26 | A2 (convolution, NCC) |
| 6 | Representation learning, generative models, VAE | slides 30, 32, 33; UDL 17 | — |
| 7 | GANs and conditional generative models | slides 32, 34; UDL 15 | — |
| 8 | Diffusion models | slides 32–34; UDL 18 | — |
| 9 | Mock exam and formula sheet | — | — |

---

# Session 1: Projective geometry, transformations, DLT and RANSAC
*(notes p90–129)*

## 1.1 Homogeneous coordinates and the projective plane P²
- **Points.** The 2D point (x, y) becomes (x, y, 1). Any scaled version (kx, ky, k) with k ≠ 0 is the same point. To go back, (x, y, w) becomes (x/w, y/w).
- **Why bother.** Projection divides by depth, which is not linear. Homogeneous coordinates turn projection, translation and rotation into matrix multiplication.
- **P² = ℝ³ − {0}, modulo scale.** Each point of P² is a ray through the origin, and each line of P² is a plane through the origin.
- **Lines.** ax + by + c = 0 is written l = (a, b, c)ᵀ, and the point x lies on l when **lᵀx = 0**.
- **The line through two points** is l = x₁ × x₂.
- **Where two lines meet** is x = l₁ × l₂ (this point–line symmetry is called *duality*).
- **Ideal points** (x, y, 0) are points at infinity, i.e. directions.
  - Parallel lines meet at an ideal point. For example, x = 1 and x = 2 meet at (0, 1, 0).
  - All ideal points lie on the **line at infinity, l∞ = (0, 0, 1)ᵀ**.
- **The horizon** is the image of the ground plane's line at infinity.

**Worked example.** Find the line through (1, 2) and (3, 4).
- (1, 2, 1) × (3, 4, 1) = (−2, 2, −2), which is the same line as (1, −1, 1).
- So the line is x − y + 1 = 0.

**Proof that a homography preserves collinearity.**
- Suppose lᵀxᵢ = 0 for every point xᵢ.
- Define l′ = H⁻ᵀl. Then l′ᵀ(Hxᵢ) = lᵀH⁻¹Hxᵢ = lᵀxᵢ = 0.
- So **points map by H and lines map by H⁻ᵀ**.

## 1.2 The hierarchy of 2D transformations

| Class | Matrix | DOF | Min. pts | Invariants (each class also keeps everything listed below it) |
|---|---|---|---|---|
| Isometry | [εcosθ −sinθ tₓ; εsinθ cosθ t_y; 0 0 1] | 3 | 2 | length, angle, area |
| Similarity | [sR t; 0ᵀ 1] | 4 | 2 | angles, ratio of lengths, ratio of areas |
| Affine | [A t; 0ᵀ 1] | 6 | 3 | parallelism, ratio of areas, ratio of lengths on parallel lines, l∞ |
| Projective | [A t; vᵀ v] | 8 | 4 | collinearity, incidence, cross-ratio |

- **Isometry:** ε = +1 is a Euclidean (orientation-preserving) motion. ε = −1 is a reflection.
- **Affine decomposition: A = R(θ) R(−φ) D R(φ)**, with D = diag(λ₁, λ₂).
  - Read it as: rotate by φ, scale non-uniformly, rotate back, then rotate by θ.
  - The 4 parameters plus 2 for translation give 6 DOF.
- **Why affine keeps parallel lines parallel.** H·(x, y, 0)ᵀ = (A[x y]ᵀ, 0), so ideal points stay ideal.
- **Why projective does not.** A projective H has a last row of the form (h₇, h₈, h₉), which makes the third coordinate h₇x + h₈y ≠ 0. The meeting point becomes finite: a **vanishing point**.
- **Projective decomposition: H = H_S · H_A · H_P**
  - H_S is a similarity (4 DOF).
  - H_A = [K 0; 0ᵀ 1], with K upper triangular and det K = 1 (2 DOF).
  - H_P = [I 0; vᵀ v] (2 DOF). This is the part that moves l∞.
  - Total: 4 + 2 + 2 = 8.
- **Cross-ratio** of 4 collinear points: Cross = (|x₁x₂| · |x₃x₄|) / (|x₁x₃| · |x₂x₄|). It is invariant under any projective map.
  - Example: points at 0, 1, 2, 3 give (1·1) / (2·2) = 1/4.

## 1.3 P³
- **Points:** X = (X, Y, Z, 1)ᵀ.
- **Planes:** πᵀX = 0. The **plane at infinity** is π∞ = (0, 0, 0, 1)ᵀ.
- **DOF:** Euclidean 6, similarity 7, affine 12, projective 15 (16 entries minus 1 for scale).
- **Invariants:**
  - Projective maps keep intersection and tangency.
  - Affine maps also keep parallel planes and keep π∞ fixed.
  - Euclidean maps also keep volume.

## 1.4 DLT for a homography
**The equations.** From x′ᵢ × Hxᵢ = 0, each correspondence gives 3 rows, but only 2 are independent (the block has rank 2):
```
A_i = [  0ᵀ       −w′·xᵢᵀ    y′·xᵢᵀ ]
      [  w′·xᵢᵀ     0ᵀ      −x′·xᵢᵀ ]      (2×9)
```
- **n = 4:** A is 8×9, so h spans its 1-D null space exactly.
- **n > 4:** minimise ‖Ah‖ subject to ‖h‖ = 1. The solution is the **last column of V** in the SVD A = UΣVᵀ.

**Normalisation (Hartley)**
- Shift the centroid to the origin and scale so the **mean distance is √2**: T = [s 0 −s·cₓ; 0 s −s·c_y; 0 0 1].
- Undo it at the end: **H = T′⁻¹ H̃ T**.
- **Why:** in a 1000×1000 image, entries of A range from about 1 to 10⁶. The condition number σ₁/σ₉ is then huge and noise gets amplified. Normalised results are also independent of the coordinate frame.

**Your A4 `homography_dlt`** does normalise, build A, SVD, take `Vt[-1]`, then `inv(T2) @ H_norm @ T1`. It flags degeneracy when `S[7] < 1e-8·S[0]` (rank < 8, e.g. 3 collinear points).
- Note: it scales with `np.std`. **In the exam, write the mean-distance-√2 rule.**

## 1.5 RANSAC
**Steps**
1. Sample s points (2 for a line, 4 for H, 7 or 8 for F).
2. Fit a model to them.
3. Count inliers (error below t).
4. Repeat, keep the model with the most inliers, then **re-fit it on all of its inliers**.

**Threshold t** (noise with standard deviation σ):
- Line fitting: roughly 3σ.
- Homography: the error is a 2-D distance, so t² = 5.99σ² and **t = √5.99·σ** (χ² with 2 degrees of freedom, 95%).
- F: the error is a 1-D distance to a line, so t² = 3.84σ².

**Number of samples.** A sample is all inliers with probability wˢ (w = inlier fraction). All N samples fail with probability (1 − wˢ)ᴺ. Setting that to 1 − p:

> **N = log(1 − p) / log(1 − wˢ)**

| | w = 0.5, p = 0.99 | w = 0.7, p = 0.99 |
|---|---|---|
| s = 4 (H) | 72 | 17 |
| s = 7 (F, 7-point) | 588 | 54 |
| s = 8 (F, 8-point) | 1177 | 78 |

- **Adaptive N:** lower N whenever a larger consensus set is found.
- **Stopping early:** stop when the consensus set reaches **T = (1 − ε)·n**. Notes example: ε = 0.02, n = 1000, so T = 980.
- **Robust homography pipeline:** SIFT → ratio test → RANSAC(H) → re-fit on inliers.
- **A4 `homography_ratio`:** if H inliers / F inliers ≈ 1, the camera only rotated or the scene is planar. There is then no parallax, so the pair is useless for depth.

## Session 1 check (answers below)
1. Where do x + y = 2 and x − y = 0 meet?
2. Why does an affine map keep parallel lines parallel?
3. Why can a panorama from a rotating camera be stitched with a homography?
4. How many RANSAC iterations does H need with 50% outliers and p = 0.99?
5. How many DOF and minimum correspondences does an affine map have?
6. H = [[1, 0, 0], [0, 1, 0], [0.001, 0, 1]]. Where does (1, 0, 0) go?
7. Points at 0, 1, 3, 4 appear in an image at 0, 2, 5, x. Find x.
8. Why does one correspondence give only 2 independent DLT equations?
9. How many RANSAC iterations does F need with 30% outliers and p = 0.99, for the 8-point and the 7-point algorithms?
10. Which of these does a similarity preserve: angles, lengths, ratio of lengths, parallel lines?

**Answers**
1. (1, 1, −2) × (1, −1, 0) = (−2, −2, −2), which is the point **(1, 1)**.
2. The last row is (0, 0, 1), so ideal points map to ideal points.
3. x₂ = K R K⁻¹ x₁. Depth cancels, so there is no parallax.
4. **72**
5. **6 DOF, 3 points**
6. (1, 0, 0.001), which is the finite point **(1000, 0)**. So H is projective.
7. 1/9 = 2(x − 5) / (5(x − 2)), so **x = 80/13 ≈ 6.15**.
8. x′ᵀ(x′ × Hx) = 0 always, so the 3 rows are dependent.
9. **78** for 8-point, **54** for 7-point.
10. **Angles, ratio of lengths, parallel lines.** Not lengths.

---

# Session 2: The camera
*(notes p130–155)*

## 2.1 The pinhole model
**Step 1: basic pinhole.** A camera-frame point (X, Y, Z) goes to (fX/Z, fY/Z). In homogeneous form, x = diag(f, f, 1)[I | 0]X.

**Step 2: principal point offset.** K = [[f, 0, pₓ], [0, f, p_y], [0, 0, 1]].

**Step 3: camera pose.** With the camera centre at C̃ and rotation R, a world point in the camera frame is X_cam = R(X̃ − C̃). So:

> **P = K R [I | −C̃] = K [R | t],  with t = −R C̃**

**Step 4: CCD camera.** Non-square pixels: αₓ = f·mₓ and α_y = f·m_y.

**Step 5: finite projective camera.** Add skew s: K = [[αₓ, s, x₀], [0, α_y, y₀], [0, 0, 1]].

| Camera | K DOF | + R, C | Total |
|---|---|---|---|
| Basic pinhole | 3 | 6 | 9 |
| CCD | 4 | 6 | 10 |
| Finite projective | 5 | 6 | 11 |
| General projective (any rank-3 3×4) | — | — | 11 |

## 2.2 Reading information off P = [M | p₄], with M = KR
- **Finite vs infinite camera:** det M ≠ 0 means a finite camera. det M = 0 means the centre is at infinity (an affine camera).
- **Camera centre:** **PC = 0**, so C is the null space of P. For a finite camera, C̃ = −M⁻¹p₄.
- **Columns of P:**
  - p₁, p₂, p₃ are the vanishing points of the world X, Y, Z axes.
  - p₄ is the image of the world origin.
- **Rows of P** are planes:
  - P³ᵀ is the principal plane (through C, parallel to the image). Its points map to infinity.
  - P¹ and P² map to the image lines x = 0 and y = 0.
- **Principal axis:** v = det(M)·m³, which points in front of the camera.
- **Principal point:** x₀ = M m³.
- **Getting K, R and C back from P:** RQ-decompose M into KR, then take C from the null space. This is "Who shot the picture?" in your notes.
- **Vanishing point of a direction d:** v = M d = KRd. Going back, the direction in the camera frame is ∝ K⁻¹v.

**Worked example.** K = [[800, 0, 320], [0, 800, 240], [0, 0, 1]], R = I, C̃ = (0, 0, −5).
- The point X̃ = (1, 0.5, 5) is at (1, 0.5, 10) in the camera frame, so it projects to **(400, 280)**.
- The direction d = (1, 0, 1) has vanishing point Kd, which is **(1120, 240)**.

## 2.3 Camera resectioning (DLT for P)
**The equations.** From xᵢ × PXᵢ = 0:
```
A_i = [  0ᵀ       −wᵢXᵢᵀ    yᵢXᵢᵀ ]
      [  wᵢXᵢᵀ      0ᵀ     −xᵢXᵢᵀ ]     (2×12)
```
- P has 11 DOF at 2 equations per point, so you need 5.5 points, i.e. **n ≥ 6**.

**Normalisation (both spaces)**
- Image points: T, giving mean distance √2.
- 3D points: U, giving mean distance **√3**.
- Undo it: **P = T⁻¹ P̃ U**.

**Gold standard method:** start from the DLT result, then minimise the reprojection error Σ d(xᵢ, PXᵢ)² with Levenberg–Marquardt. The DLT minimises an algebraic error, which has no geometric meaning.

**Degenerate cases:** all points on a plane, or the points and C on a twisted cubic.

## 2.4 Affine cameras
- The last row of P is (0, 0, 0, 1), so det M = 0.
- Parallel lines stay parallel in the image.

| Model | DOF |
|---|---|
| Orthographic | 5 |
| Scaled orthographic | 6 |
| Weak perspective | 7 |
| General affine | 8 |

- **When valid:** when the scene's depth range ΔZ is much smaller than its average depth Z₀.

## 2.5 When two views are related by a homography

| Case | H |
|---|---|
| World plane Z = 0 | One view: x = [p₁ p₂ p₄](X, Y, 1)ᵀ. Two views: x₁ = H₁H₂⁻¹x₂ |
| Same centre | H = K₁R₁(K₂R₂)⁻¹ |
| Zoom only | H = K₁K₂⁻¹ (for zoom by k: K diag(k, k, 1) K⁻¹) |
| Rotation only (PTZ, panorama) | **H = K R K⁻¹**, a conjugate rotation. Up to scale its eigenvalues are {1, e^{±iθ}} |

## 2.6 Connection to A3 (the lunar lander)
- A downward-looking camera at altitude Z sees a ground shift ΔX as an image shift **Δx = f·ΔX/Z**.
- So the **ground speed = Z · (flow in px) / (f · Δt)**.
- A single camera cannot recover absolute scale, which is why one altitude measurement was given.
- **Descent** makes the image expand outward from a point. The expansion rate gives the descent speed relative to altitude.

## Session 2 check
1. P = [[1000, 0, 500, 0], [0, 1000, 400, 0], [0, 0, 1, 0]]. Find K, R and C, and project (2, 1, 10).
2. How do you get the camera centre from P? What does det M = 0 mean?
3. What do the columns of P mean geometrically?
4. What is the minimum number of correspondences for resectioning? Why can't the points be coplanar?
5. How many DOF does a weak perspective camera have, and when is it valid?
6. What is H for a camera that only rotates, and why is depth not needed?
7. f = 2000 px, altitude 100 m, flow 40 px per frame, Δt = 0.5 s. What is the speed?

**Answers**
1. K = [[1000, 0, 500], [0, 1000, 400], [0, 0, 1]], R = I, C = 0. The point projects to (7000, 5000, 10), which is **(700, 500)**.
2. C is the SVD null space of P. **det M = 0 means the centre is at infinity, so the camera is affine.**
3. Columns 1–3: vanishing points of the X, Y, Z axes. Column 4: the image of the origin.
4. **6 points** (11 DOF / 2 per point). If the points are coplanar, only a homography (3 columns) can be determined, so P is degenerate.
5. **7 DOF.** Valid when the depth range is much smaller than the distance.
6. **H = KRK⁻¹.** Rays from the shared centre only rotate.
7. 100·40/2000 = 2 m per frame, which is **4 m/s**.

---

# Session 3: Epipolar geometry, F, E, stereo, triangulation and SfM
*(notes p156–174; triangulation and SfM come from A4 and FoCV 44 because they are not in the notes)*

## 3.1 The geometry
- **Two cameras** have centres C and C′. The segment joining them is the **baseline**.
- **Epipoles:** **e = PC′** is the image of the other camera's centre in view 1, and **e′ = P′C** is the same in view 2.
- **Epipolar plane:** the plane containing C, C′ and X.
- **Epipolar lines:** the epipolar plane cuts each image in an epipolar line. The match of x must lie on l′ = Fx, so the stereo search becomes **1-D**.
- **All epipolar lines pass through the epipole.**

## 3.2 Deriving F
**Geometric derivation**
- Take any plane π not passing through the camera centres. It induces a homography x′ = H_π x.
- The epipolar line is l′ = e′ × x′ = [e′]ₓ H_π x.
- So **F = [e′]ₓ H_π**. Since [e′]ₓ has rank 2, **F has rank 2**.

**Algebraic derivation**
- Back-project x to the ray X(λ) = P⁺x + λC.
- Project two points of that ray into view 2: P′P⁺x, and P′C = e′.
- The line through them is l′ = e′ × P′P⁺x, so **F = [e′]ₓ P′P⁺**.

**Calibrated pair.** With P = K[I | 0] and P′ = K′[R | t]:

> **F = K′⁻ᵀ [t]ₓ R K⁻¹**

Here [t]ₓ is the cross-product matrix [[0, −t₃, t₂], [t₃, 0, −t₁], [−t₂, t₁, 0]].

## 3.3 Properties of F
- **Correspondence condition:** x′ᵀFx = 0.
- **Transpose:** if F belongs to (P, P′), then Fᵀ belongs to (P′, P).
- **Epipolar lines:** l′ = Fx and l = Fᵀx′.
- **Epipoles:** Fe = 0 and Fᵀe′ = 0. That is, e is the right null vector and e′ is the left null vector.
- **DOF:** 7 (9 entries − 1 for scale − 1 for det F = 0).
- **det F = 0.** F maps a point to a line, so it is not invertible.
- **Pure rotation (t = 0) gives F = 0, which is undefined.** Use H instead.

## 3.4 Estimating F
**Linear system.** Each match x′ᵀFx = 0 gives one row:
[x′x, x′y, x′, y′x, y′y, y′, x, y, 1] · f = 0.
Stack the rows into Af = 0 (n×9, n ≥ 8) and solve with SVD.

**Enforce rank 2.** Take F = U diag(σ₁, σ₂, σ₃) Vᵀ and replace it with U diag(σ₁, σ₂, **0**) Vᵀ. This is the closest rank-2 matrix in Frobenius norm. Without this step the epipolar lines would not all meet at one epipole.

**Normalised 8-point algorithm**
1. Normalise both images with T and T′ (mean distance √2).
2. Solve for F̃ and enforce rank 2.
3. Undo the normalisation: **F = T′ᵀ F̃ T**.

**7-point algorithm**
- A is 7×9, so its null space is 2-D, spanned by F₁ and F₂.
- Set F = αF₁ + (1 − α)F₂ and impose det F = 0. This is a cubic in α, giving **1 or 3 real solutions**.

**Automatic estimation:** SIFT → ratio test → RANSAC (7 or 8 points, error = distance to the epipolar line or Sampson distance, t² = 3.84σ²) → re-estimate on inliers → optionally use guided matching along epipolar lines.

## 3.5 The essential matrix
- **Normalised coordinates:** x̂ = K⁻¹x.
- **E = [t]ₓR** and x̂′ᵀ E x̂ = 0.
- **Derivation:** the vectors x̂′, t and Rx̂ all lie in the epipolar plane. So x̂′ is perpendicular to t × Rx̂.
- **Link to F:** **F = K′⁻ᵀ E K⁻¹** and **E = K′ᵀ F K**.
- **DOF:** 5 (R has 3, t has 3, minus 1 for scale).
- **A valid E has singular values (σ, σ, 0).** To enforce this, use E = U diag(1, 1, 0) Vᵀ.

**Decomposing E**
- Let W = [[0, −1, 0], [1, 0, 0], [0, 0, 1]].
- Then R = UWVᵀ or UWᵀVᵀ, and t = ±u₃ (the 3rd column of U).
- That gives **4 solutions**. Keep the one where the triangulated points are in front of both cameras (the *cheirality* check).
- The length of t cannot be recovered: **scale is unknown.**

**Worked example: pure sideways translation**
- Take t = (1, 0, 0) and R = I. Then E = [[0, 0, 0], [0, 0, −1], [0, 1, 0]].
- x̂′ᵀ E x̂ = y − y′ = 0, so **y = y′**. The epipolar lines are horizontal, which is exactly the rectified case.

## 3.6 Rectification and stereo depth
- **Rectification** applies homographies H and H′ that send both epipoles to (1, 0, 0), i.e. to infinity along x. Epipolar lines become matching horizontal rows.
- **Depth from parallel cameras.** x_l = fX/Z and x_r = f(X − B)/Z, so the disparity is d = x_l − x_r = fB/Z:

> **Z = f·B / d**

- **Depth error:** δZ ≈ (Z² / (fB))·δd. Error grows with the square of the distance, and a larger baseline helps.
- **Example:** f = 700 px, B = 0.1 m, d = 35 px gives Z = 2 m.

## 3.7 Triangulation
- Because of noise, the two back-projected rays do not quite meet.
- **Linear (DLT) method.** From x × PX = 0, each view gives two rows:
  - x·p³ᵀ − p¹ᵀ
  - y·p³ᵀ − p²ᵀ
- Stack them into a 4×4 matrix A. **X is the last singular vector of A**; then dehomogenise.
- **Other methods:** the midpoint of the closest points on the two rays, or the optimal method (minimise the reprojection error).
- **Accuracy:** poor when the angle between the rays is small, i.e. a small baseline relative to depth. This is why A4 rejects pairs with no parallax.

## 3.8 Structure from Motion: the A4 pipeline
1. Load the images and set K from EXIF (or f ≈ 1.2 × the long side).
2. RootSIFT features → visual words → **tf-idf** retrieval → top-k candidate pairs.
3. Match with the **ratio test** plus a mutual check, then verify geometrically with **RANSAC on F** (at least 20 inliers).
4. **Query expansion** (friends of friends), then **connected components** of the match graph.
5. For a pair with enough parallax (low H/F ratio): E = KᵀFK, decompose into R and t, and **triangulate**.
6. Add more cameras by resectioning (PnP) and triangulate new points.
7. **Bundle adjustment:** minimise Σᵢⱼ ‖xᵢⱼ − π(Pᵢ, Xⱼ)‖² over all cameras and points, using Levenberg–Marquardt.

**Ambiguity of the result**
- Uncalibrated cameras: the reconstruction is only correct up to a **4×4 projective H (15 DOF)**.
- Calibrated cameras: only up to a **similarity (7 DOF)**, i.e. unknown scale, rotation and translation.

**Umeyama alignment** (A4 `umeyama`): find s, R, t minimising Σ ‖yᵢ − (sRxᵢ + t)‖².
1. Find the centroids μₓ and μ_y and the covariance Σ = (1/n) Σ (y − μ_y)(x − μₓ)ᵀ.
2. Take the SVD Σ = UDVᵀ and set S = diag(1, 1, det(UVᵀ)). This S prevents a reflection.
3. **R = U S Vᵀ**, **s = tr(DS) / σₓ²**, **t = μ_y − sRμₓ**.

## Session 3 check
1. State 5 properties of F and its DOF. Why is det F = 0?
2. P = K[I | 0] and P′ = K[I | t] with t = (B, 0, 0). Find the epipoles and describe the epipolar lines.
3. Why must you enforce rank 2 after the linear 8-point solution, and how?
4. How many DOF does E have? What are its singular values? How many (R, t) solutions does it give, and how do you pick one?
5. f = 1000 px, B = 0.2 m, d = 50 px. What is Z, and what is the depth error for 1 px of disparity error?
6. When should you fit H instead of F? What does an H/F inlier ratio ≈ 1 tell you?
7. What does bundle adjustment minimise? What ambiguity remains in a calibrated reconstruction?

**Answers**
1. Any five of x′ᵀFx = 0, l′ = Fx, l = Fᵀx′, Fe = 0, Fᵀe′ = 0, and rank 2. **F has 7 DOF.** det F = 0 because F = [e′]ₓ(…) and [e′]ₓ has rank 2.
2. **Epipoles:** e = e′ = (1, 0, 0), at infinity. **Epipolar lines:** horizontal (y = y′).
3. Noise makes the linear solution full-rank, so the epipolar lines would not meet at a single point. Zero the smallest singular value.
4. **5 DOF**, with singular values (σ, σ, 0). There are **4 (R, t) solutions**; pick the one that passes the cheirality test.
5. **Z = 1000·0.2/50 = 4 m.** **δZ = 16/200 = 0.08 m.**
6. Fit H for a planar scene or a pure rotation. A ratio ≈ 1 means no parallax, so the pair cannot give depth.
7. **Bundle adjustment** minimises the total reprojection error. **Remaining ambiguity:** a similarity (scale, R, t), i.e. 7 DOF.

---

# Session 4: Optical flow
*(notes p175–191, A3)*

## 4.1 Motion field vs optical flow
- **Motion field:** the projection of the true 3D motion onto the image.
- **Optical flow:** the apparent motion of brightness patterns.
- **They can differ:**
  - A uniform sphere *rotating*: the motion field is non-zero but the flow is zero.
  - A still sphere under a *moving light*: the flow is non-zero but the motion field is zero.

## 4.2 Brightness constancy and the aperture problem
- **Assumption:** I(x + uδt, y + vδt, t + δt) = I(x, y, t). A first-order Taylor expansion gives:

> **Iₓu + I_y v + Iₜ = 0**

- **One equation, two unknowns.** Only the flow component along the gradient (the *normal flow*) can be recovered: uₙ = −Iₜ / ‖∇I‖.
- This is the **aperture problem**: looking at an edge through a small hole, you cannot see any motion along the edge.

## 4.3 Horn–Schunck (global method)
**Energy:** E = ∬ (Iₓu + I_y v + Iₜ)² + α²(‖∇u‖² + ‖∇v‖²) dx dy.
- The first term is the data term (brightness constancy).
- The second term is a smoothness term.

**Euler–Lagrange equations:**
- Iₓ(Iₓu + I_y v + Iₜ) = α²∇²u
- I_y(Iₓu + I_y v + Iₜ) = α²∇²v

**Iterative solution.** The discrete Laplacian is ∇²u ≈ κ(ū − u), where ū is the local average. That gives the update:

> **u^{k+1} = ū^k − Iₓ (Iₓū^k + I_y v̄^k + Iₜ) / (α² + Iₓ² + I_y²)**
> **v^{k+1} = v̄^k − I_y (Iₓū^k + I_y v̄^k + Iₜ) / (α² + Iₓ² + I_y²)**

**Details**
- Gradients are averaged first differences over a **2×2×2 cube** (x, y, t).
- Boundary conditions are needed at the image edges.

**Behaviour**
- Gives dense flow and fills in flat regions from their neighbours.
- **Over-smooths at motion boundaries.**
- Accuracy: best at corners, normal flow only at edges, propagated from neighbours in flat regions.

## 4.4 Lucas–Kanade (local method)
- **Assumption:** the flow is constant in a small window (3×3 gives 9 equations):
```
A = [Iₓ₁ I_y1; …; Iₓ₉ I_y9]   (9×2),   b = −[Iₜ₁ … Iₜ₉]ᵀ
(AᵀA) [u v]ᵀ = Aᵀb,   AᵀA = [[ΣIₓ², ΣIₓI_y], [ΣIₓI_y, ΣI_y²]]
```
- **AᵀA is the Harris structure tensor.**
  - Two large eigenvalues (corner): reliable flow.
  - One small eigenvalue (edge): the aperture problem.
  - Both small (flat region): unreliable.
- **Good points to track** are those with a large **min eigenvalue** of AᵀA. This is A3 Task 1.

## 4.5 Large motion (A3 Task 3)
- The Taylor expansion is only valid for motions up to about 1 px.
- **Iterative LK:** warp the image by the current estimate, re-solve, and repeat (Newton–Raphson style).
- **Coarse-to-fine:** use image pyramids.
- **SIFT-guided:** use a feature match as the initial guess, then refine with LK.

## 4.6 Horn–Schunck vs Lucas–Kanade

| | Horn–Schunck | Lucas–Kanade |
|---|---|---|
| Type | Global, dense | Local, sparse |
| Prior | Smoothness everywhere | Constant flow in a window |
| Flat regions | Filled in from neighbours | Fails (singular AᵀA) |
| Motion boundaries | Blurred | Good locally |
| Noise | Robust (averaged) | Sensitive in small windows |

## 4.7 From 3D motion to 2D flow (FoCV 47)
- **Forward translation** makes the flow radiate from the **focus of expansion (FOE)**, which marks the direction of travel.
- **Time to contact:** τ = Z / (−dZ/dt) = r / (dr/dt), where r is the distance of an image point from the FOE.
- **The lander:** while it descends, the image expands, and the rate of that expansion gives τ.

## Session 4 check
1. Write the brightness constancy equation and explain the aperture problem.
2. Over a window, AᵀA = [[10, 0], [0, 0.01]]. What is the image structure, and can LK track it?
3. Write the Horn–Schunck energy. What does α control?
4. Why does LK fail for 10 px motion, and what are three fixes?
5. Give an example where the motion field ≠ optical flow.

**Answers**
1. Iₓu + I_y v + Iₜ = 0. It is one equation in two unknowns, so only the component along the gradient can be recovered.
2. **An edge** (the gradient is along x only). LK can recover u only, so v is ill-posed.
3. Data term + α²·smoothness. A larger α gives smoother flow.
4. The Taylor approximation is only valid for motions up to about 1 px. **Fixes:** iterative warping, a pyramid, or a SIFT initialisation.
5. A rotating uniform sphere (motion field ≠ 0, flow = 0), or a moving light on a still scene (the reverse).

---

# Session 5: Generalisation, CNNs, transformers and ViT
*(slides 11, 24, 26; UDL 10, 12)*

## 5.1 Generalisation (quick review)
- **Splits:** train, validation, test. The validation set is for choosing hyperparameters, and the test set is used once.
- **Underfitting** means high bias. **Overfitting** means high variance: low training error but high test error.
- **Regularisers:** weight decay (L2), dropout, data augmentation, early stopping, more data.
- **Inductive bias** (the assumptions built into a model, e.g. locality) reduces how much data you need.

## 5.2 Convolutional layers

| Quantity | Formula |
|---|---|
| Output size | **O = ⌊(W − K + 2P) / S⌋ + 1** |
| Parameters | **(K·K·C_in + 1)·C_out** |
| Multiply-adds | H_out·W_out·K²·C_in·C_out |
| Receptive field | r = 1 + Σₗ (kₗ − 1)·Π_{i<l} sᵢ |

- **Pooling** has no parameters.
- **1×1 convolutions** mix channels only.
- **Weight sharing and locality** give far fewer parameters and **translation equivariance**. Pooling adds some **invariance**.
- **Fully connected vs convolutional:** a 224×224×3 input into 1000 FC units needs about 150 M parameters. A 3×3 convolution with 64 filters needs 1,792.
- **Deep-learning "convolution" is actually cross-correlation** (no kernel flip). This links to A2's filtering and NCC.

**Worked example**
- A 32×32×3 input, a 5×5 convolution with 16 filters, stride 1 and padding 2 gives **32×32×16**, with (75 + 1)·16 = **1216 parameters**.
- A 2×2 max-pool then gives 16×16×16.

**Architectures**
- **LeNet** and **AlexNet** (ReLU, dropout; the 2012 ImageNet competition, ILSVRC).
- **VGG:** two stacked 3×3 layers have the same receptive field as one 5×5 layer, with 18C² parameters instead of 25C², plus an extra non-linearity.
- **ResNet:** y = F(x) + x. The skip connection lets gradients flow, which makes 100+ layers trainable.
- **BatchNorm.**

## 5.3 Transformers
- **Attention:** Q = XW_Q, K = XW_K, V = XW_V:

> **Attention(Q, K, V) = softmax(QKᵀ / √d_k) V**

- **Why divide by √d_k:** the variance of a dot product grows with d_k. Large scores saturate the softmax, and the gradients vanish.
- **Multi-head attention:** h heads, each of dimension d/h. Concatenate them and apply W_O. This lets the model attend to different relationships at once.
- **Cost:** O(N²·d) time and O(N²) memory for the attention matrix.
- **Parameters per block:** attention 4d² + MLP 8d² (hidden size 4d) ≈ **12d²**.
- **Attention is permutation-equivariant**, so **positional encodings** are needed (sinusoidal or learned).
- **Block structure (pre-norm):** x → LN → MHA → +x → LN → MLP → +x.
- **Variants:** self-attention, cross-attention (Q from one sequence, K and V from another, e.g. text conditioning in diffusion), masked (causal) attention.

## 5.4 Vision Transformer (ViT)
1. Split the image into P×P patches, giving N = HW/P² tokens.
2. Flatten each patch (P²C values) and project it linearly to D dimensions. This is the same as a **convolution with kernel = stride = P**.
3. Prepend a learnable **[CLS]** token and add learned **position embeddings**.
4. Pass everything through a transformer encoder. An MLP head on [CLS] gives the class.

- **Example:** a 224×224 image with P = 16 gives 196 patches, plus [CLS] makes **197 tokens**. Each patch has 16·16·3 = 768 values.
- **ViT vs CNN:**
  - ViT has a **weaker inductive bias** (no built-in locality or translation equivariance), so it needs a lot of data (JFT-300M) or strong augmentation and distillation (DeiT).
  - ViT has a **global receptive field from layer 1** and scales better with data.
  - CNNs win when data is small.

## Session 5 check
1. A 64×64×3 input passes through a 3×3 convolution with 32 filters, stride 2 and padding 1. What is the output shape and the parameter count?
2. Why does a stack of two 3×3 convolutions beat one 5×5?
3. Why divide by √d_k? What is the complexity in N?
4. A ViT takes 384×384 images with P = 16. How many tokens? Why does a ViT need more data than a CNN?
5. Why does a transformer need positional encoding?

**Answers**
1. ⌊(64 − 3 + 2)/2⌋ + 1 = 32, so the output is **32×32×32**, with (27 + 1)·32 = **896 parameters**.
2. Same receptive field, with 18C² instead of 25C² parameters and an extra ReLU.
3. It stops the softmax saturating. **Complexity O(N²)**.
4. **24² = 576 patches, plus [CLS] = 577 tokens.** It has a weaker inductive bias.
5. Attention is permutation-equivariant, so it has no notion of order or position otherwise.

---

# Session 6: Representation learning, generative models and the VAE
*(slides 30, 32, 33; UDL 14, 17)*

## 6.1 Representation learning
- **Goal:** learn embeddings z = f(x) that make downstream tasks easy.
- **Evaluation:** a **linear probe** (a linear classifier trained on frozen features) or k-NN.
- **Autoencoders:** a bottleneck plus a reconstruction loss. *Denoising* autoencoders reconstruct a clean x from a corrupted one.
- **Pretext tasks:** colourisation, rotation prediction, inpainting, **masked autoencoders (MAE)** (mask about 75% of patches and reconstruct them).

**Contrastive learning (SimCLR)**
- Positive pairs are two augmentations of the same image. Everything else in the batch is a negative.
- **InfoNCE loss:**

> **L = −log [ exp(sim(zᵢ, zⱼ)/τ) / Σ_{k≠i} exp(sim(zᵢ, z_k)/τ) ]**

- It encourages **alignment** (positives close together) and **uniformity** (spread over the sphere).
- τ is the temperature.
- Large batches give more negatives.
- **The choice of augmentations decides which invariances are learned.**
- **CLIP** applies the same idea to (image, text) pairs.

## 6.2 Generative models: the big picture
- **Goal:** model p(x) so you can sample new x, and sometimes evaluate likelihood.
- **Maximum likelihood** is the same as minimising KL(p_data ‖ p_θ).

| Family | Likelihood | Sampling | Quality / diversity |
|---|---|---|---|
| Autoregressive, p(x) = Π p(xᵢ \| x_<i) | Exact | Slow (sequential) | Good |
| Normalising flows | Exact (change of variables) | Fast | Limited by invertibility |
| VAE | Lower bound (ELBO) | Fast | Blurry, good coverage |
| GAN | None | Fast | Sharp, mode collapse |
| Diffusion | Bound | Slow (many steps) | Sharp *and* diverse |

- **Metrics:** FID (lower is better: compares Inception-feature statistics of real and generated images) and Inception Score.

## 6.3 The VAE
- **Model:** z ~ N(0, I), with decoder p_θ(x | z).
- **Problem:** p(x) = ∫ p(x | z) p(z) dz is intractable.
- **Fix:** add an **encoder** q_φ(z | x) = N(μ_φ(x), diag σ²_φ(x)).

**ELBO derivation**

    log p(x) = E_q[log p(x|z)] − KL(q(z|x) ‖ p(z)) + KL(q(z|x) ‖ p(z|x))
                                                    └────────── ≥ 0 ─────────┘
    ⇒ log p(x) ≥ ELBO = E_q[log p_θ(x|z)] − KL(q_φ(z|x) ‖ N(0, I))
                           reconstruction        regulariser

- The ELBO can also be derived with Jensen's inequality.
- **Maximising the ELBO** raises the likelihood *and* makes q close to the true posterior.

**The KL term in closed form:** ½ Σ (μ² + σ² − log σ² − 1).

**Reparameterisation trick**
- Write **z = μ + σ ⊙ ε, with ε ~ N(0, I)**.
- The randomness is moved into ε, so gradients can flow through μ and σ. You cannot backpropagate through sampling directly.

**Behaviour**
- **Blurry samples:** a Gaussian likelihood is the same as an MSE loss, which averages over the possible outputs.
- **β-VAE** increases the weight on the KL term, which encourages disentangled latents.
- The latent space is smooth, so interpolation works well.

## Session 6 check
1. Derive the ELBO and name its two terms.
2. Why do we need the reparameterisation trick?
3. Why are VAE samples blurry?
4. Write InfoNCE. What is the role of the augmentations and of τ?
5. Compare VAE, GAN and diffusion on likelihood, sample quality and sampling speed.

**Answers**
1. See 6.3. The terms are **reconstruction** and **KL to the prior**.
2. Sampling is not differentiable. Writing z = μ + σε makes z a deterministic function of the parameters.
3. The MSE / Gaussian likelihood averages over the possible modes.
4. See 6.1. Augmentations define which invariances are learned. τ sets how sharply the loss focuses on hard negatives.
5. See the table in 6.2.

---

# Session 7: GANs and conditional generative models
*(slides 32, 34; UDL 15)*

## 7.1 The GAN objective
- The **generator** G(z) maps noise to images. The **discriminator** D(x) ∈ (0, 1) estimates whether x is real.

> **min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p(z)}[log(1 − D(G(z)))]**

**Optimal discriminator**
- For a fixed G, maximise pointwise. The function a·log D + b·log(1 − D) is maximised at D = a/(a + b), so:

> **D\*(x) = p_data(x) / (p_data(x) + p_g(x))**

**What the generator minimises**
- Plugging D\* back in gives **V = −log 4 + 2·JSD(p_data ‖ p_g)**.
- This is minimised exactly when p_g = p_data. Then D\* = ½ everywhere and V = −log 4.

**Training**
- Alternate k discriminator steps with 1 generator step.
- **Non-saturating loss:** early in training D rejects fake images easily, so log(1 − D(G(z))) is flat and gives little gradient. **The generator therefore maximises log D(G(z)) instead.**

## 7.2 Problems and fixes
- **Mode collapse:** G produces only a few modes that fool D.
- **Instability:** the training is a two-player game with no single loss that decreases.
- **Vanishing gradients:** when the supports of p_data and p_g don't overlap, the JSD is constant at log 2.
  - **WGAN** uses the Wasserstein (earth mover's) distance. The critic must be 1-Lipschitz, enforced with weight clipping or a **gradient penalty**.
- **DCGAN guidelines:** strided convolutions instead of pooling, BatchNorm, ReLU in G and LeakyReLU in D, no FC layers.
- **StyleGAN:** a mapping network plus style modulation. It produces very high-quality faces.
- **Pros and cons:** samples are sharp and fast to generate, but there is no encoder and no likelihood.

## 7.3 Conditional generative models (slide 34)
- **Why not plain regression?** When y given x is **multimodal** (e.g. colourisation: a car could be red or blue), L2 regression predicts the *mean*, which is brown and blurry. A conditional generative model samples one plausible output instead.
- **cGAN:** G(z, y) and D(x, y). Both networks see the condition.
- **pix2pix** (paired data):

> **G\* = arg min_G max_D L_cGAN(G, D) + λ·‖y − G(x)‖₁**

  - The generator is a **U-Net**, with skip connections that keep low-level detail.
  - The discriminator is a **PatchGAN**: it classifies N×N patches as real or fake, which enforces high-frequency realism. The L1 term handles the low frequencies.
- **CycleGAN** (unpaired data): uses G: X → Y and F: Y → X with a **cycle-consistency loss** ‖F(G(x)) − x‖₁ + ‖G(F(y)) − y‖₁, on top of two GAN losses.
- **Other conditional models:** class-conditional generation, text-to-image (diffusion with cross-attention), super-resolution, inpainting.

## Session 7 check
1. Write the GAN objective and derive D\*.
2. What does G minimise when D = D\*? When is it at its optimum?
3. What is mode collapse? Why use the non-saturating loss?
4. pix2pix: why the L1 loss plus the GAN loss? What is a PatchGAN?
5. What do you use when there is no paired data?

**Answers**
1. See 7.1. D\* = p_data / (p_data + p_g).
2. 2·JSD − log 4, which is optimal when p_g = p_data.
3. **Mode collapse:** G covers only a few modes. **Non-saturating loss:** log(1 − D(G(z))) gives weak gradients when D wins.
4. L1 gives the overall structure and low frequencies, the GAN loss gives sharpness. **A PatchGAN** judges realism patch by patch.
5. **CycleGAN**, with its cycle-consistency loss.

---

# Session 8: Diffusion models
*(slides 32–34; UDL 18; Lilian Weng's "What are Diffusion Models?")*

## 8.1 Forward process (fixed, no learning)
- Add noise step by step: q(xₜ | xₜ₋₁) = N(√(1 − βₜ)·xₜ₋₁, βₜI), with βₜ small and increasing.
- Define αₜ = 1 − βₜ and **ᾱₜ = Π_{s≤t} αₛ**.
- **Closed form:** you can jump straight to any step:

> **q(xₜ | x₀) = N(√ᾱₜ·x₀, (1 − ᾱₜ)I)  ⇔  xₜ = √ᾱₜ·x₀ + √(1 − ᾱₜ)·ε,  ε ~ N(0, I)**

- **Signal-to-noise ratio:** ᾱₜ / (1 − ᾱₜ). As ᾱ_T → 0, x_T ~ N(0, I).
- **Example:** ᾱₜ = 0.36 gives xₜ = 0.6x₀ + 0.8ε.

## 8.2 Reverse process (learned)
- p_θ(xₜ₋₁ | xₜ) = N(μ_θ(xₜ, t), σₜ²I).
- When βₜ is small, the true reverse step is also approximately Gaussian, which is what makes this work.
- **The network predicts the noise:**

> **μ_θ = (1/√αₜ)·(xₜ − (βₜ / √(1 − ᾱₜ))·ε_θ(xₜ, t))**

## 8.3 Training
- The ELBO, after simplification, becomes:

> **L = E_{t, x₀, ε} ‖ε − ε_θ(√ᾱₜx₀ + √(1 − ᾱₜ)ε, t)‖²**

- **Training step:**
  1. Sample x₀ from the data, t ~ U{1, …, T}, and ε ~ N(0, I).
  2. Form xₜ with the closed form.
  3. Take a gradient step on the MSE.
- Training is stable: it is plain regression, with no adversarial game.

## 8.4 Sampling (DDPM)
```
x_T ~ N(0, I)
for t = T … 1:
    z ~ N(0, I) if t > 1 else 0
    x_{t−1} = (1/√αₜ)(xₜ − (βₜ/√(1−ᾱₜ)) ε_θ(xₜ, t)) + σₜ z      (σₜ² = βₜ)
return x₀
```
- **Network:** a **U-Net** with a sinusoidal **time embedding** and attention layers.
- **Score interpretation:** ε_θ ≈ −√(1 − ᾱₜ)·∇ log q(xₜ). The network learns the score, and sampling is like noisy gradient ascent toward data.
- **Link to the VAE:** a diffusion model is like a hierarchical VAE whose encoder is fixed (not learned), with T latent layers each the same size as x.

## 8.5 Speed and conditioning
- **Slow sampling** (T ≈ 1000 network evaluations) has two main fixes:
  - **DDIM:** deterministic, with far fewer steps (around 50).
  - **Latent diffusion (Stable Diffusion):** run the diffusion in the latent space of a VAE, which is much cheaper.
- **Classifier guidance:** add ∇ log p(y | xₜ) from a classifier trained on noisy images.
- **Classifier-free guidance (CFG):** train with the condition randomly dropped (∅). At sampling time:

> **ε̃ = (1 + w)·ε_θ(xₜ, t, c) − w·ε_θ(xₜ, t, ∅)**

  A larger w follows the condition more closely but reduces diversity.
- **Text conditioning:** text tokens enter the U-Net through **cross-attention** (link to Session 5).

## 8.6 VAE vs GAN vs diffusion (likely to be examined)

| | VAE | GAN | Diffusion |
|---|---|---|---|
| Training | Stable (ELBO) | Unstable (min-max) | Stable (MSE on noise) |
| Samples | Blurry | Sharp | Sharp |
| Mode coverage | Good | Mode collapse | Good |
| Sampling speed | 1 pass | 1 pass | Many steps |
| Likelihood | Bound | None | Bound |
| Latent | Low-dim, learned | Low-dim noise | Same size as x, fixed encoder |

## Session 8 check
1. Derive xₜ in terms of x₀ (or state it) and explain ᾱₜ.
2. Write the simplified training loss and the training algorithm.
3. Why is sampling slow, and what are two fixes?
4. Explain classifier-free guidance and what w does.
5. Why is diffusion more stable to train than a GAN?

**Answers**
1. See 8.1. ᾱₜ is the product of the αₛ and controls the signal fraction.
2. See 8.3.
3. It is sequential over T steps. **Fixes:** DDIM, latent diffusion.
4. See 8.5. A larger w gives stronger condition adherence but less diversity.
5. It is a single regression objective, with no adversarial game.

---

# Session 9: Mock exam
*(about 90 minutes; answers at the end)*

1. Find the line through (0, 0) and (2, 2), and its intersection with y = 1, using homogeneous coordinates. *(4)*
2. Fill in the DOF and the minimum number of correspondences for: a 2D homography, a camera matrix P, F, E, and a 3D projective transformation. *(5)*
3. Prove that a projective transformation preserves collinearity. How do lines transform? *(4)*
4. Explain normalisation in the DLT and write the denormalisation for H and for P. *(4)*
5. RANSAC: derive N. Compute N for F with the 8-point algorithm, 40% outliers and p = 0.99. *(5)*
6. P = K[I | 0] and P′ = K[I | t] with t = (B, 0, 0). Find E, F, the epipoles and the epipolar lines. *(6)*
7. Stereo: f = 1000 px, B = 0.2 m, d = 50 px. Find Z and the error for 1 px of disparity error. *(3)*
8. Give the four cases where two views are related by a homography. *(4)*
9. LK: AᵀA = [[10, 0], [0, 0.01]]. Interpret it. Compare LK with Horn–Schunck. *(5)*
10. Convolution: a 64×64×3 input, a 3×3 kernel, 32 filters, stride 2, padding 1. Find the output shape and parameter count. *(3)*
11. ViT: a 384×384 image with P = 16. Find the number of tokens and the size of the attention matrix per head. Why √d_k? *(4)*
12. VAE: write the loss, and explain the reparameterisation trick and why samples are blurry. *(6)*
13. GAN: derive D\* and the generator's objective at D\*. What is mode collapse? *(6)*
14. Diffusion: write q(xₜ | x₀), the training loss and classifier-free guidance. *(6)*
15. pix2pix vs CycleGAN: when do you use each? What is the role of L1 and of the PatchGAN? *(4)*

### Answer key
1. (0, 0, 1) × (2, 2, 1) = (−2, 2, 0), which is the line y = x. Crossing with y = 1, i.e. (0, 1, −1), gives (−2, −2, −2), which is the point **(1, 1)**.
2. **Homography:** 8 DOF, 4 points. **P:** 11 DOF, 6 points. **F:** 7 DOF, 7 points (8 for the linear method). **E:** 5 DOF, 5 points. **3D projective:** 15 DOF, 5 points (each 3D point gives 3 equations).
3. lᵀx = 0 implies (H⁻ᵀl)ᵀ(Hx) = 0. **Lines map by l′ = H⁻ᵀl.**
4. Normalise with mean distance √2 in 2D (√3 in 3D) to fix the conditioning. **H = T′⁻¹H̃T** and **P = T⁻¹P̃U**.
5. N = log(1 − p) / log(1 − wˢ). Here w = 0.6, so 0.6⁸ ≈ 0.0168 and N = log(0.01) / log(0.9832) ≈ **272**.
6. **E = [t]ₓ = B[[0, 0, 0], [0, 0, −1], [0, 1, 0]]**, and **F = K⁻ᵀEK⁻¹**. **Epipoles:** (1, 0, 0) in both images. **Epipolar lines:** horizontal, y = y′.
7. **Z = 4 m.** **δZ = Z²δd / (fB) = 0.08 m.**
8. A planar scene; the same camera centre; zoom only; rotation only (H = KRK⁻¹).
9. **An edge** (aperture problem): only u can be recovered. Comparison: see the table in 4.6.
10. **32×32×32**, with **896 parameters**.
11. **577 tokens.** The attention matrix is **577×577** per head. √d_k keeps the softmax from saturating.
12. **Loss:** −ELBO = −E_q[log p(x | z)] + KL(q ‖ N(0, I)). **Reparameterisation:** z = μ + σε. **Blurry:** the MSE averages over the possible modes.
13. **D\* = p_d / (p_d + p_g).** **Generator objective:** 2·JSD − log 4. **Mode collapse:** G produces few modes.
14. **q(xₜ | x₀):** xₜ = √ᾱₜx₀ + √(1 − ᾱₜ)ε. **Loss:** ‖ε − ε_θ(xₜ, t)‖². **CFG:** ε̃ = (1 + w)ε_c − wε_∅.
15. **pix2pix:** paired data, L1 for structure, a PatchGAN for local realism. **CycleGAN:** unpaired data, with a cycle-consistency loss.

---

# One-page formula sheet

| Item | Formula / value |
|---|---|
| Line through two points / intersection of two lines | l = x₁ × x₂ / x = l₁ × l₂ |
| Line at infinity / plane at infinity | (0, 0, 1)ᵀ / (0, 0, 0, 1)ᵀ |
| 2D DOF | Euclidean 3, similarity 4, affine 6, projective 8 |
| 3D DOF | 6, 7, 12, 15 |
| Line transform | l′ = H⁻ᵀl |
| DLT solution | last column of V; normalise to √2 (2D), √3 (3D) |
| Denormalise | H = T′⁻¹H̃T; P = T⁻¹P̃U; F = T′ᵀF̃T |
| RANSAC iterations | N = log(1 − p) / log(1 − wˢ) |
| RANSAC thresholds | t² = 5.99σ² (H), 3.84σ² (F) |
| Camera matrix | P = KR[I \| −C̃]; 11 DOF; PC = 0; columns = vanishing points |
| Resectioning | n ≥ 6 |
| Affine cameras | orthographic 5, scaled orthographic 6, weak perspective 7, general affine 8 |
| Homography cases | rotation only: KRK⁻¹; plane: [p₁ p₂ p₄] |
| Fundamental matrix | x′ᵀFx = 0; F = [e′]ₓH_π = [e′]ₓP′P⁺ = K′⁻ᵀ[t]ₓRK⁻¹; rank 2; 7 DOF |
| Epipoles and lines | Fe = 0, Fᵀe′ = 0; l′ = Fx |
| Essential matrix | E = [t]ₓR = K′ᵀFK; 5 DOF; singular values (σ, σ, 0); 4 solutions |
| Stereo depth | Z = fB/d; δZ ≈ Z²δd / (fB) |
| Brightness constancy | Iₓu + I_y v + Iₜ = 0 |
| Lucas–Kanade | AᵀA = Harris matrix |
| Horn–Schunck | denominator α² + Iₓ² + I_y² |
| Convolution | O = ⌊(W − K + 2P)/S⌋ + 1; params (K²C_in + 1)C_out |
| Attention | softmax(QKᵀ/√d)V; O(N²d); about 12d² parameters per block |
| ViT tokens | N = HW/P² + 1 |
| VAE | ELBO = E[log p(x \| z)] − KL; z = μ + σε; KL = ½Σ(μ² + σ² − log σ² − 1) |
| GAN | D\* = p_d / (p_d + p_g); 2·JSD − log 4 |
| Diffusion | xₜ = √ᾱₜx₀ + √(1 − ᾱₜ)ε; loss ‖ε − ε_θ‖²; CFG (1 + w)ε_c − wε_∅ |
| InfoNCE | −log exp(s⁺/τ) / Σ exp(s/τ) |
