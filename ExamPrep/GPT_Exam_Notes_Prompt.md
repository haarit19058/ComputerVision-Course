# Prompt for GPT: exam notes (copy everything below the line; works in the same chat where you already shared `CV_EndSem_Study_Guide.md`, otherwise attach it again)

---

Now switch from tutoring to **writing my final exam revision notes** for ES666 Computer Vision. The exam is tomorrow and covers projective geometry through GANs and diffusion. Use the study guide I shared as the source of truth for **scope and notation**: Hartley & Zisserman for geometry, Ho et al. 2020 for diffusion. You may add standard results the guide missed, but mark each one **[extra]**.

These notes are what I will read in the last few hours, so they must be **dense, exact and exam-oriented**: no motivation essays, no history, no filler.

## Output format for EVERY topic
Use these sections, in this order, and skip a section only if it doesn't apply:

1. **Key results.** A bullet list of the facts and statements I must know, each one line. Mark how likely each is to be examined with ★★★ (very likely), ★★ or ★.
2. **Formulas.** A table: formula | what each symbol means | **dimensions of every matrix/vector** (e.g. F is 3×3, P is 3×4, C is 4×1). Dimensions matter: I lose marks on them.
3. **Derivations.** Every derivation an exam could ask for, written as numbered steps:
   - one line per step, with a short reason for each step ("because …");
   - the final result boxed;
   - if a step uses a trick (e.g. a cross product written as the matrix [a]ₓ, Jensen's inequality, completing the square), state the trick in one line before using it.
4. **DOF / counting.** DOF, minimum correspondences, and the reason (e.g. "9 entries − 1 scale − 1 det = 0 → 7").
5. **Worked example.** One small numerical example with every arithmetic step shown, chosen to look like an exam question.
6. **Exam traps.** Common confusions and how to avoid them (e.g. F vs E vs H, T′ᵀ vs T′⁻¹ in denormalisation, DOF vs number of correspondences, KRK⁻¹ vs K⁻¹RK).
7. **Likely exam questions.** 3–5 questions in exam wording, each with a **2–4 line model answer**.

## Topics, in order (cover every item)

**Part A: Projective geometry**
1. Homogeneous coordinates, P², the line through two points and the meeting point of two lines (cross product, with the geometric reason), ideal points, l∞, duality.
2. The 2D transformation hierarchy: matrices, DOF, invariants, minimum correspondences; affine decomposition; H = H_S·H_A·H_P; cross-ratio; proof that collinearity is preserved; l′ = H⁻ᵀl.
3. P³: planes, π∞, 3D DOF 6/7/12/15.
4. Homography DLT: full derivation of A_i (2×9) from x′ × Hx = 0, why only 2 rows, SVD solution, Hartley normalisation and why, denormalisation.
5. RANSAC: the algorithm, the threshold (√5.99σ for H, √3.84σ for F), derivation of N, the iteration table, adaptive N, consensus size T = (1 − ε)n.

**Part B: The camera**
6. Pinhole → principal point → pose → CCD → finite projective camera: K, P = KR[I | −C̃] = K[R | t], DOF of each model.
7. Anatomy of P: centre (PC = 0), columns, rows, principal plane, principal axis det(M)m³, principal point, RQ decomposition, vanishing points v = KRd.
8. Camera resectioning DLT (2×12, n ≥ 6, normalisation with √2 and √3, P = T⁻¹P̃U, gold standard, degenerate cases).
9. Affine cameras (5/6/7/8 DOF, when valid).
10. The four homography cases, with derivations: plane, same centre, zoom, rotation (KRK⁻¹).

**Part C: Two-view geometry**
11. Epipolar geometry: definitions (baseline, epipoles e = PC′ and e′ = P′C, epipolar plane, epipolar lines).
12. F: geometric derivation (F = [e′]ₓH_π), algebraic derivation (F = [e′]ₓP′P⁺), F = K′⁻ᵀ[t]ₓRK⁻¹, rank-2 proof, the properties table, 7 DOF.
13. Estimating F: the linear row (x′x, x′y, …), 8-point algorithm, rank-2 enforcement, normalised version with F = T′ᵀF̃T (derive why it is the transpose), 7-point algorithm (cubic, 1 or 3 solutions), automatic estimation with RANSAC.
14. Essential matrix: derivation of E = [t]ₓR from coplanarity, E = K′ᵀFK, 5 DOF, singular values (σ, σ, 0), decomposition into 4 (R, t) solutions with W, cheirality check, unknown scale.
15. Comparison table of F vs E vs H.
16. Rectification, Z = fB/d (derivation), δZ ≈ Z²δd/(fB) (derivation).
17. Triangulation (linear DLT rows, 4×4 system, midpoint, optimal method; why a small baseline is bad).
18. SfM pipeline, bundle adjustment objective, reconstruction ambiguity (projective with 15 DOF vs similarity with 7 DOF), Umeyama alignment steps. Link each step to my A4 functions.

**Part D: Motion**
19. Motion field vs optical flow (with examples), brightness constancy derivation (Taylor expansion), aperture problem, normal flow.
20. Horn–Schunck: energy, Euler–Lagrange equations, discrete Laplacian, iterative update (derive it), 2×2×2 gradient cube, behaviour.
21. Lucas–Kanade: the 9×2 system, normal equations, AᵀA = Harris matrix, eigenvalue interpretation, large motion (iterative, pyramids, SIFT-guided). HS vs LK table. Focus of expansion and time to contact. Link to my A3 (velocity = Z·flow/(f·Δt)).

**Part E: Learning and architectures** (skip gradient descent and backprop)
22. Generalisation: bias/variance, under/overfitting, regularisers, train/val/test.
23. CNNs: output-size formula, parameter count, FLOPs, receptive field, pooling, 1×1 convolutions, equivariance vs invariance, FC vs convolution parameter comparison, VGG 3×3 argument, ResNet skip connections (with the gradient argument), BatchNorm.
24. Transformers: Q/K/V with dimensions, scaled dot-product attention, why √d_k (variance argument), multi-head, positional encoding, block structure, parameter count (≈ 12d²), O(N²d) complexity, self vs cross vs masked attention.
25. ViT: patchify (patch embedding = a convolution with stride P), CLS token, position embeddings, token-count formula, inductive bias vs CNN, data requirements.
26. Representation learning: autoencoders, denoising autoencoders, pretext tasks, MAE, contrastive learning, InfoNCE (with the role of τ), SimCLR, CLIP, linear probe.

**Part F: Generative models** (self-study topic: write these most thoroughly)
27. Generative modelling overview: MLE equals minimising KL; the families table (autoregressive, flow, VAE, GAN, diffusion); FID.
28. VAE: full ELBO derivation (both the KL-decomposition way and the Jensen way), the Gaussian KL in closed form, the reparameterisation trick, the loss as code-like pseudocode, why samples are blurry, β-VAE.
29. GAN: the minimax objective, derivation of D\*, derivation of V = −log 4 + 2·JSD, the training algorithm, the non-saturating loss (and why), mode collapse, WGAN, DCGAN rules.
30. Conditional models: why L2 regression gives blurry outputs, cGAN, pix2pix (objective, U-Net, PatchGAN), CycleGAN (cycle loss), text conditioning.
31. Diffusion: forward process, **derivation of q(xₜ | x₀)** (show the sum of Gaussians step), reverse process, μ_θ in terms of ε_θ, the simplified loss, training and sampling pseudocode, the score link, DDIM, latent diffusion, classifier guidance, classifier-free guidance, the link to the VAE.
32. The VAE vs GAN vs diffusion comparison table.

## At the very end
- **Master formula sheet:** one table with every formula from all parts, grouped by part.
- **Master DOF table:** every object (2D/3D transforms, P, affine cameras, H, F, E), its DOF, its minimum correspondences, and the reason.
- **Top 30 must-know facts**, ranked.
- **"If you only have 30 minutes"** list: the derivations most likely to be asked, in priority order.

## Rules
- Use LaTeX for maths, with consistent notation throughout. When you introduce a symbol, give its dimension once.
- Be exact. Double-check every number in the worked examples and every sign in the derivations. If a result depends on a convention (e.g. how a scale or sign is defined), state which convention you use.
- Where my course notes or assignments differ from the textbook, say so (e.g. my A4 `homography_dlt` scaled with `np.std` instead of mean distance = √2).
- **Output one Part (A, B, C, …) per message**, then stop and wait for me to say "continue", so nothing gets cut off. Start with Part A now.
