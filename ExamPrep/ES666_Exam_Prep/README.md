# ES666 Computer Vision: End-Semester Exam Prep

The ChatGPT tutoring session (`ES666 Exam Preparation.md`), split into one file per session, reformatted, and illustrated with a diagram for each worked calculation.

Use it with [../CV_EndSem_Study_Guide.md](../CV_EndSem_Study_Guide.md), which defines the scope and notation.

## How the material is organised

| Part | What it is | Use it for |
|---|---|---|
| **Tutoring sessions** (01–08) | Teaching order: intuition → derivation → worked numbers → check questions | Learning or re-learning a topic |
| **Revision notes** (09–11) | Geometry only. Each topic follows the same 7 headings: key results, formulas, derivations, DOF, worked example, exam traps, likely questions | Final pass on the geometry half |
| **Exam 2 solutions** (12) | Full answers to a past paper, plus a last-minute formula table | Practice under exam conditions |

## Files

### Tutoring sessions

| # | File | Topics |
|---|---|---|
| 01 | [Session 2: The Camera](01_Session2_Camera.md) | Pinhole model, K / R / t, P = KR[I \| −C̃], anatomy of P, resectioning DLT, affine cameras, the 4 homography cases, A3 lunar-lander velocity |
| 02 | [Recap quiz + Session 3: Epipolar Geometry](02_Session3_Epipolar_Geometry.md) | 6-question recap quiz on Sessions 1–2 (no answers given), then F (two derivations), 8-point algorithm, E and its decomposition, rectification, stereo depth, triangulation, SfM / A4 |
| 03 | [Session 3, second pass](03_Session3_Epipolar_Geometry_Second_Pass.md) | Shorter re-explanation of the same material, with checkpoint questions |
| 04 | [Session 4: Optical Flow](04_Session4_Optical_Flow.md) | Brightness constancy, aperture problem, Horn–Schunck (energy, Euler–Lagrange, update, 2×2×2 cube), Lucas–Kanade, pyramids, focus of expansion, A3 |
| 05 | [Session 5: CNNs, Transformers, ViT](05_Session5_CNNs_Transformers_ViT.md) | Generalisation, convolution arithmetic, pooling, receptive field, architectures, attention, ViT |
| 06 | [Session 6: Representation Learning and VAE](06_Session6_Representation_Learning_VAE.md) | Autoencoders, MAE, SimCLR / InfoNCE, generative-model families, ELBO, KL, reparameterisation |
| 07 | [Session 7: GANs and Conditional Models](07_Session7_GANs_Conditional_Models.md) | Minimax game, optimal D, JSD, training problems, WGAN, cGAN, pix2pix, CycleGAN |
| 08 | [Session 8: Diffusion](08_Session8_Diffusion.md) | Forward process, closed form, reverse mean, training loss, sampling, U-Net, DDIM, latent diffusion, guidance |

### Revision notes (geometry)

| # | File | Topics |
|---|---|---|
| 09 | [Part A: Projective Geometry](09_Revision_Notes_A_Projective_Geometry.md) | A1 P² and homogeneous coordinates · A2 transformation hierarchy · A3 P³ · A4 homography DLT and Hartley normalisation · A5 RANSAC |
| 10 | [Part B: The Camera](10_Revision_Notes_B_Camera.md) | B6 pinhole · B7 anatomy of P · B8 resectioning · B9 affine cameras · B10 the 4 homography cases |
| 11 | [Part C: Two-View Geometry](11_Revision_Notes_C_Two_View_Geometry.md) | C11 epipolar definitions · C12 F · C13 estimating F · C14 E · C15 F vs E vs H · C16 rectification and stereo · C17 triangulation · C18 SfM, BA, Umeyama |

### Past paper

| # | File | Topics |
|---|---|---|
| 12 | [Examination 2: Worked Solutions](12_Exam2_Solutions.md) | Parameter counts (FC / conv / attention) · tanh vs sigmoid · Leaky ReLU · F ↔ E and DOF · encoder–decoder imputation · contrastive learning · Horn–Schunck and Lucas–Kanade steps · formula table |

## Figures for the calculations

All figures are in [figures/](figures/). Each one is placed next to the topic or calculation it illustrates. Wherever a worked example exists, the figure uses its numbers.

| Topic | Figures |
|---|---|
| Projective geometry | [line intersection](figures/a1_line_intersection.png) · [transformation hierarchy](figures/a2_transform_hierarchy.png) · [Hartley normalisation](figures/a4_hartley_normalisation.png) · [RANSAC iterations](figures/a5_ransac_iterations.png) |
| Camera | [pinhole projection](figures/b6_pinhole.png) · [world → camera frame](figures/b6_world_camera_frames.png) · [vanishing point](figures/b7_vanishing_point.png) · [resectioning](figures/b8_resectioning.png) · [perspective vs affine](figures/b9_perspective_vs_affine.png) · [homography cases](figures/b10_homography_cases.png) · [lunar-lander velocity](figures/b_lunar_velocity.png) |
| Two-view | [epipolar geometry](figures/c11_epipolar_geometry.png) · [F from a plane homography](figures/c12_F_from_plane_homography.png) · [sideways translation, y′ = y](figures/c14_sideways_translation.png) · [rectification](figures/c16_rectification.png) · [stereo depth and error](figures/c16_stereo_depth.png) · [triangulation](figures/c17_triangulation.png) · [SfM pipeline](figures/c18_sfm_pipeline.png) · [Umeyama](figures/c18_umeyama.png) |
| Optical flow | [flow field](figures/of_flow_field.png) · [aperture problem](figures/of_aperture_problem.png) · [Horn–Schunck update](figures/of_hs_update.png) · [2×2×2 gradient cube](figures/of_gradient_cube.png) · [AᵀA eigenvalues](figures/of_lk_eigenvalues.png) · [Lucas–Kanade solve](figures/of_lk_solution.png) · [pyramid](figures/of_pyramid.png) · [focus of expansion](figures/of_focus_of_expansion.png) |
| CNN / transformer | [under/overfitting](figures/ml_fit_regimes.png) · [convolution arithmetic](figures/cnn_conv_example.png) · [max pooling](figures/cnn_maxpool.png) · [receptive field](figures/cnn_receptive_field.png) · [attention example](figures/tf_attention_example.png) · [ViT patches](figures/tf_vit_patches.png) |
| Representation learning / VAE | [autoencoder](figures/rl_autoencoder.png) · [SimCLR](figures/rl_simclr.png) · [InfoNCE](figures/rl_infonce.png) · [VAE](figures/vae_architecture.png) · [KL example](figures/vae_kl_example.png) · [reparameterisation](figures/vae_reparameterisation.png) |
| GANs | [GAN](figures/gan_architecture.png) · [optimal discriminator](figures/gan_optimal_discriminator.png) · [U-Net and PatchGAN](figures/cgan_unet_patchgan.png) · [CycleGAN](figures/cgan_cyclegan.png) |
| Diffusion | [forward process](figures/diff_forward_process.png) · [reverse step](figures/diff_reverse_step.png) · [latent diffusion](figures/diff_latent_diffusion.png) |
| Exam 2 | [sigmoid vs tanh](figures/exam_sigmoid_tanh.png) · [ReLU vs Leaky ReLU](figures/exam_relu_leaky.png) |

## Changes from the original chat

- **Correction: InfoNCE example (file 06).** The chat gave L ≈ 0.33. The correct value is −log(6.05 / 8.76) ≈ **0.37**. The file marks the correction.
- **Correction: tanh sketch (file 12).** The chat's tanh sketch was drawn decreasing. It is replaced by a plotted figure.
- I re-checked every other worked example used in a figure, and the numbers match.
- ChatGPT's interactive quiz widgets are now plain numbered question lists, with hints.
- The image placeholders were ChatGPT stock-image searches. They are replaced by the figures above where a diagram helps, and removed otherwise.
- The citations to study-guide line numbers are removed. They pointed to an older version of the guide.
- Your prompts to ChatGPT are not repeated here. They are already in [../GPT_Tutor_Prompt.md](../GPT_Tutor_Prompt.md) and [../GPT_Exam_Notes_Prompt.md](../GPT_Exam_Notes_Prompt.md).
- The past-paper photo was not in the export. The six questions appear as headings in file 12.

Math uses `$…$` and `$$…$$`, which render in GitHub and in the VS Code Markdown preview.
