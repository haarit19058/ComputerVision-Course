### 1.7 Predict first, then measure

**Prediction:** I thought a 9x9 box filter would improve both PSNR and SSIM. Since the filter averages out the noise, I expected the result to look more like the original image, which should improve both scores.

**Measurement (from terminal):**

* **PSNR in:** 24.53
* **PSNR out:** 21.55
* **SSIM in:** 0.4241
* **SSIM out:** 0.7225

**Conclusion:** I was right about SSIM (it went up from 0.42 to 0.72), but wrong about PSNR (it dropped from 24.53 to 21.55).

### 1.8 The two metrics disagree

PSNR looks at exact pixel differences. A 9x9 box filter causes a lot of blur. Even though it removes noise, it also destroys sharp edges and fine details. Because the blurred edge pixels don't match the sharp original pixels, the error goes up and the PSNR score goes down.

SSIM looks at overall structure and patterns, similar to how human eyes work. By removing the random noise, the filter brings back the main shapes in the image. Even with softer edges, the basic structure matches the original well, so the SSIM score goes up.

For template matching, I would trust **SSIM**. Template matching looks for patterns, shapes, and contrast, not exact pixel matches.

### 1.9 What colour costs you

**The lines added/changed to run this:**

```python
clean_gray = to_grayscale(clean)
noisy_gray = add_gaussian_noise(clean_gray, sigma=0.06)
salted_gray = add_salt_pepper_noise(clean_gray, amount=0.04)

box_on_gaussian_g = convolve2d(noisy_gray, box_9x9)
gauss_on_gaussian_g = gaussian_blur(noisy_gray, sigma=2.0)
median_on_salt_g = median_filter(salted_gray, size=5)
gauss_on_salt_g = gaussian_blur(salted_gray, sigma=1.5)


```

**Colour Table:**

```text
Colour images, 3 channels
filter                             PSNR in  PSNR out   SSIM in  SSIM out
box 9x9 on gaussian                  24.53     21.55    0.4241    0.7225
gaussian s=2 on gaussian             24.53     23.45    0.4241    0.8078
median 5x5 on salt+pepper            18.82     24.79    0.4084    0.9129
gaussian s=1.5 on salt+pepper        18.82     24.12    0.4084    0.7580


```

**Grayscale Table:**

```text
Grayscale images, 1 channel
filter                             PSNR in  PSNR out   SSIM in  SSIM out
box 9x9 on gaussian                  24.48     22.04    0.4040    0.7411
gaussian s=2 on gaussian             24.48     23.93    0.4040    0.8190
median 5x5 on salt+pepper            18.82     25.12    0.3971    0.9129
gaussian s=1.5 on salt+pepper        18.82     24.52    0.3971    0.7607


```

**Explanation:**
The scores don't change much between the two tables because the noise is spread out the same way in both color and grayscale. Grayscale is just a mix of the red, green, and blue channels, so the ratio of signal to noise stays roughly the same.

That doesn't mean we should ignore color. These math formulas don't know what the image actually shows. For example, in Task 2, we need to find Waldo based on his red and white stripes. In grayscale, red and dark green can look exactly the same. If we remove color, we lose the best way to spot him in a crowded picture.

### 1.10 Padding

If we use zero padding (`pad_mode="constant"`) with a large blur filter, a dark halo appears around the edges of the image. This happens because zero padding pretends everything outside the image is completely black. When the filter hits the edge, it mixes the image pixels with this fake black border, making the edges dark.

Reflect padding is better before template matching because it doesn't create fake edges. Zero padding creates a harsh drop-off at the borders. A template matcher or edge detector might get confused and think this dark border is a real feature in the image. Reflect padding just mirrors the image data at the edges, keeping things natural and avoiding false matches.