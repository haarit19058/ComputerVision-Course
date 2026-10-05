### 2.7 Why NCC and not sum-of-squared-differences?

Let $I$ be an image patch and $T$ be the template. If we change the image's brightness and contrast, the new image patch is $I' = aI + b$ (where $a > 0$).

The average of the new patch is $\bar{I'} = a\bar{I} + b$.
If we subtract the average, the brightness change ($b$) drops out:

$$I' - \bar{I'} = (aI + b) - (a\bar{I} + b) = a(I - \bar{I})$$

When we plug this into the NCC formula:

$$\mathrm{NCC}(I', T) = \frac{\sum a(I-\bar{I})(T-\bar{T})}{\sqrt{\sum a^2(I-\bar{I})^2}\sqrt{\sum (T-\bar{T})^2}}$$

Since $a > 0$, $\sqrt{a^2} = a$. We can pull $a$ out of the top and bottom:

$$\mathrm{NCC}(I', T) = \frac{a \sum (I-\bar{I})(T-\bar{T})}{a \sqrt{\sum (I-\bar{I})^2}\sqrt{\sum (T-\bar{T})^2}} = \mathrm{NCC}(I, T)$$

The $a$ terms cancel out perfectly. This proves NCC ignores overall contrast ($a$) and brightness ($b$) changes.

**SSD Behavior:**
Sum of Squared Differences just squares the differences: $\mathrm{SSD}(I, T) = \sum (I - T)^2$. If we change brightness or contrast, it becomes $\sum (aI + b - T)^2$. This means SSD changes a lot if $a$ or $b$ changes. If Waldo is hiding in a shadow, SSD will give a huge error and fail to find him.



### 2.8 Grayscale against colour, measured

| Scene | RGB Score | RGB IoU | Gray Score | Gray IoU |
| --- | --- | --- | --- | --- |
| `scene_easy` | 0.894 | 1.000 | 0.862 | 1.000 |
| `scene_medium` | 0.864 | 1.000 | 0.844 | 1.000 |
| `scene_rotated` | 0.838 | 0.000 | 0.797 | 0.000 |
| `scene_hard` | 0.823 | 1.000 | 0.786 | 1.000 |

**Which scenes still work:**
Grayscale still finds Waldo in `scene_easy`, `scene_medium`, and `scene_hard` (IoU = 1.000). `scene_rotated` fails in both color and grayscale because a basic sliding window can't handle rotated shapes.

**Score changes:**
The top scores drop by about 0.02 to 0.04 in grayscale. Waldo stands out mainly because of his bright red and white striped shirt. In grayscale, the red stripes look exactly like dark green leaves or brown buildings. Without color, the computer has a harder time separating Waldo from the background clutter.

**Lines changed:**

```python
    # Task 2.8: Grayscale Benchmark
    template_gray = to_grayscale(template)
    for name in scene_names:
        scene_gray = load_image_as_grayscale("imgs/{}.png".format(name))
        found_gray = detect_multiscale(scene_gray, template_gray)


```



### 2.9 The mask you were given and never used

* **Unmasked Best Score:** 0.8233 at Scale 1.50
* **Masked Best Score:** 0.9387 at Scale 1.50

**Why the score changed:**
The score goes up a lot with the mask. A normal template is a rectangle, so it grabs extra background pixels around Waldo's body and head. In `scene_hard`, Waldo is partly covered up and surrounded by background junk that isn't in the template image.

The unmasked version forces the math to include those mismatched corners, which hurts the score. The masked version only looks at Waldo's actual pixels. It ignores the background, giving a much better match.


### 2.10 Scores are not comparable across scales

| Scale | Pixels | Max Score | Std Dev |
| --- | --- | --- | --- |
| 0.60 | 1734 | 0.6521 | 0.1389 |
| 0.72 | 2400 | 0.6954 | 0.1338 |
| 0.85 | 3456 | 0.6555 | 0.1281 |
| 1.00 | 4704 | 0.7261 | 0.1162 |
| 1.15 | 6144 | 0.7170 | 0.1188 |
| 1.35 | 8664 | 0.4974 | 0.1116 |
| **1.50** | **10584** | **0.8233** | **0.1075** |
| 1.70 | 13680 | 0.3893 | 0.1011 |

**Would dividing each map by its own standard deviation make the scales comparable?**

No. The data shows that as the scale (and pixel count) gets bigger, the standard deviation goes down. Larger templates smooth out random background noise better. Smaller templates bounce around more over random background shapes, so their standard deviation is higher.

If you divide by standard deviation, you punish smaller templates (by dividing by a larger number) and boost larger ones. Standard deviation just measures how messy the background is; it doesn't measure how strong the actual Waldo match is. Because of this, dividing by standard deviation won't make the scores safe to compare across different sizes.