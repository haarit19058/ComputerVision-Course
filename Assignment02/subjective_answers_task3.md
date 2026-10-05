### 3.6 Corner or blob?

The third number, $\sigma$ (sigma), is the size of the blob. It comes from the blur level where a point stands out the most.

Task 4 needs to match patches even if Waldo is zoomed in or out. Harris corners only give an (x, y) location, not a size. If we used them, we wouldn't know how big of a window to look at. A fixed-size window around a corner in a zoomed-in image would capture completely different details than the same window in a zoomed-out image, causing the matching to fail.

### 3.7 The ladder is a trade



Here are the results of running `dog_keypoints` on the template with three different settings:

1. **Default** `(sigma0=0.8, n_scales=8, k=1.25)`: 29 keypoints. $\sigma$ range: [1.0, 1.95].
2. **Finer/Lower** `(sigma0=0.5, n_scales=10, k=1.2)`: 41 keypoints. $\sigma$ range: [0.72, 2.15].
3. **Coarser/Higher** `(sigma0=1.5, n_scales=5, k=1.5)`: 2 keypoints. $\sigma$ range: [2.25, 5.06].

Terminal Dump from task3.py
```txt
--- 3.7 Ladder Trade Test ---
sigma0=0.8, n_scales=8, k=1.25 -> Keypoints: 29, σ range: [1.00, 1.95]
sigma0=0.5, n_scales=10, k=1.2 -> Keypoints: 41, σ range: [0.72, 2.15]
sigma0=1.5, n_scales=5, k=1.5 -> Keypoints: 2, σ range: [2.25, 5.06]
```

**What breaks:**

* **At the small end:** The detector picks up tiny image grain and noise. It gets flooded with fake points that aren't real features.
* **At the big end:** The heavy blur washes out the image. The details melt together into a giant blob, so the detector finds lesser points.

### 3.8 More keypoints, fewer matches

**The Loop:** Written in Task 4 python file.

```python
scene_easy = to_grayscale(load_image_as_rgb("imgs/scene_easy.png"))

for upsample in [1.5, 2.0, 3.0, 4.0]:
    res = find_waldo(scene_easy, template, upsample=upsample)
    n_kp = len(res["kp_template"])
    n_matches = res["n_matches"]
    print(f"upsample={upsample:.1f} | Template Keypoints: {n_kp} | Matches: {n_matches}")

```
Result : 
```txt
upsample=1.5 | Template Keypoints: 67 | Matches: 243
upsample=2.0 | Template Keypoints: 82 | Matches: 265
upsample=3.0 | Template Keypoints: 119 | Matches: 218
upsample=4.0 | Template Keypoints: 98 | Matches: 208
```


**The Trend:**
Making the image bigger (upsampling) creates a lot more keypoints on the template, but the final number of matches actually drops.

**Explanation:**
Waldo’s shirt has repeating red and white stripes. A bigger template means the detector finds points on several identical stripes.

Because the stripes look exactly the same, the data for these points is identical. The matching rule in Task 4 requires the best match to be much better than the second-best match. But since the stripes are repeating, the best match and the second-best match look exactly the same. The test flags them as ambiguous and throws them out, even though they were actually looking at Waldo.

### 3.9 Why does this task throw the colour away?

In grayscale, an edge gives one clear direction going from dark to light.

If you try to find edges using RGB color, each color channel creates its own direction. Imagine an edge going from solid red to solid green. The overall brightness stays the same. But the red channel's direction points one way (because red is fading out), and the green channel points the exact opposite way (because green is fading in). If you mix them, the opposing directions cancel each other out. The program would completely miss a clear edge. Converting to grayscale makes sure all edges have just one clear direction.