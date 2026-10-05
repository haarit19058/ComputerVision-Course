### 4.6 Why the median fails

The median only works if more than half of your data is correct. In image matching, over 85% of the matches are usually wrong. The few correct matches clump tightly directly over Waldo, while the wrong matches are scattered randomly across the entire image. If you take the median of all these points, the huge number of random points outweighs the good ones. The math just finds the physical middle of all that random noise—usually the center of the image—completely missing Waldo.

### 4.7 The abnormality: each method fails where the other does not

**(a) Why Task 2 fails on `scene_rotated`:**
Task 2 only resizes the template; it never tilts it. Because Waldo is leaning in this scene, the upright template pixels will never line up with him, no matter what size we make the template. The match score stays low.

**(b) Why Task 4 works on rotated scenes (and why `orientations=None` breaks it):**
Task 4 calculates the angle of each feature and finds the difference between the scene's angle and the template's angle ($\Delta\theta$). It uses this difference to physically rotate the voting direction. If we pass `orientations=None`, the code assumes everything is perfectly upright ($\Delta\theta = 0$). Without rotating the votes, Task 4 fails on leaning objects just like Task 2.

**(c) Exact pixels vs. fuzzy guesses:**
Task 2 slides over the image pixel by pixel. When it finds the best match, it gives you that exact pixel coordinate. Task 4 works by guessing the center from a distance, using the angles and sizes of a few small patches. Because these angles and distances are just estimates, the predicted center is a bit fuzzy. The final box is an average of these fuzzy guesses, making it much looser than Task 2's exact pixel lock.

### 4.8 The check that every tutorial recommends

When we add a strict two-way check, the number of matches and votes drops massively. A one-to-one rule is bad for this specific task because Waldo's shirt has a repeating pattern of identical red and white stripes. One stripe on the template might legitimately match three different stripes in the scene. A mutual check forces the code to pick just one and throw the others away. By doing this, we delete perfectly good clues and starve the system of the votes it needs to build a strong cluster.

### 4.9 Agreement without an answer key

Task 2 and Task 4 find things in completely different ways. Task 2 looks at raw pixel values in a strict grid, while Task 4 looks at the angles and edges of small, scattered patches. Because they search so differently, it is highly unlikely they will both make the exact same mistake in the exact same spot. If both point to the same box, you can be very confident Waldo is there. If they disagree, something in the image confused at least one of them.

