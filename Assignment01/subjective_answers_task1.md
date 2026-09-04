# subjective answers

## Question 01
Why using top k singular values and vectors works for our tasks ?? Can we use less than top k singular values and vectors?? if yes what is the minimum value, if not why ??

## Answer 01
### Why using top k singular values and vector works for our tasks ?

Role of SVD in our case:
- By embedding the watermark into the singular values rather than specific pixels, the hidden information is mathematically distributed across the entire spatial domain of the photograph.

- Setting $k = \min(H, W)$ captures the maximum possible rank of the cover image. It ensures you are using all available non-zero singular values of the cover to perfectly match the $k$ singular values of your resized $k \times k$ QR code watermark

- Because the watermark is tied to the fundamental underlying frequencies (especially the larger singular values) rather than fragile pixel structures, it is highly resilient to image tampering like cropping, filtering, or JPEG compression.


### Can we use less than top k singular values and vectors ??
- Yes we can reconstruct using lesser singular values as lowest singular values mostly represents negligible details.
- This may lead to distortion in image based on top r < k singular values we choose.

### What is the minimum value ??
I have analysed the singular value decomposition of the watermark and base image and these are the findings that determine the minimum value:

* The watermark (QR code) requires an absolute minimum of 21 singular values to be fully reconstructed, as its data is highly concentrated and drops to exactly 0.0000 by the 22nd value.
* However, the base images possess a long tail of significant singular values extending well beyond 50, meaning we cannot truncate the operation to the watermark's minimum of 21 without discarding essential high-frequency details and severely distorting the cover photograph.

Run the watermark_singular_decomposition.py to see these values.


## Question 02
Assume grayscale cover image of size H x W, watermark of size lxl where l < min H,W and number of singular values for watermark is less than that of the cover image. To which singular values of cover image will you add the singular values of the water mark and why ?? Options are 


## Answer 02
a) Top l singular values of cover image

The top singular values contain the most significant energy and core structural information of the photograph. By embedding the watermark here, it becomes tightly bound to the fundamental features of the image, making it highly resilient to attacks.

