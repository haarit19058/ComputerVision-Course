# Subjective Answers for task 2

## Question 01
What do you think PSNR is measuring ?? What could be a disadvantage of it ??


## Answer 01

Intuitive Meaning of PSNR ..
PSNR (Peak Signal-to-Noise Ratio) measures the logarithmic ratio between the maximum possible pixel value (MAX) and the Mean Squared Error (MSE) between two images. In this context, it acts as a mathematical metric to evaluate how "alike" two images are, specifically measuring the invisibility of the watermark (original photo vs. watermarked photo) and the recoverability of the QR code (original QR vs. extracted QR) as written in assignment paper.


Drawback of PSNR .. 
A major drawback of PSNR is that it relies on strict, pixel-by-pixel mathematical differences rather than human visual perception. Because it treats all errors equally, a tiny geometric shift (like moving the image by a single pixel) or a slight contrast adjustment can result in a drastically low PSNR, even if the images look completely identical to the human eye.


## Question 02

Briefly explain, what does the script task2.py is doing ??

## Answer 02

- It systematically loops through a list of different $\alpha$ values, embedding and then extracting the QR code for each one. During this sweep, it computes the PSNR for both the watermarked photograph (invisibility) and the recovered QR code (recoverability). 

- Finally, it plots these metrics on a graph, allowing you to visually identify  where the watermark is strong enough to be recovered without visibly ruining the original photograph.

- Effective it helps to visually find the optimal alpha.