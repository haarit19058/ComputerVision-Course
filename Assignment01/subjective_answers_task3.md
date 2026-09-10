# Subjective answers for task 3


## For each image and each stddev write the minimum value of alpha for which a qr scanner is able to extract your roll number ?? In which case the qr scanner failed to extract the roll no ??



| stddev | alpha | is qr readable |
| --- | --- | --- |
| 0.01 | 0.01 | yes |
| 0.01 | 0.05 | yes |
| 0.01 | 0.10 | yes |
| 0.01 | 0.15 | yes |
| 0.01 | 0.20 | yes |
| 0.05 | 0.01 | no |
| 0.05 | 0.05 | no |
| 0.05 | 0.10 | no |
| 0.05 | 0.15 | no |
| 0.05 | 0.20 | no |
| 0.10 | 0.01 | no |
| 0.10 | 0.05 | no |
| 0.10 | 0.10 | no |
| 0.10 | 0.15 | no |
| 0.10 | 0.20 | no |
| 0.15 | 0.01 | no |
| 0.15 | 0.05 | no |
| 0.15 | 0.10 | no |
| 0.15 | 0.15 | no |
| 0.15 | 0.20 | no |

Observation : Only the images with std 0.01 can be used to successfully recover qr code images.



## Write key observations from the plots in task3_psnr_vs_alpha.png ??

### Invisibility - from top row
- As alpha increases, the PSNR decreases. This means that using a higher scaling factor (alpha) changes the original image too much. While it helps hide the QR code strongly, it damages the visual quality of the main image.

- For lower variance noise, PSNR is relatively higher. This simply means that when less random noise is added to the image, the original image naturally stays cleaner and looks better.


### Recoverability - from second row
- As alpha increases, PSNR gradually increases to a certain extent. This happens because a higher alpha embeds the QR code more strongly into the image. A stronger watermark survives noise better, making it easier to pull out a clear QR code later.

- For lower variance noise, there is a peak. If you increase alpha past this peak, it actually makes the recovered QR code look worse.

- For lower variance, PSNR is higher overall. Less noise means the hidden QR code gets damaged less in transit, so we get a much sharper and cleaner result when we extract it.