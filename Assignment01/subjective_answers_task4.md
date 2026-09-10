# Subjective Answers for task 4

## for each image and each quality factor qf, write minimum value of alpha for which a qr code scanner is able to extract your roll number ..


| qf | alpha | is qr readable |
| --- | --- | --- |
| 30 | 0.01 | no |
| 30 | 0.05 | no |
| 30 | 0.10 | no |
| 30 | 0.15 | 2/3 |
| 30 | 0.20 | 2/3 |
| 50 | 0.01 |  no|
| 50 | 0.05 | no |
| 50 | 0.10 | 2/3 |
| 50 | 0.15 | yes |
| 50 | 0.20 | yes |
| 70 | 0.01 | no |
| 70 | 0.05 | 2/3 |
| 70 | 0.10 | yes |
| 70 | 0.15 | yes |
| 70 | 0.20 | yes |
| 90 | 0.01 | 2/3 |
| 90 | 0.05 | yes |
| 90 | 0.10 | yes |
| 90 | 0.15 | yes |
| 90 | 0.20 | yes |


Qrcode extracted from the last image was very distroted. Therefore in some columns I've written 2/3 that means first two images give recovered and intact qr.



## Key observations


### Invisibility - from top row
- As alpha increases, the PSNR decreases. This means that using a higher scaling factor (alpha) changes the original image too much. While it helps hide the QR code strongly, it damages the visual quality of the main image.

- For higher quality factor, PSNR is relatively higher. This simply means that when image is compressed with higher qf, the original image naturally stays same.

### Recoverability - from second row
- As alpha increases, PSNR gradually increases to a certain extent. This happens because a higher alpha embeds the QR code more strongly into the image. A stronger watermark survives noise better, making it easier to pull out a clear QR code later.

- For higher qf PSNR is higher overall. It means that image is compressed lesser and it retains most of its data. Hence the new image with qr water mark has higher psnr with base image.