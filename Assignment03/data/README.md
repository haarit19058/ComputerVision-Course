# Data

## `frames/`

Frames 270 to 413 of the descent camera, after the lander pitched over to
look straight down: 144 consecutive grayscale PNG images of 2352 x 1728
pixels. The frame number is the file name (`270.png` is frame 270).

## `mission_constants.json`

Provided by mission control (the course staff):

| Key | Meaning |
|-----|---------|
| `focal_length_px` | camera focal length in pixels, at full resolution (2352 x 1728) |
| `frame_interval_s` | the camera's frame interval, in seconds |
| `anchor_frame` | a frame number between 270 and 413 |
| `anchor_altitude_m` | the lander's altitude at `anchor_frame`, in metres |

The run scripts work at a reduced scale (`--scale 0.25` by default) and
convert the focal length to that scale themselves. If a value is `null`,
Task 4 runs in relative units (see `provided/task4/run.py`).

Where the values come from:

* The images are Chang'e-5 landing camera (LCAM) level 2A data, released by
  the Ground Research and Application System of China's Lunar and Planetary
  Exploration Program (https://moon.bao.ac.cn). Frame `k` is LCAM-1 image
  number `k + 1`, taken on 1 December 2020 during the final descent.
* `focal_length_px` follows from the camera's published field of view,
  59 x 45 degrees over 2352 x 1728 pixels, for a pinhole camera. It is good
  to about 1 %.
* `frame_interval_s` is the camera's rate of one image per second.
* The altitude reading (about 100 m at frame 381) comes from a
  published reconstruction of the descent trajectory from radio tracking
  (Remote Sensing 13, 4837, 2021). Treat it as approximate.
