# Setup

This document explains how to set up a Python environment to run the code in this assignment.

## 1. Python version

Use atleast **Python 3.9** (tested with 3.10.9 and 3.10.12). Check your version with:

```bash
python3 --version
```

In some cases you may have the command `python` instead of `python3`.

If you don't have Python, install it from [python.org](https://www.python.org/downloads/) or via your system's package manager before continuing.
If you have problem installing Python. Try googling for the solution.

## 2. Create a virtual environment

From this directory, create an isolated environment named `cv_assignment1`:

```bash
python3 -m venv cv_assignment1
```

Activate it:

- **Linux / macOS**
  ```bash
  source cv_assignment1/bin/activate
  ```
- **Windows (PowerShell)**
  ```powershell
  cv_assignment1\Scripts\Activate.ps1
  ```

Your shell prompt should now be prefixed with `(cv_assignment1)`. Keep the environment activated for every step below and whenever you run the task scripts.

## 3. Install dependencies

Upgrade `pip` first, then install the pinned packages from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

| Package | Version | Used for |
| --- | --- | --- |
| `numpy` | 1.26.4 | Array/matrix operations, SVD math |
| `opencv-python` | 4.7.0.72 | Reading/writing images, JPEG compression |
| `matplotlib` | 3.8.4 | Plotting results |
| `python-dotenv` | 1.2.1 | Loading `watermark_path` from `.env` |

## 4. Verify the setup

Run:

```bash
python -c "import cv2, numpy, matplotlib, dotenv; print('All good')"
```

If this prints `All good` with no errors, you're ready to run the task scripts (`task1.py`, `task2.py`, `task3.py`, `task4.py`).

## 5. Deactivate when done

```bash
deactivate
```
