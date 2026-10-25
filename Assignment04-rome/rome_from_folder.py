"""Your own photos to a 3D point cloud, with the functions you wrote in rome_in_a_day.ipynb.

Give it 50 or more photos of one place: a folder, a zip of them, a direct link to a zip (a GitHub release asset, say),
or a Google Drive folder or zip link (shared as "Anyone with the link"). It runs the notebook's pipeline on them:
visual words with your tfidf_matrix and top_k_neighbours to propose pairs, your match_descriptors and a check on F for
each pair, your query_expansion and connected_components, the toolbox's maximum spanning tree, then the SfM (with your
homography_ratio, which uses your homography_dlt, choosing the first pair) and densify for the dense cloud.

  In the notebook (Colab or your computer), once your functions have run (the -i lets the script see them); the
  notebook's Chapter 7 (Task 10) does this, with a Taj Mahal example, and shows the results with show_results(out):
      %run -i rome_from_folder.py my_photos
  (On Colab the notebook's setup cell puts this file next to the notebook, from the course repository.)
  From a terminal, taking your functions from the saved notebook (uv run python if you set up with uv):
      python rome_from_folder.py my_photos --notebook rome_in_a_day.ipynb
  With the library versions (ref.<name>) instead of yours:
      python rome_from_folder.py my_photos --reference

Photos can be JPEG, PNG, WebP or TIFF, or a zip of them; not HEIC (on an iPhone: Settings > Camera > Formats > Most
Compatible, or export as JPEG). It writes the folder <name>_cloud (or --out): dense.ply and sparse.ply (open them in
MeshLab, CloudCompare or Blender; in Blender's PLY import choose Y as the up axis), view.html (the 3D view, opens in a
browser), cameras.csv and summary.txt. It needs the crate (data/mini_rome): its vocabulary turns your photos into
visual words. The notebook's setup cell downloads it. 150 photos take about 2 minutes on a laptop, a few
minutes on Colab.

Every photo set is different. The defaults are the notebook's; if yours misbehaves, change one setting at a time
(on the command line, after %run -i as well, or, after a first %run -i, as photos_to_cloud(..., top_k=20)). A run
into the same --out replaces the earlier results once it has found the photos, even if it then stops; give each try
its own --out to keep them:

  It says                                What to try
  "only N of M photos overlap"           --top-k 20 (or 30): each photo is compared with more photos; and take more
                                         photos in between the ones you have
  few photos placed, or a model in       more photos in between, then --top-k 20; photos from one spot (turning
  pieces, or "no start pair"             without walking) give no depth at all
  the model is a mess (pairs that are    --min-inliers 30: repeated windows and symmetric buildings make wrong pairs
  not the same view)                     (it cannot go below 20: the 3D step needs 20 matches per pair)
  the model is bent or curved            --focal 0.8 (or the right value for your camera): the focal length guess, in
                                         long sides, used for photos with no EXIF (WhatsApp and screenshots strip it);
                                         if cameras.csv says "exif", add --ignore-exif to use the guess for all
  the dense cloud is thin                --dense-images 120 (more photos testify), --long-side 960 (more pixels)
  it is too slow                         --long-side 480, --top-k 5, --rounds 1, --dense-images 32
  the cloud is upside down               --flip turns it over, in the PLY files and view.html (photos taken
                                         straight down, from a drone or a balcony, can come out that way)
  the cloud leans back                   turn it in the viewer: photos that all look up the same way with a narrow
                                         spread of headings (walking along a tall facade) keep their tilt
  more than 400 photos                   it works, but give it time; --rounds 1 saves the most
"""

if __name__ == "__main__" and "__file__" in globals() and getattr(
        __import__("sys").modules.get("IPython"), "get_ipython", lambda: None)() is not None:
    # %run (with -i or without) in a notebook: the script runs in a module of its own, so none of its names replace
    # yours (your functions are still found in the notebook, through __main__); only photos_to_cloud and
    # show_results come back
    import importlib.util as _rff_util
    import sys as _rff_sys
    try:
        _rff_spec = _rff_util.spec_from_file_location("rome_from_folder", __file__)
        _rff = _rff_util.module_from_spec(_rff_spec)
        _rff_sys.modules["rome_from_folder"] = _rff
        try:
            _rff_spec.loader.exec_module(_rff)
        except BaseException:
            _rff_sys.modules.pop("rome_from_folder", None)   # no half-built module left behind
            raise
        photos_to_cloud, show_results = _rff.photos_to_cloud, _rff.show_results
        _rff.run_as_script(_rff_sys.argv[1:])
    finally:
        for _rff_name in ("_rff_util", "_rff_sys", "_rff_spec", "_rff", "_rff_name"):
            globals().pop(_rff_name, None)
    raise SystemExit(0)                           # done: IPython ends a script quietly on SystemExit(0)

import argparse
import ast
import contextlib
import csv
import inspect
import io
import json
import linecache
import math
import lzma
import os
import re
import shutil
import struct
import symtable
import tempfile
import sys
import time
import traceback
import zipfile
import zlib

import cv2
import numpy as np
from PIL import Image, ImageOps

_here = os.path.dirname(os.path.abspath(globals().get("__file__") or "."))
for _base in (os.getcwd(), _here, os.path.dirname(_here)):   # the folder you run from, the script's, one up
    if os.path.isfile(os.path.join(_base, "toolbox", "reference.py")):   # a complete toolbox (not src/toolbox)
        sys.path.insert(0, _base)
        break

try:
    from toolbox.data import _read_rgb
    from toolbox.dense import densify
    from toolbox.geometry import camera_center
    from toolbox.geometry import ransac_fundamental_cv
    from toolbox.graph import component_graph, maximum_spanning_tree, neighbour_pairs
    from toolbox.plots import plot_reconstruction
    from toolbox.reference import ref
    from toolbox.sfm import _sift, landmark_scene, register_rest, run_sfm_skeleton
    from toolbox._yours import NotDefinedYet, yours
except ImportError as _err:                       # no toolbox next to the notebook (a fresh Colab session, say)
    raise SystemExit(f"stopped: no toolbox here ({_err}). Run the setup cell of rome_in_a_day.ipynb first, from "
                     "the folder that holds this script") from None

MIN_PHOTOS = 50          # photos needed in the folder
MIN_GROUP = 30           # photos needed in the largest group that overlaps
MANY_PHOTOS = 400        # more than this runs, with a note about time
LONG_SIDE = 640          # photos are resized to this long side (never enlarged), like the crate
N_FEATURES = 800         # RootSIFT keypoints per photo for the visual words, like the crate
N_MATCH = 2000           # RootSIFT keypoints per photo for checking pairs, like landmark_scene
MIN_INLIERS = 20         # a pair is real with this many matches that fit one F, as in the notebook
DEFAULT_TOP_K, DEFAULT_ROUNDS = 10, 2   # neighbours per photo, rounds of query expansion (the notebook's; other
                                        # names than the notebook's TOP_K and ROUNDS, which %run -i must not reset)
MAX_NEW_PAIRS = 20000          # new pairs checked per round of query expansion at most, as in the notebook
FOCAL = 1.2              # focal length guess for photos without EXIF, in long sides (the crate's guess for strays)
DENSE_IMAGES = 64        # photos densify uses, as in the notebook
MIN_LONG_SIDE = 64       # smaller photos have too few keypoints to match
EXTENSIONS = (".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".tif", ".tiff")   # what Pillow reads
OTHER_IMAGES = (".heic", ".heif", ".avif")   # phone formats this script cannot read: export them as JPEG
MARKER = ".rome_from_folder"   # in every output folder, so a later run never reads it as photos
DOWNLOAD = "rome_from_folder_download"   # where fetch_source unpacks a zip or link, inside its work folder
JUNK = {".DS_Store", "desktop.ini", "Thumbs.db"}   # files a system drops in any folder: not somebody's work
RESULTS = ("dense.ply", "sparse.ply", "view.html", "cameras.csv", "summary.txt")   # what a run writes in out
ZIP_ERRORS = (zipfile.BadZipFile, OSError, EOFError, ValueError, NotImplementedError, RuntimeError, zlib.error,
              lzma.LZMAError)   # ValueError: a name that is not the UTF-8 it claims, say
NAMES = ("tfidf_matrix", "top_k_neighbours", "match_descriptors", "query_expansion", "connected_components",
         "homography_dlt", "homography_ratio", "triangulate_points", "umeyama")
USED = NAMES[:7]         # the toolbox triangulates for the SfM itself, and there is no reference model to align to
CRATE_NAMES = ("mini_rome", "mini_rome_small")   # the course crate, never read as photos
CRATE_DIRS = [os.path.join(base, name) for base in ("data", os.path.join("data", "data")) for name in CRATE_NAMES]


class PhotoError(Exception):
    """A problem with the photos or the input, explained in plain words."""


def is_drive_link(url):
    """A Google Drive (or Google Docs) share link, which needs gdown; any other link is downloaded as it is."""
    return bool(re.match(r"https?://(drive|docs)\.google\.com/", url))


def link_id(url):
    """The file id in a Google Drive link, or None."""
    m = re.search(r"/d/([\w-]+)|[?&]id=([\w-]+)", url)
    return (m.group(1) or m.group(2)) if m else None


def source_name(source):
    """A short name for the photos: the folder's or zip's name (also for a direct link to a zip), or
    drive_<first 8 letters of the id> for a Drive link."""
    if re.match(r"https?://", source) and not is_drive_link(source):
        base = os.path.basename(re.sub(r"[?#].*$", "", source.rstrip("/"))) or "photos"
        return base[:-4] if base.lower().endswith(".zip") else base
    if re.match(r"https?://", source):
        m = re.search(r"/folders/([\w-]+)", source)
        fid = m.group(1) if m else link_id(source)
        return f"drive_{fid[:8]}" if fid else "drive_photos"
    base = os.path.basename(os.path.abspath(source)) or "photos"
    return base[:-4] if base.lower().endswith(".zip") else base


def fetch_source(source, work, say=print):
    """A local folder with the photos: source itself if it is a folder, else a zip file, a direct link to a zip, a
    Drive folder link or a Drive zip link, downloaded (and unzipped) into work/DOWNLOAD, which is emptied first (a new
    link never sees older photos)."""
    if os.path.isdir(source):
        return source
    folder = os.path.join(work, DOWNLOAD)
    if os.path.isfile(source) and zipfile.is_zipfile(source):
        shutil.rmtree(folder, ignore_errors=True)
        try:
            with zipfile.ZipFile(source) as zf:
                zf.extractall(folder)             # zipfile drops absolute paths and '..' from member names
        except ZIP_ERRORS as err:
            raise PhotoError(f"{source} could not be unzipped ({err}): zip the photos again as a plain zip (no "
                             "password)") from err
        return folder
    if not re.match(r"https?://", source):
        raise PhotoError(f"{source} is neither a folder, a zip file nor a link")
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    if not is_drive_link(source):                 # a direct link to a zip (the course's release, say)
        import urllib.request
        name = os.path.join(folder, "photos.zip")
        say(f"downloading {source}")
        import http.client
        try:
            with urllib.request.urlopen(source, timeout=60) as response, open(name, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
                expected = int(response.headers.get("Content-Length") or 0)
            if os.path.getsize(name) < expected:
                raise http.client.IncompleteRead(b"", expected - os.path.getsize(name))
        except (http.client.IncompleteRead, ConnectionError, TimeoutError) as err:
            raise PhotoError(f"the download of {source} stopped before the end ({err!r}): run it again") from err
        except Exception as err:                  # HTTP errors, no connection, a timeout
            raise PhotoError(f"could not download {source} ({err}). Check the link: it must lead straight to a zip "
                             "of the photos (or be a Google Drive link)") from err
    else:
        import gdown
        old = int(re.match(r"\d*", gdown.__version__).group() or 0) < 6   # gdown 5 lists 50 files of a folder
        try:
            if "/folders/" in source and old:
                raise PhotoError("this gdown (older than 6) gets at most 50 files of a Drive folder, and this part "
                                 "needs 50 or more: zip the folder and pass the link to the zip, or run !pip install "
                                 "-U gdown, restart the session and run again")
            if "/folders/" in source:
                try:
                    got = gdown.download_folder(source, output=folder, quiet=False)
                except Exception as err:          # gdown 6 downloads what it can, then raises for the rest
                    if not list_photos(folder):
                        raise
                    say(f"note: some files of the folder did not download ({err}); going on with the photos that "
                        "did")
                    return folder
                if got is None:
                    raise PhotoError(f"could not list the Drive folder {source}. Check that it is shared as "
                                     "'Anyone with the link'")
                return folder
            fid = link_id(source)
            name = gdown.download(**(dict(id=fid) if fid else dict(url=source)),
                                  output=os.path.join(folder, "photos.zip"), quiet=False)
        except PhotoError:
            raise
        except Exception as err:                  # gdown raises several kinds
            raise PhotoError(f"could not download {source} ({err}). Check that it is shared as 'Anyone with the "
                             "link'")
    if not name or not zipfile.is_zipfile(name):
        raise PhotoError(f"{source} is not a zip file: give a link to a zip of the photos, or a Drive folder link")
    try:
        with zipfile.ZipFile(name) as zf:
            zf.extractall(folder)                 # zipfile drops absolute paths and '..' from member names
    except ZIP_ERRORS as err:
        raise PhotoError(f"the zip from {source} could not be unzipped ({err}): upload it again as a plain zip (no "
                         "password)") from err
    os.remove(name)
    return folder


def list_photos(folder, skip=(), extensions=EXTENSIONS):
    """Photo files (JPEG, PNG, WebP, TIFF; any case) in folder and its subfolders, sorted. Left out: hidden files,
    __MACOSX, the folders in skip, output folders of earlier runs (they hold MARKER) and the course crate."""
    skip = {os.path.abspath(s) for s in skip}

    def wanted(root, d):
        path = os.path.join(root, d)
        return not (d.startswith((".", "__MACOSX")) or os.path.abspath(path) in skip
                    or os.path.exists(os.path.join(path, MARKER))
                    or (d in CRATE_NAMES and os.path.basename(os.path.abspath(root)) == "data"))
    found = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if wanted(root, d))
        found += [os.path.join(root, f) for f in sorted(files)
                  if f.lower().endswith(extensions) and not f.startswith(".")]
    return found


def exif_focal(img, w, h, full=None):
    """Focal length in pixels for the photo resized to w x h, from EXIF: FocalLengthIn35mmFilm (defined on the
    diagonal: a 35 mm frame's is 43.27 mm), else FocalLength with the sensor's FocalPlaneXResolution (some cameras
    record only those), measured against full, the photo's size as stored (img.size unless a JPEG was opened at a
    reduced size). None when the camera recorded neither, or the values make no sense."""
    try:
        ifd = img.getexif().get_ifd(0x8769)
        f35 = float(ifd.get(0xA405) or 0)
        if f35 > 0:
            f = f35 * float(np.hypot(w, h)) / 43.27
        else:
            f_mm, res = float(ifd.get(0x920A) or 0), float(ifd.get(0xA20E) or 0)
            per_mm = {2: 1 / 25.4, 3: 1 / 10.0, 4: 1.0}.get(int(ifd.get(0xA210) or 2), 0)   # inch, cm, mm
            if not (f_mm > 0 and res > 0 and per_mm):
                return None
            # a sensor under 12 mm: most likely the photo was resized after shooting (editors keep the camera's
            # resolution tag, even when they update PixelXDimension), so guess; a compact camera's own small sensor
            # gets the guess too (cameras.csv says "guess")
            stored = max(full or img.size)
            own = max(int(ifd.get(0xA002) or 0), int(ifd.get(0xA003) or 0))   # PixelX/YDimension, if recorded
            if own > stored:                      # the camera's own size, kept by an export: it proves the resize
                stored = own                      # and gives the size the resolution tag belongs to
            elif stored / (res * per_mm) < 12:
                return None
            f = f_mm * res * per_mm * max(w, h) / stored   # pixels at the camera's size, then resized
    except (TypeError, ValueError, OSError, SyntaxError, ZeroDivisionError, OverflowError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _rgb(img):
    """An RGB uint8 array of a PIL image; 16 and 32 bit images are scaled down to 8 bit, not clipped."""
    if img.mode in ("I", "I;16", "I;16L", "I;16B", "I;16N", "F"):
        a = np.asarray(img, dtype=np.float64)
        a = a / 257.0 if a.max() > 255 else a       # 16 bit: 65535 -> 255
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L")
    return np.asarray(img.convert("RGB"))


TURNS = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
         5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
         8: Image.Transpose.ROTATE_90}       # EXIF orientation -> the turn that shows the photo upright


def _upright(im):
    """The photo turned upright by its EXIF orientation: Pillow's exif_transpose, or, when a damaged EXIF block makes
    that fail, the same turn read from the orientation tag alone."""
    try:
        return ImageOps.exif_transpose(im)
    except Exception:                             # Pillow rewrites the whole EXIF block on the way: it can trip
        try:
            turn = TURNS.get(int(im.getexif().get(0x0112) or 1))
        except Exception:
            turn = None
        return im.transpose(turn) if turn is not None else im


def load_photos(paths, folder, long_side=LONG_SIDE, focal=FOCAL, root=None, ignore_exif=False):
    """Read each photo (its EXIF rotation applied), resize it to a long side of long_side px (never up) and save it as
    folder/NNNN.jpg. The focal length comes from EXIF, else it is focal x the long side. Returns dict(paths (n,) the
    saved copies, names (the original files, relative to root if given), sizes (n,2) (width, height), K (n,3,3),
    from_exif (n,) bool: focal length from EXIF, skipped: the files that could not be read)."""
    os.makedirs(folder, exist_ok=True)
    keep, names, sizes, Ks, from_exif, skipped = [], [], [], [], [], []
    limit, Image.MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS, None   # your own photos: 200 MP phones are fine
    try:
        for p in paths:
            try:
                with Image.open(p) as im:
                    full = im.size                # before draft: a JPEG may decode at 1/2, 1/4 or 1/8 of it
                    im.draft("RGB", (long_side, long_side))   # a JPEG decodes at a smaller size: faster, less memory
                    rgb = _rgb(_upright(im))
                    h, w = rgb.shape[:2]
                    if max(h, w) > long_side:
                        s = long_side / max(h, w)
                        rgb = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))),
                                         interpolation=cv2.INTER_AREA)
                    h, w = rgb.shape[:2]
                    f = None if ignore_exif else exif_focal(im, w, h, full)
            except (OSError, ValueError, SyntaxError, TypeError, KeyError, IndexError, struct.error,
                    Image.DecompressionBombError, cv2.error):
                skipped.append(p)
                continue
            dst = os.path.join(folder, f"{len(keep):04d}.jpg")
            if not cv2.imwrite(dst, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise PhotoError(f"could not write {dst}: use an output folder (--out) you can write to, with a plain "
                                 "(ASCII) path")
            fx = f if f is not None else focal * max(h, w)
            keep.append(dst)
            names.append(os.path.relpath(p, root) if root else os.path.basename(p))
            sizes.append((w, h))
            Ks.append([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])
            from_exif.append(f is not None)
    finally:
        Image.MAX_IMAGE_PIXELS = limit
    return dict(paths=np.array(keep, dtype=str), names=names, sizes=np.array(sizes, int).reshape(-1, 2),
                K=np.array(Ks, float).reshape(-1, 3, 3), from_exif=np.array(from_exif, bool), skipped=skipped)


def word_features(img):
    """RootSIFT keypoints (n,2) float32 and descriptors (n,128) float32 of an RGB photo, made as the crate's words were
    made (dev/make_dataset.py): the N_FEATURES strongest, every keypoint kept."""
    kp, des = cv2.SIFT_create(nfeatures=N_FEATURES).detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None)
    if des is None or len(kp) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    kp, des = kp[:N_FEATURES], des[:N_FEATURES]
    des = des / np.maximum(des.sum(1, keepdims=True), 1e-7)
    return np.array([k.pt for k in kp], np.float32), np.sqrt(des).astype(np.float32)


def nearest_words(desc, vocab, block=4096):
    """For each descriptor (n,128), the number of the nearest word of vocab (W,128), plain L2, in blocks."""
    V = np.asarray(vocab, np.float32)
    vv = np.einsum("ij,ij->i", V, V)
    D = np.asarray(desc, np.float32)
    out = np.empty(len(D), np.int64)
    for s in range(0, len(D), block):
        out[s:s + block] = np.argmin(vv[None, :] - 2 * D[s:s + block] @ V.T, axis=1)   # |d|^2 is the same per row
    return out


def _crate_bases():
    """Where the crate may be: under the folder you run from, the script's folder, or the one above (as the toolbox)."""
    return [os.getcwd(), _here, os.path.dirname(_here)]


def find_vocab():
    """The crate's vocabulary (W,128) float16. The notebook's setup cell downloads the crate."""
    for base in _crate_bases():
        for folder in CRATE_DIRS:
            path = os.path.join(base, folder, "features", "vocab.npy")
            if os.path.isfile(path):
                return np.load(path)
    raise PhotoError("no crate vocabulary (data/mini_rome/features/vocab.npy): run the setup cell of "
                     "rome_in_a_day.ipynb first, from this folder; it downloads the crate")


def build_crate(photos, vocab, root):
    """A crate like load_mini_rome's for the photos: RootSIFT keypoints and their visual words, the K of each photo,
    no hidden labels and no survey cameras. The toolbox functions work on it unchanged."""
    t0 = time.time()
    xy, words, offsets = [], [], [0]
    for path in photos["paths"]:
        pts, desc = word_features(_read_rgb(str(path)))
        xy.append(pts)
        words.append(nearest_words(desc, vocab))
        offsets.append(offsets[-1] + len(pts))
    n = len(photos["paths"])
    gt = dict(index=np.zeros(0, np.int64), K=np.zeros((0, 3, 3)), R=np.zeros((0, 3, 3)), t=np.zeros((0, 3)),
              group=np.zeros(0, dtype=str), secondary=np.zeros(0, bool))
    rome = dict(root=root, ids=np.array([f"{k:04d}" for k in range(n)]), images=np.asarray(photos["paths"]),
                sizes=photos["sizes"], labels=-np.ones(n, np.int64), groups=[], K=photos["K"],
                has_gt=np.zeros(n, bool), gt=gt, offsets=np.array(offsets, np.int64),
                xy=np.concatenate(xy).astype(np.float32) if xy else np.zeros((0, 2), np.float32),
                words=np.concatenate(words).astype(np.int64) if words else np.zeros(0, np.int64),
                vocab=vocab, attribution=None)
    print(f"{n} photos: {len(rome['words']):,} keypoints and their visual words in {time.time() - t0:.0f} s")
    return rome


def verify_by_matching(pairs, feats, min_inliers=MIN_INLIERS):
    """Check pairs (M,2) with YOUR match_descriptors (ratio test and mutual check) on RootSIFT, then RANSAC on F within
    1.5 px, as landmark_scene does. With a few hundred photos this finds far more real pairs than the word matches
    of verify_pairs, which need the crate's many near duplicates. feats: (xy, desc) per photo.
    Returns (n_inl (M,) int64 inlier counts, ok (M,) bool)."""
    match_descriptors = yours("match_descriptors")
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    n_inl = np.zeros(len(pairs), np.int64)
    for k, (i, j) in enumerate(pairs.tolist()):
        (x1, d1), (x2, d2) = feats[i], feats[j]
        m = np.asarray(match_descriptors(d1.astype(np.float32), d2.astype(np.float32)), np.int64).reshape(-1, 2)
        if len(m) >= min_inliers:
            try:
                n_inl[k] = ransac_fundamental_cv(x1[m[:, 0]], x2[m[:, 1]], thresh=1.5, iters=2000)[1].sum()
            except cv2.error:                     # OpenCV gives up on a degenerate pair: not a real pair
                n_inl[k] = 0
    return n_inl, n_inl >= min_inliers


def _literal(node):
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False


def _python(src):
    """A notebook cell as plain Python: a cell magic line (%%time) is dropped, line magics and shell lines (%, !)
    become calls that are left out later (IPython's own rewriting when it is there, which knows continuation lines)."""
    lines = src.splitlines()
    while lines and not lines[0].strip():
        lines = lines[1:]                         # Jupyter allows blank lines above a cell magic
    if lines and lines[0].lstrip().startswith("%%"):
        lines = lines[1:]                         # %%time, %%capture: the body is Python
    src = "\n".join(lines) + "\n"
    try:
        from IPython.core.inputtransformer2 import TransformerManager
        return TransformerManager().transform_cell(src)
    except ImportError:
        return "\n".join("" if line.lstrip().startswith(("%", "!")) else line for line in src.splitlines())


CONSTRUCTORS = ("__init__", "__new__", "__post_init__")   # what Class() runs
HEAVY = {"load_mini_rome", "find_crate", "fetch_zip", "unzip", "landmark_scene", "run_sfm_skeleton", "register_rest",
         "densify", "densify_survey", "verify_pairs", "survey_scene", "st_peters_maps", "input"}   # never run these


def _runs(fn, depth=0):
    """The code that runs when function fn is called: its body, the decorators and defaults of the functions it
    defines, class bodies, and the bodies of inner functions it calls by name; with depth=1 also the bodies of all the
    functions it defines (a decorator factory returns one, which runs when the decorator is applied). Lambda bodies
    are left out."""
    out, inner, opened, todo = [], {}, set(), list(fn.body)
    while True:
        while todo:
            n = todo.pop()
            out.append(n)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner[n.name] = n.body
                todo += n.decorator_list + n.args.defaults + [d for d in n.args.kw_defaults if d is not None]
                if depth > 0:
                    out += _runs(n, depth - 1)
                continue
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Lambda):
                todo.append(n.func.body)          # (lambda: ...)() runs it now
            if isinstance(n, ast.Lambda):
                todo += n.args.defaults + [d for d in n.args.kw_defaults if d is not None]
                if depth > 0:                     # a factory may return it: it runs when the decorator is applied
                    todo.append(n.body)
                continue
            if isinstance(n, ast.ClassDef):       # its body runs now; its constructor when it is called by name
                inner[n.name] = [b for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                                 and m.name in CONSTRUCTORS for b in m.body]
                todo += n.decorator_list + n.bases + [k.value for k in n.keywords]
                returned = any(isinstance(r, ast.Return) and isinstance(r.value, ast.Name) and r.value.id == n.name
                               for r in ast.walk(fn))
                if depth > 0 and returned:        # a factory that returns the class: @factory(...) constructs it
                    todo += inner[n.name]
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        todo += m.decorator_list + m.args.defaults + [d for d in m.args.kw_defaults if d is not None]
                    else:
                        todo.append(m)
                continue
            todo += list(ast.iter_child_nodes(n))
        called = {c.func.id for c in out if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                  and c.func.id in inner} - opened
        if not called:
            return out
        for name in sorted(called):               # an inner helper or class it calls runs now as well
            opened.add(name)
            todo += list(inner[name])


def _root(expr):
    """The name an attribute chain starts from: np for np.random.default_rng, os for os.environ.get."""
    while isinstance(expr, (ast.Attribute, ast.Call, ast.Subscript)):
        expr = expr.func if isinstance(expr, ast.Call) else expr.value
    return getattr(expr, "id", "")


def _heavy_in(nodes, also=frozenset()):
    """Does any call among nodes do something slow or interactive (the crate, the pipeline, a plot), or call one of
    the notebook's own functions, classes or methods in also (those that do: "f", "Class", "Class.method",
    ".method"; "import:name" marks a module, whose functions are never the notebook's)?"""
    for n in nodes:
        if isinstance(n, ast.Call):
            name = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
            attr = isinstance(n.func, ast.Attribute)
            base, root = (getattr(n.func.value, "id", ""), _root(n.func)) if attr else ("", "")
            method = attr and root != "plt" and f"import:{root}" not in also
            if isinstance(n.func, ast.Call) and isinstance(n.func.func, ast.Name) and f"{n.func.func.id}()" in also:
                return True                       # Loader()(...): a heavy __call__
            if (name in HEAVY or name.startswith(("show_", "plot_")) or base == "plt"
                    or (base and f"{base}.{name}" in also if attr else name in also) or method and f".{name}" in also):
                return True
    return False


def _heavy(node, also=frozenset()):
    """_heavy_in over the whole statement or expression."""
    return _heavy_in(ast.walk(node), also)


def _heavy_decorator(node, also_now, also_one):
    """Does a decorator run something heavy when it is applied? @deco runs deco's own code (also_now: not the wrapper
    it returns, which runs later); @factory(...) runs the factory and then the function it returns (also_one); a
    heavy function merely named in the arguments does not run."""
    if isinstance(node, ast.Call):
        callee, applied = node.func, also_one
        if any(_heavy(a, also_now) for a in node.args + [k.value for k in node.keywords]):
            return True
    else:
        callee, applied = node, also_now
    if isinstance(callee, ast.Name):
        return callee.id in applied or callee.id in HEAVY
    if isinstance(callee, ast.Attribute):              # @reg.add: runs reg.add(f)
        root, base = _root(callee), getattr(callee.value, "id", "")
        return f"{base}.{callee.attr}" in applied or f"import:{root}" not in applied and f".{callee.attr}" in applied
    return _heavy(callee, also_now)


def _import_block(node):
    """A top-level try or if made only of imports, pass and literal assignments (try: import x except: x = None)."""
    if not isinstance(node, (ast.Try, ast.If)) or isinstance(node, ast.If) and any(
            isinstance(n, ast.Call) for n in ast.walk(node.test)):
        return False                              # an if-test with a call could do anything: left out
    inner = [n for n in ast.walk(node) if isinstance(n, ast.stmt) and n is not node]
    return all(isinstance(n, (ast.Import, ast.ImportFrom, ast.Pass)) or isinstance(n, ast.Assign) and _literal(n.value)
               for n in inner) and any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in inner)


def _globals_read(node):
    """The global names a statement reads when it runs or is called: for a def or class, its free globals (symtable
    knows parameters, locals and loop variables are not), plus its decorators and defaults."""
    table = symtable.symtable(ast.unparse(node), "<notebook>", "exec")
    out = {sym.get_name() for sym in table.get_symbols() if sym.is_referenced()}
    todo = list(table.get_children())
    while todo:
        t = todo.pop()
        out |= {sym.get_name() for sym in t.get_symbols() if sym.is_global() and sym.is_referenced()}
        todo += t.get_children()
    return out


def notebook_definitions(path):
    """The code of a saved notebook that the 9 functions need: imports, functions and classes, constants (NAME = a
    literal), assignments to the 9 names (tfidf_matrix = ref.tfidf_matrix, say), and assignments of the other global
    names the 9 functions (and the helpers they call) read, like rng = np.random.default_rng(0). Never an assignment
    that loads the crate, runs the pipeline or draws (see HEAVY); never magics, shell lines or cells that do not
    parse."""
    nodes = _notebook_nodes(path)
    is_def = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    binds = {k: _bound(node) for k, node in enumerate(nodes)}          # plain assignments: the names they bind
    binds = {k: b for k, b in binds.items() if b}
    modules = {f"import:{(a.asname or a.name).split('.')[0]}" for node in nodes
               if isinstance(node, (ast.Import, ast.ImportFrom)) for a in node.names}

    def runs_heavy(fn, found, depth):
        return _heavy(fn, found) if depth is None else _heavy_in(_runs(fn, depth), found)

    def heavy_parts(depth):
        """The notebook's own functions, classes and methods that do heavy work when called, as names for _heavy:
        anywhere inside them (depth None), or in the code that runs at once (_runs with depth 0 or 1)."""
        found = set(modules)
        while True:
            more = set()
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and runs_heavy(node, found, depth):
                    more.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    for m in node.body:
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and runs_heavy(m, found, depth):
                            more |= {f"{node.name}.{m.name}", f".{m.name}"}   # Q.load(), q.load(), self.load()
                            if m.name in CONSTRUCTORS:
                                more.add(node.name)                         # P() runs it (class X(metaclass=P) too)
                            if m.name == "__call__":
                                more.add(f"{node.name}()")                  # P()(...) runs it
            if more <= found:
                return frozenset(found)
            found |= more
    also = also_now = heavy_parts(0)              # what runs when a statement runs (not what it only defines)
    also_one = set(also_now)                      # what @name(...) runs: name's code, then the function it returns
    instances = {}                                # reg = Reg(): reg.add(...) is Reg's add
    for node in nodes:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    instances.setdefault(node.value.func.id, set()).add(t.id)
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _heavy_in(_runs(node, 1), also_now):
            also_one.add(node.name)
        elif isinstance(node, ast.ClassDef):
            methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if any(m.name in CONSTRUCTORS + ("__call__",) and _heavy_in(_runs(m, 0), also_now) for m in methods):
                also_one.add(node.name)           # @Cls(...): Cls(...) then its __call__(f)
            for m in methods:                     # @reg.add(...), @Tools.with_crate(...): a method as the factory
                if _heavy_in(_runs(m, 1), also_now):
                    also_one |= {f"{node.name}.{m.name}"} | {f"{v}.{m.name}" for v in instances.get(node.name, ())}
    also_one = frozenset(also_one)
    keep = {k for k, node in enumerate(nodes)
            if isinstance(node, is_def) and not _heavy_header(node, also, also_now, also_one) or _import_block(node)}
    keep |= {k for k, b in binds.items() if not _heavy(nodes[k], also) and (
        b <= set(NAMES) or isinstance(nodes[k], (ast.Assign, ast.AnnAssign)) and _literal(nodes[k].value))}
    defined = {}                                  # name -> statements that bind it, in order
    for k, node in enumerate(nodes):
        names = {node.name} if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else binds.get(
            k, set())
        for name in names:
            defined.setdefault(name, []).append(k)
    needed, todo = set(), list(NAMES)
    todo += [n.id for k in keep if isinstance(nodes[k], ast.If) for n in ast.walk(nodes[k].test)
             if isinstance(n, ast.Name)]                # if IN_COLAB: import ...  needs IN_COLAB
    while todo:                                   # from the 9 names, through what each reads
        name = todo.pop()
        if name in needed:
            continue
        needed.add(name)
        for k in defined.get(name, []):
            if _heavy(nodes[k], also) if k in binds else _heavy_header(nodes[k], also, also_now, also_one):
                continue
            keep.add(k)
            todo += sorted(_globals_read(nodes[k]) - needed)
    return ast.unparse(ast.Module(body=[nodes[k] for k in sorted(keep)], type_ignores=[]))


def _notebook_nodes(path):
    """The top-level statements of a saved notebook's code cells (magics and shell lines rewritten, cells that do not
    parse left out)."""
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    nodes = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        try:
            nodes += ast.parse(_python("".join(cell.get("source", "")))).body
        except SyntaxError:
            continue
    return nodes


def _bound(node):
    """The names a plain assignment binds (a = ..., a, b = ..., a: int = ..., a += ...), else an empty set."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
        targets = [node.target]
    else:
        return set()
    names = set()
    for t in targets:
        parts = t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]
        if not all(isinstance(e, ast.Name) for e in parts):
            return set()                          # a[0] = ..., obj.x = ...: not a plain binding
        names |= {e.id for e in parts}
    return names


def _heavy_header(node, also=frozenset(), also_now=frozenset(), also_one=frozenset()):
    """Does a def or class run something heavy when it is defined: decorators, defaults, a class's bases and body,
    and the decorators and defaults of its methods?"""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parts = node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
        return any(_heavy(n, also) for n in parts) or any(_heavy_decorator(n, also_now, also_one)
                                                          for n in node.decorator_list)
    if isinstance(node, ast.ClassDef):
        meta = [k.value for k in node.keywords]   # class X(metaclass=Meta): Meta(...) runs now
        defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        return (any(_heavy_decorator(n, also_now, also_one) for n in node.decorator_list)
                or any(_heavy(n, also) or _heavy_decorator(n, also_now, also_one) for n in meta)
                or any(_heavy(n, also) for n in node.bases)
                or any(_heavy_header(n, also, also_now, also_one) if isinstance(n, defs) else _heavy(n, also)
                       for n in node.body))
    return False


def notebook_names(path):
    """Every global name a saved notebook binds (assignments, loops, imports, defs, in blocks too; not the variables
    of comprehensions, which stay inside them)."""
    out, todo = set(), list(_notebook_nodes(path))
    while todo:                                   # (symtable cannot tell: Python 3.12+ inlines comprehensions)
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in node.names}
            continue
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda)):
            continue                              # their variables stay inside them
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        todo += list(ast.iter_child_nodes(node))
    return out


def use_notebook(path):
    """Run the definitions of a saved notebook (notebook_definitions) in a namespace of their own, and put the 9
    functions it defines into __main__, where the toolbox looks for them. A statement that fails (an import of
    google.colab outside Colab, say) is skipped with a note."""
    if not os.path.isfile(path):
        raise PhotoError(f"no notebook at {path}")
    ns = {"__name__": "rome_in_a_day"}
    src = notebook_definitions(path)
    name = f"<notebook {path}>"                   # tracebacks show this code, not the notebook file's JSON
    linecache.cache[name] = (len(src), None, src.splitlines(True), name)
    for node in ast.parse(src).body:
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), ns)
        except Exception as err:                  # the student's code: anything can fail
            print(f"note: skipped from {path}: {ast.unparse(node).splitlines()[0][:80]} ({type(err).__name__}: {err})")
    main = sys.modules["__main__"]
    found = [n for n in NAMES if callable(ns.get(n))]
    for n in found:
        setattr(main, n, ns[n])
    print(f"from {path}: {', '.join(found) or 'none of the 9 functions'}")


@contextlib.contextmanager
def reference_functions():
    """While the block runs, the 9 functions in __main__ are the library versions ref.<name>; afterwards whatever
    __main__ had before comes back (your own functions are not lost)."""
    main = sys.modules["__main__"]
    saved = {n: main.__dict__[n] for n in NAMES if n in main.__dict__}
    for n in NAMES:
        setattr(main, n, getattr(ref, n))
    try:
        yield
    finally:
        for n in NAMES:
            if n in saved:
                setattr(main, n, saved[n])
            else:
                main.__dict__.pop(n, None)


def viewer_rotation(cams, flip=False):
    """Rotation (3,3) that turns the model for 3D viewers: up becomes +y and the cameras' mean viewing direction -z.
    Up is the direction all the cameras' x axes are level with (photos are held level, whatever their pitch), signed
    by their mean up (minus y: image y points down), or for steep photos by whether they look up or down. When the x
    axes differ only by a hand's roll, or there are fewer than 3 photos, it is their mean up. So neither a first photo
    held sideways nor photos looking up at a tower tilt the cloud. For one level camera it is (x, -y, -z); flip=True
    turns the result over (--flip).
    Near straight up or down, rotations cannot tell a camera looking up at a dome from one looking down from a drone
    once SfM adds a degree of error; this script is for photos of buildings, so it trusts the look-up cue (looking up
    at 75 to 85 degrees from one side: right 95 to 100% from 20 degrees of heading spread). Drone photos straight
    down are right 97% from 60 degrees of heading spread and always with a full turn, but may be upside down with
    less (run again with --flip). Photos all tilted the same way with a narrow spread of headings (under about 10
    degrees with a hand's roll, more for moderate pitch: walking along a tall facade looking up) keep that tilt."""
    Rs = np.array([R for R, _ in cams.values()]).reshape(-1, 3, 3)
    up = -Rs[:, 1].mean(axis=0)
    view = Rs[:, 2].mean(axis=0)
    _, sv, Vt = np.linalg.svd(Rs[:, 0])           # photos are held level: their x axes are horizontal
    n = Vt[2]
    along_view = np.linalg.norm(view) > 1e-6 and abs(n @ view) > 0.9 * np.linalg.norm(view)
    gate = 0.07 if along_view else 0.05           # x axes spread only by hand roll leave the view axis as the gap
    # (0.07 measured on simulated sets with 3 degrees of roll: level sets right 97 to 100% even with 2 degrees of
    # heading spread; steep sets right 96% from 10 degrees of spread; below that, steep and level look the same)
    if len(Rs) >= 3 and sv[1] > gate * sv[0] and sv[2] < 0.3 * sv[1]:   # one direction they all miss: that is up
        if abs(n @ up) > 0.5:                     # mostly level photos: their mean up says which way
            sign = n @ up
        else:                                     # steep photos: look up or down? a camera looking up has its up
            c = -np.mean((-Rs[:, 1] @ n) * (Rs[:, 2] @ n))   # pointing back, one looking down forward: per camera,
            sign = -(n @ view) * (np.sign(c) if abs(c) > 0.02 else 1.0)   # (up . view = 0, so this is the dot of their
            #                                     horizontal parts); no cue (straight down): a drone, looking down
        up = n if sign > 0 else -n
    y = up / max(np.linalg.norm(up), 1e-12)
    z = -Rs[:, 2].mean(axis=0)
    z -= (z @ y) * y
    if np.linalg.norm(z) < 1e-6:                  # every camera looks straight up or down: use the first one's x
        x = Rs[0, 0] - (Rs[0, 0] @ y) * y
        z = np.cross(x / np.linalg.norm(x), y)
    z /= np.linalg.norm(z)
    M = np.array([np.cross(y, z), y, z])
    return np.diag([1.0, -1.0, -1.0]) @ M if flip else M   # flip: turned over (half a turn about the viewer's x)


def viewer_frame(pts, M=None):
    """Points turned for 3D viewers with the rotation M (viewer_rotation); the default (x, -y, -z) suits a model whose
    first camera is level (it looks along +z with y down; viewers want y up)."""
    M = np.diag([1.0, -1.0, -1.0]) if M is None else np.asarray(M, float)
    return (np.asarray(pts, np.float64).reshape(-1, 3) @ M.T).astype(np.float32)


def level_view(fig, M):
    """Turn the 3D view of fig so its up is M's (the PLY's up, in the model's own coordinates), instead of the
    cameras' mean up, which leans back by the photos' mean pitch. The eye keeps its distance and sits above and
    behind the cameras, as the toolbox places it."""
    cam = fig.layout.scene.camera
    up = np.asarray(M, float)[1]

    def vec(v, default):
        return np.array([v.x, v.y, v.z], float) if v is not None and v.x is not None else np.array(default, float)
    centre = vec(cam.center, [0, 0, 0])
    d = vec(cam.eye, [1.25, 1.25, 1.25]) - centre  # plotly's default eye when there is none
    old_up = vec(cam.up, up)
    old_up = old_up / max(np.linalg.norm(old_up), 1e-12)
    h = d - (d @ old_up) * old_up
    slope = (d @ old_up) / max(np.linalg.norm(h), 1e-12)   # how far above its horizon the toolbox put the eye
    back = d - (d @ up) * up
    if np.linalg.norm(back) < 1e-9:               # the old eye was on the new up axis: look along -z of the PLY
        back = np.asarray(M, float)[2]
    back = back / np.linalg.norm(back)
    view = (back + slope * up) / np.linalg.norm(back + slope * up)
    eye = centre + np.linalg.norm(d) * view
    fig.update_layout(scene_camera=dict(up=dict(zip("xyz", up.tolist())), eye=dict(zip("xyz", eye.tolist())),
                                        center=dict(zip("xyz", centre.tolist()))))
    return fig


def write_ply(path, pts, colors):
    """Binary PLY with x, y, z (float32) and red, green, blue (uint8). Rows with a NaN or inf are left out. Returns
    the number of points written."""
    pts = np.asarray(pts, np.float32).reshape(-1, 3)
    colors = np.asarray(colors, np.uint8).reshape(-1, 3)
    ok = np.isfinite(pts).all(axis=1)
    rows = np.empty(int(ok.sum()), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                                          ("blue", "u1")])
    for k, a in enumerate(("x", "y", "z")):
        rows[a] = pts[ok, k]
    for k, a in enumerate(("red", "green", "blue")):
        rows[a] = colors[ok, k]
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rows)}\n"
              + "".join(f"property float {a}\n" for a in "xyz")
              + "".join(f"property uchar {a}\n" for a in ("red", "green", "blue")) + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rows.tobytes())
    return len(rows)


def read_ply(path):
    """Points (n,3) float32 and colours (n,3) uint8 of a PLY written by write_ply."""
    with open(path, "rb") as f:
        data = f.read()
    head, body = data.split(b"end_header\n", 1)
    n = int(next(line for line in head.decode("ascii").splitlines() if line.startswith("element vertex")).split()[-1])
    rows = np.frombuffer(body, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"),
                                      ("b", "u1")], count=n)
    return (np.stack([rows["x"], rows["y"], rows["z"]], 1), np.stack([rows["r"], rows["g"], rows["b"]], 1))


def show_results(out, n_photos=12, max_points=80000):
    """Show what a run wrote in out: summary.txt, some of the photos it placed, and the dense cloud as the PLY lies
    (y up) from the front and from above. A run that stopped shows its summary only."""
    import matplotlib.pyplot as plt
    out = os.path.expanduser(os.fspath(out))
    summary = os.path.join(out, "summary.txt")
    if not os.path.isfile(summary):
        raise PhotoError(f"no results in {out}: rome_from_folder.py has not been run with --out {out}, or its "
                         "run stopped before writing any (its messages say why)")
    with open(summary, encoding="utf-8") as f:
        print(f.read())
    cameras, dense = os.path.join(out, "cameras.csv"), os.path.join(out, "dense.ply")
    if not (os.path.isfile(cameras) and os.path.isfile(dense)):
        print("(this run stopped before writing the cloud: see the summary above)")
        return
    with open(cameras, newline="", encoding="utf-8") as f:
        placed = [r for r in csv.DictReader(f) if r["placed"] == "1"]
    spread = np.linspace(0, len(placed) - 1, min(n_photos, len(placed))).astype(int) if placed else []
    pick = [placed[k] for k in spread]
    if pick:
        cols = 6
        rows = -(-len(pick) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.8 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, r in zip(axes.ravel(), pick):
            ax.imshow(_read_rgb(os.path.join(out, "images", r["photo"])))
            ax.set_title(r["file"][-22:], fontsize=7)
        fig.suptitle(f"{len(placed)} photos placed (a few of them)", fontsize=10)
        plt.show()
    P, C = read_ply(dense)
    if len(P) == 0:
        print("the dense cloud is empty")
        return
    if len(P) > max_points:
        keep = np.random.default_rng(0).choice(len(P), max_points, replace=False)
        P, C = P[keep], C[keep]
    lo, hi = np.percentile(P, [1, 99], axis=0)
    inside = np.all((P >= lo) & (P <= hi), axis=1)
    P, C = P[inside], C[inside] / 255.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    views = [(0, 1, 2, "from the front (x right, y up)"), (0, 2, 1, "from above (x right, the front at the bottom)")]
    for ax, (a, b, depth, title) in zip(axes, views):
        order = np.argsort(P[:, depth])           # the nearest points last, so they cover the ones behind
        ax.scatter(P[order, a], P[order, b], c=C[order], s=2.0, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].invert_yaxis()                        # looking down, the cameras' side (+z) is nearest the bottom edge
    fig.suptitle(f"dense.ply: {len(P):,} of its points", fontsize=10)
    plt.show()
    print(f"files in {out}: dense.ply and sparse.ply (MeshLab, CloudCompare, Blender), view.html, cameras.csv")


def write_cameras(path, photos, scene, rec, M=None):
    """cameras.csv: one row per photo with its original file, size, the focal length used and where it came from,
    and the camera centre (in the PLY's frame, turned by M as the PLY is) if it was placed."""
    local = {int(m): k for k, m in enumerate(scene["members"])}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo", "file", "width", "height", "focal_px", "focal_from", "placed", "cx", "cy", "cz"])
        for i, name in enumerate(photos["names"]):
            k = local.get(i)
            placed = k is not None and k in rec["cams"]
            C = viewer_frame(camera_center(*rec["cams"][k]), M)[0].tolist() if placed else ["", "", ""]
            w.writerow([f"{i:04d}.jpg", name, *photos["sizes"][i].tolist(), f"{photos['K'][i][0, 0]:.1f}",
                        "exif" if photos["from_exif"][i] else "guess", int(placed), *C])


def _without_i():
    """Run from a notebook with plain %run (not %run -i): the notebook's functions are out of sight."""
    ip = sys.modules.get("IPython")
    shell = ip.get_ipython() if ip is not None else None
    return shell is not None and sys.modules.get("__main__") is not shell.user_module


def _in_notebook():
    ip = sys.modules.get("IPython")
    return bool(ip is not None and ip.get_ipython() is not None)


def _shows_inline():
    """A Jupyter or Colab kernel draws plotly figures in the page; terminal IPython would start a web browser."""
    ip = sys.modules.get("IPython")
    shell = ip.get_ipython() if ip is not None else None
    return getattr(shell, "kernel", None) is not None


def _probe_stubs():
    """Call each of your functions once with empty arguments: one that is not written yet raises
    NotImplementedError(<its name>) at once, so it shows now, not after minutes of loading and matching. Any other
    outcome (an error from the empty arguments, most likely) is ignored."""
    main = sys.modules["__main__"]
    for name in USED:
        f = getattr(main, name)
        try:
            n_args = sum(p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                         for p in inspect.signature(f).parameters.values())
        except (TypeError, ValueError):
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                f(*[None] * n_args)
        except NotImplementedError as err:
            if str(err) in NAMES:
                raise
        except Exception:                         # empty arguments: an error is expected from a written function
            pass


def cap_pairs(new, rng):
    """At most MAX_NEW_PAIRS of the proposed pairs (M,2), a random choice in their order, as the notebook does."""
    return new if len(new) <= MAX_NEW_PAIRS else new[np.sort(rng.choice(len(new), MAX_NEW_PAIRS, replace=False))]


def reconstruct(scene, tree_edges):
    """run_sfm_skeleton, then register_rest; no start pair becomes a PhotoError with advice."""
    try:
        return register_rest(run_sfm_skeleton(scene, tree_edges), scene)
    except RuntimeError as err:
        if "no start pair" not in str(err):
            raise
        raise PhotoError(f"{err}. The photos overlap too little for a first pair with real depth: walk around the "
                         "place and take more photos in between, or try --top-k 20") from err


def bad_settings(top_k, rounds, min_inliers, long_side, focal, dense_images):
    """What is wrong with the settings, as a list of plain sentences (empty when they are fine)."""
    bad = [f"top_k (--top-k) must be at least 1, not {top_k}"] if top_k < 1 else []
    bad += [f"rounds (--rounds) must be 0 or more, not {rounds}"] if rounds < 0 else []
    bad += [f"min_inliers (--min-inliers) must be at least {MIN_INLIERS} (the 3D step keeps only pairs with "
            f"{MIN_INLIERS} matches), not {min_inliers}"] if min_inliers < MIN_INLIERS else []
    bad += [f"long_side (--long-side) must be at least {MIN_LONG_SIDE}, not {long_side}"
            ] if long_side < MIN_LONG_SIDE else []
    bad += [f"focal (--focal) must be positive, not {focal}"] if not focal > 0 else []
    bad += [f"dense_images (--dense-images) must be at least 1, not {dense_images}"] if dense_images < 1 else []
    return bad


def photos_to_cloud(source, out=None, reference=False, show=None, top_k=DEFAULT_TOP_K, rounds=DEFAULT_ROUNDS,
                    min_inliers=MIN_INLIERS, long_side=LONG_SIDE, focal=FOCAL, dense_images=DENSE_IMAGES, flip=False,
                    ignore_exif=False):
    """The whole pipeline on the photos in source (a folder, a zip, a direct link to a zip, a Drive folder link or a
    Drive zip link); the results go to out (default <name>_cloud). reference=True uses ref.<name> instead of your
    functions (yours come back afterwards). show: draw the 3D view inline (default: when run in a notebook). The other
    settings are the ones in the table at the top of this file; flip=True turns the saved cloud over; ignore_exif=True
    uses the focal guess for every photo. Returns dict(rome, scene, rec, dense, out, figure: the 3D view)."""
    source, out = os.fspath(source), (os.fspath(out) if out is not None else None)   # a pathlib.Path works too
    if not re.match(r"https?://", source):
        source = os.path.expanduser(source)       # ~ is the home folder here too, as in %run
    out = os.path.expanduser(out) if out is not None else None
    settings = dict(top_k=top_k, rounds=rounds, min_inliers=min_inliers, long_side=long_side, focal=focal,
                    dense_images=dense_images)
    if bad_settings(**settings):
        raise PhotoError("; ".join(bad_settings(**settings)))
    with reference_functions() if reference else contextlib.nullcontext():
        return _pipeline(source, out, _shows_inline() if show is None else show, reference, flip, ignore_exif,
                         **settings)


def _pipeline(source, out, show, reference, flip, ignore_exif, top_k, rounds, min_inliers, long_side, focal,
              dense_images):
    t_start = time.time()
    missing = [n for n in USED if not callable(getattr(sys.modules["__main__"], n, None))]
    if missing:
        how = ("You ran it without -i: use %run -i rome_from_folder.py ..., which lets it see your functions"
               if _without_i() else
               "Run the cells that define them, then run the script with %run -i (the -i lets it see them), "
               "or use the library versions (--reference)" if _in_notebook() else
               "Take them from your saved notebook with --notebook rome_in_a_day.ipynb, or use the library versions "
               "with --reference")
        raise NotDefinedYet(f"not defined yet: {', '.join(missing)}. {how}")
    name = source_name(source)
    if out and is_drive_link(source):            # a Drive link names nothing: the place is named after --out
        name = re.sub(r"_cloud$", "", os.path.basename(os.path.normpath(out))) or name
    out = out or f"{name}_cloud"
    is_zip = os.path.isfile(source) and zipfile.is_zipfile(source)
    is_link = bool(re.match(r"https?://", source)) or is_zip   # a link or a zip: unpacked into a temporary folder
    with open(source, "rb") if os.path.isfile(source) else contextlib.nullcontext() as fh:
        zip_like = fh is not None and (source.lower().endswith(".zip") or fh.read(2) == b"PK")
    if not is_link and zip_like:
        raise PhotoError(f"{source} is a damaged or incomplete zip (an upload cut short, say): upload it or zip the "
                         "photos again")
    if not is_link and not os.path.exists(source):
        raise PhotoError(f"no such file or folder: {source} (check the path; on Colab, upload it first, or mount "
                         "Drive for a path on it)")
    if not is_link and not os.path.isdir(source):
        raise PhotoError(f"{source} is neither a folder, a zip file nor a link")
    if not is_link and os.path.exists(os.path.join(source, MARKER)):
        raise PhotoError(f"{source} is an output folder of this script (resized copies, no EXIF): pass the folder "
                         "with your photos")
    real_out, real_src = (os.path.normcase(os.path.realpath(x)) for x in (out, source))   # links, and case on
    if not is_link and real_out == real_src:   # Windows (on macOS a case-only difference is caught by the check that
                                               # refuses an existing folder holding files that are not the script's)
        raise PhotoError(f"the output folder is the same folder as the photos ({source}): pick another --out")
    if not is_link and real_src.startswith(real_out + os.sep):
        raise PhotoError(f"the photos ({source}) are inside the output folder {out}, where the script writes its own "
                         "files (images/, for one): move the photos, or pick another --out")
    if real_out.startswith("/content/drive/") and not os.path.ismount("/content/drive"):
        raise PhotoError(f"--out {out} is on Google Drive, but Drive is not mounted, so the results would stay on this "
                         "session's disk and go when it ends: mount it first (from google.colab import drive; "
                         "drive.mount('/content/drive')), then run again")
    if os.path.exists(out) and not os.path.isdir(out):
        raise PhotoError(f"--out {out} is a file, not a folder: pick a new folder")
    if os.path.isdir(out) and set(os.listdir(out)) - JUNK and not os.path.exists(os.path.join(out, MARKER)):
        raise PhotoError(f"{out} already holds files that are not this script's: pick a new folder for --out (the "
                         "script writes images/, dense.ply and more there)")
    _probe_stubs()
    find_vocab()                                  # no crate: say so now, before any photo is downloaded or loaded
    log = []

    def say(msg):
        print(msg)
        log.append(msg)
    whose = "the library" if reference else "your"

    work = tempfile.mkdtemp(prefix="rome_from_folder_") if is_link else out   # a zip or link is unpacked here, on
    try:                                                                     # this machine (not on a mounted Drive)
        folder = fetch_source(source, work, say)  # nothing is written to out until the photos are found
        paths = list_photos(folder, skip=[out])
        others = len(list_photos(folder, skip=[out], extensions=OTHER_IMAGES))
        also = (f"{others} HEIC or AVIF files, which this script cannot read: export them as JPEG (on an "
                "iPhone: Settings > Camera > Formats > Most Compatible)" if others else "")
        if len(paths) < MIN_PHOTOS:
            raise PhotoError(f"found {len(paths)} photos (JPEG, PNG, WebP or TIFF) in {source}"
                             f"{'; it also holds ' + also if also else ''}; this part needs at least "
                             f"{MIN_PHOTOS} of one place")
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, MARKER), "w").close()   # a later run never reads this folder as photos
        for f in RESULTS:                         # the photos are found: this run replaces an earlier one's results
            if os.path.isfile(os.path.join(out, f)):   # (they must not sit next to this run's images)
                os.remove(os.path.join(out, f))
        shutil.rmtree(os.path.join(out, "images"), ignore_errors=True)
        try:
            result = _build(out, name, show, flip, ignore_exif, top_k, rounds, min_inliers, long_side, focal,
                            dense_images, paths, folder, also, say, log, whose, t_start)
        except BaseException as err:              # the summary says what happened this time (your own code too)
            why = str(err) if isinstance(err, PhotoError) else f"{type(err).__name__}: {err}"
            with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(log + [f"stopped: {why}"]) + "\n")
            raise
    finally:
        if is_link:
            shutil.rmtree(work, ignore_errors=True)   # the unpacked copy of a zip or link
    if show:                                      # after every file is written: an interrupted plot leaves them
        result["figure"].show()
    return result


def _build(out, name, show, flip, ignore_exif, top_k, rounds, min_inliers, long_side, focal, dense_images, paths,
           folder, also, say, log, whose, t_start):
    """The pipeline once the photos are found: load, words, pairs, groups, SfM, dense cloud, outputs."""
    if len(paths) > MANY_PHOTOS:
        print(f"note: {len(paths)} photos. More than {MANY_PHOTOS} works, but it takes a while")
    _read_rgb.cache_clear()                        # a rerun rewrites out/images: never reuse an older picture
    if also:
        say(f"note: left out {also}")
    photos = load_photos(paths, os.path.join(out, "images"), long_side, focal, root=folder, ignore_exif=ignore_exif)
    n = len(photos["paths"])
    if photos["skipped"]:
        names = [os.path.relpath(f, folder) for f in photos["skipped"][:3]]
        say(f"skipped {len(photos['skipped'])} files that are not readable photos, e.g. {names}")
    if n < MIN_PHOTOS:
        raise PhotoError(f"only {n} of {len(paths)} files are readable photos; at least {MIN_PHOTOS} are needed")
    say(f"{n} photos (long side at most {long_side} px); focal length from EXIF for {photos['from_exif'].sum()}, "
        f"guessed ({focal} x the long side) for the others")
    say(f"settings: top_k {top_k}, rounds {rounds}, min_inliers {min_inliers}, long_side {long_side}, focal {focal}, "
        f"dense_images {dense_images}{', flip' if flip else ''}{', ignore_exif' if ignore_exif else ''}")
    rome = build_crate(photos, find_vocab(), out)
    rome["groups"], rome["labels"][:] = [name], 0  # one place, named after the folder (no survey cameras for it)

    # 1 to 3 of the notebook: which photos to compare, checking them, and the groups
    T = yours("tfidf_matrix")(rome["offsets"], rome["words"], len(rome["vocab"]))
    idx, _ = yours("top_k_neighbours")(T, min(top_k, n - 1))
    pairs = neighbour_pairs(idx)
    t0 = time.time()
    feats = [_sift(_read_rgb(str(p)), N_MATCH) for p in photos["paths"]]
    n_inl, ok = verify_by_matching(pairs, feats, min_inliers)
    edges, weights = pairs[ok], n_inl[ok]
    tested = set(map(tuple, pairs.tolist()))
    rng = np.random.default_rng(0)
    for _ in range(rounds):
        new = cap_pairs(np.asarray(yours("query_expansion")(edges, tested), np.int64).reshape(-1, 2), rng)
        if len(new) == 0:
            break
        n_new, ok_new = verify_by_matching(new, feats, min_inliers)
        tested |= set(map(tuple, new.tolist()))
        edges, weights = np.concatenate([edges, new[ok_new]]), np.concatenate([weights, n_new[ok_new]])
    say(f"{len(tested):,} pairs checked with {whose} match_descriptors, {len(edges):,} of them real "
        f"({time.time() - t0:.0f} s)")
    comp = np.asarray(yours("connected_components")(n, edges), np.int64)
    sizes = np.bincount(comp, minlength=1)
    members, e_c, w_c = component_graph(comp, int(sizes.argmax()), edges, weights)
    say(f"largest group: {len(members)} of {n} photos")
    if len(members) < MIN_GROUP:
        raise PhotoError(f"only {len(members)} of {n} photos overlap with each other, and at least {MIN_GROUP} are "
                         "needed. Take more photos of the same place from nearby spots, each sharing about half of "
                         f"its view with the next, or try --top-k {max(2 * top_k, 20)}")

    # 4 of the notebook: the tree, matching, SfM and the dense cloud
    keep = maximum_spanning_tree(len(members), e_c, w_c)
    scene = landmark_scene(rome, members, e_c, w_c)
    rec = reconstruct(scene, e_c[keep])
    placed = len(rec["cams"])
    say(f"placed {placed} of {len(members)} photos: {len(rec['pts']):,} points, rms {rec['rms']:.2f} px")
    if placed < 10:
        say("warning: few photos were placed; the photos may overlap too little, or all be taken from one spot "
            "(turning without moving gives no depth)")
    dense = densify(rec, scene, n_images=min(dense_images, placed))

    M = viewer_rotation(rec["cams"], flip)
    n_dense = write_ply(os.path.join(out, "dense.ply"), viewer_frame(dense["pts"], M), dense["colors"])
    n_sparse = write_ply(os.path.join(out, "sparse.ply"), viewer_frame(rec["pts"], M), rec["colors"])
    write_cameras(os.path.join(out, "cameras.csv"), photos, scene, rec, M)
    fig = plot_reconstruction(rec, scene, dense=dense,
                              title=f"{scene['name']}: {placed} photos, {n_dense:,} points",
                              show=False)
    level_view(fig, M)                            # the view's up is the PLY's up (and --flip turns it too)
    fig.write_html(os.path.join(out, "view.html"), include_plotlyjs=True)
    say(f"dense cloud: {n_dense:,} points, sparse: {n_sparse:,}; {time.time() - t_start:.0f} s in all")
    say(f"wrote {out}: dense.ply, sparse.ply (MeshLab, CloudCompare, Blender), view.html, cameras.csv, summary.txt")
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    return dict(rome=rome, scene=scene, rec=rec, dense=dense, out=out, figure=fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a folder of photos, a zip of them, a direct link to a zip, a Drive folder link or "
                                   "a Drive zip link")
    ap.add_argument("--out", help="output folder (default: <name>_cloud)")
    how = ap.add_mutually_exclusive_group()
    how.add_argument("--notebook", help="take your functions from this saved notebook (rome_in_a_day.ipynb)")
    how.add_argument("--reference", action="store_true", help="use the library versions ref.<name>")
    tune = ap.add_argument_group("settings (see the table above)")
    tune.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                      help=f"neighbours per photo to check (default {DEFAULT_TOP_K})")
    tune.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                      help=f"rounds of query expansion (default {DEFAULT_ROUNDS})")
    tune.add_argument("--min-inliers", type=int, default=MIN_INLIERS,
                      help=f"matches on one F for a real pair (default {MIN_INLIERS})")
    tune.add_argument("--long-side", type=int, default=LONG_SIDE, help=f"photo size in px (default {LONG_SIDE})")
    tune.add_argument("--focal", type=float, default=FOCAL,
                      help=f"focal length guess without EXIF, in long sides (default {FOCAL})")
    tune.add_argument("--dense-images", type=int, default=DENSE_IMAGES,
                      help=f"photos the dense cloud uses (default {DENSE_IMAGES})")
    tune.add_argument("--flip", action="store_true", help="turn the saved cloud over (if it comes out upside down)")
    tune.add_argument("--ignore-exif", action="store_true", help="use the --focal guess for every photo, EXIF or not")
    args = ap.parse_args(argv)
    bad = bad_settings(args.top_k, args.rounds, args.min_inliers, args.long_side, args.focal, args.dense_images)
    if bad:
        ap.error("; ".join(bad))
    if args.notebook and _in_notebook():
        print("stopped: in a notebook your functions are already here: leave out --notebook (it would replace them "
              "with the versions saved in the file)")
        return 1
    try:
        if args.notebook:
            use_notebook(args.notebook)
            main_ns = sys.modules["__main__"]
            missing = [n for n in USED if not callable(getattr(main_ns, n, None))]
            if missing:
                print(f"stopped: the notebook {args.notebook} does not define {', '.join(missing)}: write them there "
                      "(or borrow them, as in tfidf_matrix = ref.tfidf_matrix), save it, and run again")
                return 1
        photos_to_cloud(args.source, args.out, reference=args.reference, top_k=args.top_k, rounds=args.rounds,
                        min_inliers=args.min_inliers, long_side=args.long_side, focal=args.focal,
                        dense_images=args.dense_images, flip=args.flip, ignore_exif=args.ignore_exif)
    except (PhotoError, NotDefinedYet) as err:
        print(f"stopped: {err}")
        return 1
    except NameError as err:
        if not args.notebook or getattr(err, "name", None) not in notebook_names(args.notebook):
            raise                                 # a typo in your code, say: the traceback shows where
        where = [f.name for f in traceback.extract_tb(err.__traceback__) if not f.name.startswith("<")][-1]
        print(f"stopped: {err} in {where}: it reads a notebook variable that --notebook does not set up (it skips "
              "loading the crate and running the pipeline). Run the script in the notebook with %run -i, or pass "
              "that value as an argument")
        return 1
    except NotImplementedError as err:            # the notebook's unwritten functions raise NotImplementedError(name)
        if str(err) not in NAMES:
            raise                                 # a real error in your code (a SciPy limit, say): keep the traceback
        how = ("Write it in the notebook, save the notebook, and run again" if args.notebook else
               "Write it in the notebook and run its cell")
        print(f"stopped: not written yet: {err}. {how}, or use the library versions with --reference")
        return 1
    return 0


def run_as_script(argv):
    """main, as the command line runs it; in a notebook (%run -i) a message is enough: no SystemExit traceback."""
    try:
        code = main(argv)
    except SystemExit as err:                     # argparse refused an option and already said why
        if not _in_notebook():
            raise
        return err.code
    if code and not _in_notebook():
        sys.exit(code)
    return code


if __name__ == "__main__":
    run_as_script(sys.argv[1:])
