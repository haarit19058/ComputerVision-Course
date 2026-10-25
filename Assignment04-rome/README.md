# Rome in a Day: A City Hidden in Photos

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pikabell/es666-rome-in-a-day/blob/main/rome_in_a_day.ipynb)

https://github.com/user-attachments/assets/d8983818-e28e-4347-8e54-d8e5f565018a

The 2 minute teaser, also [in HD on YouTube](https://youtu.be/2wWmkO6Hl9w).

ES666 assignment: Structure from Motion at city scale. You write 9 functions that sort about 9,000 tourist photos of
Rome, match them and rebuild landmarks in 3D, following *Building Rome in a Day* (Agarwal et al., 2009). Everything
happens in one notebook, `rome_in_a_day.ipynb`; the rest is in the `toolbox/` folder next to it.

## On Google Colab

Click **Open in Colab** above, save a copy to your Drive so your work is kept (File > Save a copy in Drive), and run
the cells from the top. The first code cell fetches the toolbox and the photos (about 580 MB) from this repository.

## On your own computer

```bash
git clone https://github.com/pikabell/es666-rome-in-a-day.git
cd es666-rome-in-a-day
uv sync
uv run jupyter lab       # then open rome_in_a_day.ipynb
```

Without uv: `python -m venv .venv`, activate it, `pip install -r requirements.txt`, then `jupyter lab`. The notebook's
pictures load from this repository on GitHub, so they need a connection.

## What you write

`tfidf_matrix`, `top_k_neighbours`, `match_descriptors`, `query_expansion`, `connected_components`,
`homography_dlt`, `homography_ratio`, `triangulate_points`, `umeyama`. Each has a check cell and a library
version in `ref` to compare with (or borrow, if you are stuck).

## Your own photos (Chapter 7)

Task 10 runs your functions on photos you take yourself (50 or more of one place) and shows the 3D point cloud; it
starts with a Taj Mahal example. It uses `rome_from_folder.py`, which is in this repository.

## Credits and licence

Plates by G. B. Piranesi and G. B. Nolli, public domain (see `CREDITS.md`). Photos are CC BY, CC BY-SA, CC0 or public
domain; authors are listed in each photo collection's `attribution.csv`. Code: MIT (`LICENSE`). Text and diagrams:
CC BY 4.0. Teaser: music by Kevin MacLeod (incompetech.com), CC BY 4.0; characters: Open Peeps by Pablo Stanley, CC0; voices by
Breeze TTS 2, for non-commercial use.

Thanks to Claude for helping :)
