"""One quiet style for every picture, like the atlas plates: ink on paper, one red accent, muted colours for the
landmarks. Importing the toolbox sets it for matplotlib; the 3D views use the same colours."""

import matplotlib
import matplotlib.pyplot as plt

INK, INK2, PAPER, RULE = "#2f2620", "#7d7064", "#f7f2e7", "#d9cfbd"     # ink, thinned ink, paper, faint rule
RED, VERD, OCHRE = "#a33b2b", "#4f7d6d", "#b08a3e"                     # cinnabar (the key thing), verdigris, ochre
GREEN, GREY = VERD, INK2                                                # right, quiet
FAINT = "#b3a68f"     # minor items on paper (small components): lighter than INK2, still visible against PAPER
# landmarks or components 0 .. 9: verdigris, ochre, slate, brick, olive, plum, sky, rose, deep green, umber. Muted like
# the plates, and picked so that any two differ clearly (CIEDE2000 above 15, also against INK2 and FAINT)
PALETTE = ["#3f8a78", "#c39a3a", "#3c5a92", "#9a4630", "#879836", "#7e4f8e", "#62a8c4", "#cf8597", "#2d4b3a",
           "#5a3a2a"]
INK_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("ink", [PAPER, "#cdb99a", "#8a5a3c", INK])
# similarity: paper, ochre, orange, cinnabar, dark ink; lightness falls steadily, so higher always reads darker
SIM_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "similarity", [PAPER, "#ecd29a", "#d99a4e", "#c0612f", RED, "#5e1f17", INK])
SIM_CMAP.set_bad(PAPER)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "normal",
    "axes.facecolor": PAPER, "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.prop_cycle": plt.cycler(color=[RED, INK, VERD, OCHRE]), "xtick.color": INK2, "ytick.color": INK2,
    "grid.color": RULE, "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.2,
    "image.cmap": "gray", "figure.dpi": 80,
})
