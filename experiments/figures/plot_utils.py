"""Shared style for all paper figures.

Every figure imports from here so the paper reads as one visual system.
Palette is Okabe-Ito (colour-blind safe); every series must additionally
differ in marker or line style so the figures survive grayscale printing.
"""

import matplotlib as mpl

# Okabe-Ito
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#999999"
BLACK = "#000000"

# The canvas width must equal the final width in the LaTeX layout; otherwise
# \includegraphics rescales the figure and its text with it (a 7.0in figure
# shown at 6.30in turns 8pt labels into 7.2pt, below the 9pt minimum).
# Two-column a4paper layout, margin=2.5cm, columnsep=0.6cm:
FULL_W = (21.0 - 5.0) / 2.54            # full text width, 6.30in
COL_W = (21.0 - 5.0 - 0.6) / 2 / 2.54   # single column, 3.03in
FS = 9                                   # minimum font size inside figures


def setup() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,  # embed TrueType so text stays editable/searchable
    })


def save(fig, results_dir, name: str, bottom: float = 0.0,
         top: float = 1.0) -> None:
    # bbox_inches="tight" is not used: it crops to content, so the output width
    # would no longer equal figsize. tight_layout keeps the canvas size and pulls
    # legends and axis labels back inside it instead of cropping them.
    # bottom: fraction of the canvas reserved for a fig.legend placed below the
    # axes; with a fixed canvas it has to be reserved explicitly. top: the same
    # for a fig.legend placed above the axes (upper edge of the axes area).
    fig.tight_layout(pad=0.3, rect=[0, bottom, 1, top])
    for ext in ("pdf", "png"):
        fig.savefig(results_dir / "figures" / f"{name}.{ext}")
    print(f"  wrote figures/{name}.pdf .png")
