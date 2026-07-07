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

FULL_W = 7.0   # double-column figure width (inches)
COL_W = 3.4    # single-column figure width


def setup() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
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


def save(fig, results_dir, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(results_dir / "figures" / f"{name}.{ext}", bbox_inches="tight")
    print(f"  wrote figures/{name}.pdf .png")
