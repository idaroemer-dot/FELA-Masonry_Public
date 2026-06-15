import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


THESIS_COLORS = {
    "rose_light": np.array([233, 213, 215]) / 255,
    "rose": np.array([200, 121, 128]) / 255,
    "dark_rose": np.array([150, 80, 85]) / 255,
    "grey": np.array([100, 100, 100]) / 255,
    "dark": np.array([30, 30, 30]) / 255,
    "lightgrey": np.array([150, 150, 150]) / 255,
    "blue": np.array([0, 9, 255]) / 255,
    "red": np.array([255, 0, 0]) / 255,
}


def set_thesis_plot_style(use_latex=False):
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "0.20",
        "axes.labelcolor": "0.10",
        "xtick.color": "0.10",
        "ytick.color": "0.10",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.color": "0.85",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.borderaxespad": 0.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": use_latex,
    })


def save_figure(fig, filename, folder="figures", formats=("pdf", "png")):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(folder / f"{filename}.{fmt}")
