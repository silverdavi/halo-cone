"""Shared figure style for the halo-cone paper (Rules/FIGURES.md).

One source of truth for palette, fonts and saving. Figure scripts import from
here and never set rcParams or hex inline. Colour encodes a variable:
halo class (isotropic vs any), the neutrino target, the SHM reference.
Pure-Python constants live at module top so the SVG compose step can import
them without matplotlib.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = Path(os.environ.get("HALO_FIGDIR", ROOT / "paper" / "figs"))

FONT_STACK = ["TeX Gyre Heros", "Helvetica Neue", "Arial", "DejaVu Sans"]
FONT_CSS = ", ".join(FONT_STACK + ["sans-serif"])

# warm archival palette, light background
COLORS = {
    "ink": "#2b2b2b",
    "slate": "#6b6b6b",
    "mist": "#b8b8b8",
    "shade": "#ece6da",      # mass window, where the exact mimic exists
    "cloud": "#e3eaf1",      # kinematic band that is not the mass window
    "terracotta": "#b5553a",
    "dustyblue": "#5b7fa6",
    "moss": "#6f8a4f",
    "gold": "#9a7010",       # darkened: #cf9b3f is ~2.5:1 on white
    "violet": "#7b5e9a",
}

# semantic roles -> palette names
ROLES = {
    "isotropic": "terracotta",   # class I, the paper's subject
    "anyhalo": "ink",            # class S, the unconstrained reference
    "neutrino": "dustyblue",     # the 8B target
    "shm": "gold",               # Standard Halo Model reference
    "edge": "slate",             # analytic edges, reference lines
    "window": "shade",           # region where the exact mimic exists
    "second": "moss",            # second target (CaWO4)
}

SEQUENTIAL = "cividis"           # ordered strata; sampled only on the dark half

# figure widths in inches for a two-column APS page; HALO_FIG_W2 sets the full width
# for a single-column journal (JCAP text width: 440 pt = 6.09 in)
W1 = 3.375
W2 = float(os.environ.get("HALO_FIG_W2", 7.0))


def c(name: str) -> str:
    return COLORS[name]


def role(name: str) -> str:
    return COLORS[ROLES[name]]


def set_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "mathtext.fontset": "stixsans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "legend.title_fontsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "figure.dpi": 150,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.02,
        "text.color": COLORS["ink"],
    })


def _srgb_to_lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    r, g, b = (_srgb_to_lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(hex_color: str, against: str = "#ffffff") -> float:
    """WCAG contrast ratio. Lines need about 3:1 on the page."""
    lighter = max(luminance(hex_color), luminance(against))
    darker = min(luminance(hex_color), luminance(against))
    return (lighter + 0.05) / (darker + 0.05)


def mix(a: str, b: str, t: float) -> str:
    """t=0 is a, t=1 is b."""
    def rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    A, B = rgb(a), rgb(b)
    out = tuple(round(A[i] * (1 - t) + B[i] * t) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*out)


def sequential(n: int):
    """n ordered colours, light to dark. The light end is truncated so it
    stays above 3:1 on white (cividis yellow does not)."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(SEQUENTIAL)
    return [cmap(0.45 + 0.55 * i / max(n - 1, 1)) for i in range(n)]


def role_ramp(role_name: str, n: int, on: str = "#ffffff", dark: float = 0.0) -> list:
    """n colours of one semantic role, light to dark.

    The light end is the palest mix with white that still clears 3:1
    against `on`. `dark` mixes the deep end toward ink so neighbours
    separate; 0 keeps the role colour as the darkest swatch.
    """
    base = role(role_name)
    deep = mix(base, COLORS["ink"], dark) if dark else base
    lo = 0.55
    while contrast(mix("#ffffff", base, lo), on) < 3.05 and lo < 0.98:
        lo += 0.03
    if n == 1:
        return [deep]
    light = mix("#ffffff", base, lo)
    return [mix(light, deep, i / (n - 1)) for i in range(n)]


def panel_label(ax, letter: str, dx: float = -0.01, dy: float = 1.05) -> None:
    """Bold lower-case panel letter, top-left, Nature style."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left", color=COLORS["ink"],
            clip_on=False)


def legend_below(ax, ncol: int = 2, y: float = -0.28, **kw):
    """Legend under the axes, never floating in the data."""
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                     borderaxespad=0.0, columnspacing=1.2, handletextpad=0.5, **kw)


def save_fig(fig, name: str) -> None:
    """Manuscript PDF plus 300-dpi review PNG, at the figure's exact size.

    bbox is not tightened: LaTeX scales the PDF to the column, and a tight
    crop would shrink the 7–8 pt type below print size.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches=None, facecolor="white")
    fig.savefig(FIGDIR / f"{name}.png", dpi=300, bbox_inches=None, facecolor="white")
    print(f"saved {FIGDIR}/{name}.pdf and .png")
