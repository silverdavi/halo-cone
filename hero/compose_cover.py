"""Compose the cover (8.5 x 11 in, 300 dpi) and a 16:9 slide from the Cycles renders.

Every mark on the scene comes from a point projected through the render camera
(hero/out/<view>_points.json): the mass scale, the window edges, and the dashed
footprint of the any-halo fog (every mass >= m_lo), which is drawn here and never
rendered. Numbers are read from paper/numbers.tex and paper/macros.tex.

  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib .venv/bin/python hero/compose_cover.py [cover wide]
Writes hero/cover.png (2550 x 3300), hero/cover_16x9.png (3840 x 2160), and their .svg.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("FONTCONFIG_FILE", str(ROOT / "viz" / "fonts.conf"))
for _p in ("/opt/homebrew/lib/libcairo.2.dylib", "/usr/local/lib/libcairo.2.dylib"):
    try:
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        break
    except OSError:
        continue

import cairosvg  # noqa: E402
import drawsvg as draw  # noqa: E402

sys.path.insert(0, str(ROOT / "config"))
from figstyle import COLORS as C, FONT_CSS  # noqa: E402

OUT = HERE / "out"
GOLD = C["gold"]
PALE = "#ffffff"
MIST = C["mist"]
NIGHT = "#070b14"   # the render's sky colour, sampled


def tex_number(name: str) -> str:
    for f in ("numbers.tex", "macros.tex"):
        m = re.search(r"\\newcommand\{\\" + name + r"\}\{([-0-9.]+)\}", (ROOT / "paper" / f).read_text())
        if m:
            return m.group(1)
    raise KeyError(name)


N = {k: tex_number(k) for k in ("nMlo", "nMhi", "nEthrXe", "nVE", "nVesc")}
WINDOW = f"{N['nMlo']}\u2013{N['nMhi']} GeV"
B8 = "\u2078B"
PAPER_TITLE = (f"Isotropy, not the Maxwellian, decides where solar {B8} neutrinos "
               "can exactly mimic dark matter in xenon")


def text(d, s, x, y, size, fill=PALE, anchor="start", weight="normal", style="normal", spacing=0, opacity=1.0,
         halo=False):
    if halo:   # dark copy underneath so a label stays legible over the fog's glow
        d.append(draw.Text(s, size, x, y, font_family=FONT_CSS, fill=NIGHT, stroke=NIGHT, stroke_width=size * 0.13,
                           stroke_linejoin="round", stroke_opacity=0.5, fill_opacity=0.5, text_anchor=anchor,
                           font_weight=weight, font_style=style, letter_spacing=spacing))
    d.append(draw.Text(s, size, x, y, font_family=FONT_CSS, fill=fill, text_anchor=anchor,
                       font_weight=weight, font_style=style, letter_spacing=spacing, fill_opacity=opacity))


def lerp(a, b, t):
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def scene(d, P, s, label_masses, fs):
    """Marks on the floor and at the fog: scale labels, window edges, any-halo footprint."""
    # any-halo footprint: the floor rectangle m >= m_lo, |y| <= depth, dashed and faintly filled
    quad = [P["ghost_lo_f0"], P["ghost_far_f0"], P["ghost_far_b0"], P["ghost_lo_b0"]]
    flat = [c for p in quad for c in p[:2]]
    d.append(draw.Lines(*flat, close=True, fill=PALE, fill_opacity=0.05, stroke=PALE, stroke_opacity=0.75,
                        stroke_width=3.0 * s, stroke_dasharray=f"{14 * s},{11 * s}"))
    # the isotropic window on the baseline
    lo, hi = P["axis_lo"], P["axis_hi"]
    d.append(draw.Line(*lo[:2], *hi[:2], stroke=GOLD, stroke_width=9 * s, stroke_linecap="butt"))
    for p, lab, anc, dx in ((lo, N["nMlo"], "end", 0.35), (hi, N["nMhi"], "start", -0.35)):
        d.append(draw.Line(p[0], p[1] - 16 * s, p[0], p[1] + 16 * s, stroke=GOLD, stroke_width=4 * s))
        text(d, lab, p[0] + dx * fs, p[1] + 62 * s, fs, fill=GOLD, anchor=anc, weight="bold", halo=True)
    for m in label_masses:
        p = P[f"tick{m}"]
        text(d, str(m), p[0], p[1] + 50 * s, fs * 0.9, fill=MIST, anchor="middle")
    return quad


def cover():
    meta = json.loads((OUT / "cover_points.json").read_text())
    W, H = meta["size"]
    P = meta["points"]
    d = draw.Drawing(W, H, origin=(0, 0))
    d.append(draw.Image(0, 0, W, H, path=str(OUT / "cover_render.png"), embed=True))
    # a faint darkening behind the footer only
    grad = draw.LinearGradient(0, H - 520, 0, H)
    grad.add_stop(0, "#000000", 0)
    grad.add_stop(1, "#000000", 0.55)
    d.append(draw.Rectangle(0, H - 520, W, 520, fill=grad))

    s = 1.0
    M = 170
    scene(d, P, s, label_masses=[4, 5, 10, 20, 30, 50, 100], fs=48)
    text(d, "dark-matter mass (GeV)", W - M, P["tick20"][1] + 140, 46, fill=MIST, anchor="end")

    # labels at the fog: window and the two classes
    a, b = P["win_hi_b0"], P["win_hi_bh"]
    e = lerp(a, b, 0.52)
    lx = e[0] + 150
    d.append(draw.Line(e[0] + 40, e[1], lx - 20, e[1], stroke=GOLD, stroke_width=4))
    text(d, WINDOW, lx, e[1] + 40, 118, fill=GOLD, weight="bold")
    text(d, "isotropic halos", lx, e[1] + 112, 54, fill=PALE)
    text(d, "exact mimicry only here", lx, e[1] + 176, 54, fill=MIST)

    q = P["ghost_far_b0"]
    g0 = P["ghost_lo_b0"]
    gx = W - M
    gy = lerp(g0, q, (gx - g0[0]) / (q[0] - g0[0]))[1] - 40
    text(d, "any halo: every mass above " + N["nMlo"] + " GeV", gx, gy, 46, fill=PALE, anchor="end", opacity=0.9)

    # masthead block
    text(d, f"SOLAR {B8} NEUTRINOS  \u00b7  DARK MATTER  \u00b7  XENON", M, 250, 46, fill=GOLD, weight="bold", spacing=6)
    text(d, "Isotropy confines", M, 440, 168, weight="bold")
    text(d, "the neutrino fog", M, 625, 168, weight="bold")
    sub = [f"If the halo can be anything, solar {B8} neutrinos can exactly mimic",
           f"dark matter in xenon at every mass above {N['nMlo']} GeV. If it is",
           f"isotropic, only between {N['nMlo']} and {N['nMhi']} GeV."]
    for i, line in enumerate(sub):
        text(d, line, M, 760 + i * 72, 56, fill="#d9dde6")

    text(d, PAPER_TITLE, M, H - 190, 40, fill=PALE, style="italic", opacity=0.92)
    text(d, f"Ideal xenon detector, {N['nEthrXe']} keV threshold; Earth speed {N['nVE']} km/s, "
            f"Galactic escape speed {N['nVesc']} km/s", M, H - 128, 36, fill=MIST)
    return d, (W, H), "cover"


def wide():
    meta = json.loads((OUT / "wide_points.json").read_text())
    W, H = meta["size"]
    P = meta["points"]
    d = draw.Drawing(W, H, origin=(0, 0))
    d.append(draw.Image(0, 0, W, H, path=str(OUT / "wide_render.png"), embed=True))
    s = 1.0
    scene(d, P, s, label_masses=[3, 4, 5, 10, 20, 30, 50, 100], fs=46)
    text(d, "dark-matter mass (GeV)", P["tick100"][0], P["tick20"][1] + 130, 44, fill=MIST, anchor="end")
    q, g0 = P["ghost_far_b0"], P["ghost_lo_b0"]
    gx = P["tick100"][0]
    gy = lerp(g0, q, (gx - g0[0]) / (q[0] - g0[0]))[1] - 36
    text(d, "any halo: every mass above " + N["nMlo"] + " GeV", gx, gy, 44, fill=PALE, anchor="end", opacity=0.9)

    a, b = P["win_lo_f0"], P["win_lo_fh"]
    e = lerp(a, b, 0.52)
    lx = e[0] - 150
    d.append(draw.Line(e[0] - 40, e[1], lx + 20, e[1], stroke=GOLD, stroke_width=4))
    text(d, WINDOW, lx, e[1] + 36, 104, fill=GOLD, weight="bold", anchor="end")
    text(d, "isotropic halos", lx, e[1] + 104, 50, fill=PALE, anchor="end")
    text(d, "exact mimicry only here", lx, e[1] + 164, 50, fill=MIST, anchor="end")

    X0 = 2060
    text(d, f"SOLAR {B8} NEUTRINOS  \u00b7  DARK MATTER  \u00b7  XENON", X0, 300, 42, fill=GOLD, weight="bold", spacing=6)
    text(d, "Isotropy confines", X0, 480, 150, weight="bold")
    text(d, "the neutrino fog", X0, 645, 150, weight="bold")
    sub = [f"If the halo can be anything, solar {B8} neutrinos can",
           f"exactly mimic dark matter in xenon at every mass",
           f"above {N['nMlo']} GeV. If it is isotropic, only between",
           f"{N['nMlo']} and {N['nMhi']} GeV."]
    for i, line in enumerate(sub):
        text(d, line, X0, 780 + i * 66, 52, fill="#d9dde6")
    text(d, f"Ideal xenon detector, {N['nEthrXe']} keV threshold", X0, 1100, 38, fill=MIST)
    return d, (W, H), "cover_16x9"


def main(views):
    for v in views:
        d, (W, H), stem = {"cover": cover, "wide": wide}[v]()
        svg = d.as_svg()
        (HERE / f"{stem}.svg").write_text(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(HERE / f"{stem}.png"),
                         output_width=int(W), output_height=int(H))
        print("wrote", HERE / f"{stem}.png")


if __name__ == "__main__":
    main(sys.argv[1:] or ["cover", "wide"])
