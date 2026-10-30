"""Figure 1: geometry of a boosted isotropic shell (a-c) and its halo function (d).

Lab-frame velocity space, 1 unit = 100 km/s. The shell of Galactic speed u is the
sphere |v + v_E| = u about C = -v_E; the detector is at rest at O. Points of the
shell with |v| > w contribute to eta(w). Both spheres are cut at the plane z = 0
and only the lower halves are drawn.

Everything is exact vector geometry through one pinhole camera: flat fills split
exactly at the circle |v| = w, painter-ordered facets, outlines and the circle
|v| = w as lines, hidden parts of lines dashed. No lighting, no shadows.

  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib .venv/bin/python viz/compose_shell.py
Writes paper/figs/shell.pdf (vector) and shell.png (300 dpi at 7 in).
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("FONTCONFIG_FILE", str(HERE / "fonts.conf"))
for _p in ("/opt/homebrew/lib/libcairo.2.dylib", "/usr/local/lib/libcairo.2.dylib"):
    try:
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        break
    except OSError:
        continue

import cairosvg  # noqa: E402
import drawsvg as draw  # noqa: E402

sys.path.insert(0, str(ROOT / "config"))
from figstyle import COLORS as C, FONT_CSS, FIGDIR  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

UNITS_PER_IN = 300
W = 7.0 * UNITS_PER_IN           # full text width
PANEL_W = 500
PANEL_H = 400
TOP = 92                          # room for panel letters and regime lines
H = TOP + PANEL_H + 40
D_X = 3 * PANEL_W + 40            # panel d origin

# HALO_SHELL_FONT_SCALE enlarges the type when the figure is printed narrower than 7 in
# (1.146 keeps 7.2/6.2 pt at the 440 pt JCAP text width)
_FONT_SCALE = float(os.environ.get("HALO_SHELL_FONT_SCALE", 1.0))
FS = round(30 * _FONT_SCALE)      # 7.2 pt at 300 units/in
FS_SMALL = round(26 * _FONT_SCALE)
D_W = W - D_X - 10 - round(300 * (_FONT_SCALE - 1))   # larger type needs a right margin
INK = C["ink"]

W_VALUES = (100, 400, 700)
U, VE = 400.0, 232.0
S = 0.01                          # units per km/s
CPOS = np.array([-VE * S, 0.0, 0.0])


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def mix(a, b, t):
    """Colour t of the way from palette colour a to b (hex in, hex out)."""
    ca = hex_rgb(C[a] if a in C else a)
    cb = hex_rgb(C[b] if b in C else b)
    r = np.round(ca + (cb - ca) * t).astype(int)
    return "#%02x%02x%02x" % tuple(r)


TERRA = C["terracotta"]
TERRA_OUT = mix("terracotta", "#ffffff", 0.32)
NEUT = mix("mist", "#ffffff", 0.50)
NEUT_OUT = mix("mist", "#ffffff", 0.74)
BLUE = C["dustyblue"]


# ---------------------------------------------------------------- camera

class Camera:
    def __init__(self, loc, target, lens=55.0, sensor=36.0, size=(PANEL_W, PANEL_H), shift=(0.0, 0.0)):
        self.loc = np.asarray(loc, float)
        f = np.asarray(target, float) - self.loc
        self.f = f / np.linalg.norm(f)
        r = np.cross(self.f, [0, 0, 1.0])
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.f)
        self.k = lens / sensor * size[0]
        self.cx = size[0] / 2 + shift[0]
        self.cy = size[1] / 2 + shift[1]

    def __call__(self, p):
        d = np.asarray(p, float) - self.loc
        zc = d @ self.f
        return np.stack([self.cx + self.k * (d @ self.r) / zc,
                         self.cy - self.k * (d @ self.u) / zc], axis=-1)

    def depth(self, p):
        return np.linalg.norm(np.asarray(p, float) - self.loc, axis=-1)


CAM = Camera(loc=(-1.2, -15.0, 9.2), target=(-1.25, 0.0, -1.7), lens=43.0)


# ---------------------------------------------------------------- geometry

def alpha_b(w):
    """Polar angle about +x (from C) of the circle |v| = w on the shell; None if absent."""
    a, u = VE * S, U * S
    c = (a * a + u * u - (w * S) ** 2) / (2 * a * u)
    return None if abs(c) >= 1 else float(np.arccos(c))


def shell_point(al, be):
    u = U * S
    return CPOS + u * np.stack([np.cos(al), np.sin(al) * np.cos(be), np.sin(al) * np.sin(be)], axis=-1)


def occluded(pts, eps=1e-4):
    """True where the segment camera->point crosses the lower half of the shell before the point."""
    o = CAM.loc
    d = pts - o
    oc = o - CPOS
    A = np.einsum("ij,ij->i", d, d)
    B = 2 * d @ oc
    Cq = oc @ oc - (U * S) ** 2
    disc = B * B - 4 * A * Cq
    hit = np.zeros(len(pts), bool)
    ok = disc > 0
    sq = np.sqrt(np.where(ok, disc, 0))
    for sgn in (-1, 1):
        t = (-B + sgn * sq) / (2 * A)
        z = o[2] + t * d[:, 2]
        hit |= ok & (t > eps) & (t < 1 - eps) & (z <= 1e-9)
    return hit


def silhouette(center, r, n=6001):
    """Lower half (z <= 0) of the camera's contact circle on a sphere."""
    center = np.asarray(center, float)
    v = CAM.loc - center
    dist = np.linalg.norm(v)
    nvec = v / dist
    c0 = center + nvec * r * r / dist
    rc = r * np.sqrt(1 - (r / dist) ** 2)
    e1 = np.cross(nvec, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nvec, e1)
    t = np.linspace(0, 2 * np.pi, n)
    p = c0 + rc * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))
    return p[p[:, 2] <= 0]


def rim(center, r, n=721):
    t = np.linspace(0, 2 * np.pi, n)
    return np.asarray(center, float) + r * np.stack([np.cos(t), np.sin(t), 0 * t], axis=-1)


# ---------------------------------------------------------------- drawing helpers

def text(d, s, x, y, size=FS, anchor="start", fill=INK, weight="normal", style="normal"):
    d.append(draw.Text(s, size, x, y, font_family=FONT_CSS, fill=fill, text_anchor=anchor,
                       font_weight=weight, font_style=style, dominant_baseline="middle"))


def math(d, parts, x, y, size=FS, fill=INK, anchor="start", halo=False):
    """Inline math as one <text> with <tspan> runs: parts = [(text, 'it'|'rm'|'sub'), ...]."""
    # cairosvg mislays multi-tspan text with a non-start anchor; shift x by an estimated width instead
    if anchor != "start":
        est = sum(len(s) * size * (0.72 if k == "sub" else 1.0) * 0.53 for s, k in parts)
        x = x - est / 2 if anchor == "middle" else x - est
    # cairosvg ignores paint-order, so a halo is a white-stroked copy drawn first
    styles = [dict(stroke="white", stroke_width=8, stroke_linejoin="round", fill="white")] if halo else []
    for extra in styles + [dict(fill=fill)]:
        t = draw.Text("", size, x, y, font_family=FONT_CSS, text_anchor="start",
                      dominant_baseline="middle", **extra)
        pending_dy = 0.0
        for s, kind in parts:
            kw = {}
            if kind == "it":
                kw["font_style"] = "italic"
            if kind == "sub":
                kw["font_size"] = size * 0.72
                kw["dy"] = size * 0.22 - pending_dy
                pending_dy = size * 0.22
            else:
                if pending_dy:
                    kw["dy"] = -pending_dy
                    pending_dy = 0.0
            t.append(draw.TSpan(s, **kw))
        d.append(t)


def leader(d, x0, y0, x1, y1, fill=INK):
    d.append(draw.Line(x0, y0, x1, y1, stroke="white", stroke_width=6, stroke_linecap="round"))
    d.append(draw.Line(x0, y0, x1, y1, stroke=fill, stroke_width=2.2))
    d.append(draw.Circle(x1, y1, 6, fill=fill, stroke="white", stroke_width=2))


def polyline(g, xy, color, width, dash=None, casing=0.0, g_case=None):
    """Polyline; its white casing goes to g_case (drawn under every line) when given."""
    if len(xy) < 2:
        return
    flat = [float(c) for p in xy for c in p]
    if casing:
        (g if g_case is None else g_case).append(
            draw.Lines(*flat, fill="none", stroke="white", stroke_width=width + casing,
                       stroke_linejoin="round", stroke_linecap="round"))
    kw = dict(stroke_dasharray=dash) if dash else {}
    g.append(draw.Lines(*flat, fill="none", stroke=color, stroke_width=width,
                        stroke_linejoin="round", stroke_linecap="butt" if dash else "round", **kw))


def hidden_line(g, pts, ox, oy, color, width, casing=0.0, dash="9,7", draw_hidden=True, g_case=None):
    """Draw a 3D curve: solid where visible, dashed where the shell hides it."""
    if len(pts) < 2:
        return
    occ = occluded(pts)
    xy = CAM(pts) + [ox, oy]
    start = 0
    for i in range(1, len(pts) + 1):
        if i == len(pts) or occ[i] != occ[start]:
            seg = xy[max(start - 1, 0):i]
            if occ[start]:
                if draw_hidden:
                    polyline(g, seg, color, width * 0.75, dash=dash)
            else:
                polyline(g, seg, color, width, casing=casing, g_case=g_case)
            start = i


def arrow2d(g, p0, p1, color, width=3.0, head=20.0):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-t[1], t[0]])
    base = p1 - t * head
    g.append(draw.Line(*p0, *base, stroke=color, stroke_width=width))
    tri = [*p1, *(base + n * head * 0.42), *(base - n * head * 0.42)]
    g.append(draw.Lines(*tri, close=True, fill=color, stroke=color, stroke_width=1, stroke_linejoin="round"))


# ---------------------------------------------------------------- panels a-c

def shell_facets(w, na=72, nb=64):
    """Facets of the lower half-shell, split exactly at |v| = w; colour by region and side."""
    ab = alpha_b(w)
    if ab is None:
        al_edges = np.linspace(0, np.pi, na + 1)
    else:
        n1 = max(4, int(round(na * ab / np.pi)))
        al_edges = np.concatenate([np.linspace(0, ab, n1 + 1), np.linspace(ab, np.pi, na - n1 + 1)[1:]])
    be_edges = np.linspace(np.pi, 2 * np.pi, nb + 1)
    out = []
    for i in range(len(al_edges) - 1):
        a0, a1 = al_edges[i], al_edges[i + 1]
        am = 0.5 * (a0 + a1)
        vmag = np.linalg.norm(shell_point(am, 1.5 * np.pi))
        contrib = vmag > w * S
        for j in range(nb):
            b0, b1 = be_edges[j], be_edges[j + 1]
            quad = shell_point(np.array([a0, a1, a1, a0]), np.array([b0, b0, b1, b1]))
            cen = shell_point(am, 0.5 * (b0 + b1))
            nout = (cen - CPOS) / (U * S)
            outer = (CAM.loc - cen) @ nout > 0
            if contrib:
                fill = TERRA_OUT if outer else TERRA
            else:
                fill = NEUT_OUT if outer else NEUT
            out.append((CAM.depth(cen), quad, fill))
    out.sort(key=lambda t: -t[0])
    return out


def anchor_points(w):
    u, ws = U * S, w * S
    pu = CPOS + np.array([0, -u, 0])
    pw = np.array([0, -ws, 0])
    # callout targets on the visible inner wall (back half, y > 0): far from O, and next to O
    far = shell_point(np.radians(128), np.radians(318))
    near = shell_point(np.radians(40), np.radians(205))        # outer face, front, next to O
    assert not occluded(np.array([far, near])).any(), "callout anchor hidden by the shell"
    return {"O": np.zeros(3), "C": CPOS, "pu": pu, "pw": pw, "u_mid": (CPOS + pu) / 2,
            "w_mid": pw / 2, "vE_lab": CPOS * 0.2, "far": far, "near": near}


def panel_render(d, i, w):
    ox, oy = i * PANEL_W, TOP
    clip = draw.ClipPath()
    clip.append(draw.Rectangle(ox + 10, oy - 10, PANEL_W - 20, PANEL_H + 10))
    g = draw.Group(clip_path=clip)

    # flat facets; each stroked in its own colour so no seams show
    for _, quad, fill in shell_facets(w):
        xy = CAM(quad) + [ox, oy]
        g.append(draw.Lines(*[float(c) for p in xy for c in p], close=True, fill=fill,
                            stroke=fill, stroke_width=0.9, stroke_linejoin="round"))

    g_case, g_line = draw.Group(), draw.Group()
    g.append(g_case)
    g.append(g_line)
    # shell outlines: rim at z = 0 and the lower silhouette
    hidden_line(g_line, rim(CPOS, U * S), ox, oy, INK, 2.6)
    hidden_line(g_line, silhouette(CPOS, U * S), ox, oy, INK, 2.6)
    # the sphere |v| = w about O: rim and silhouette, hidden parts dashed
    hidden_line(g_line, rim([0, 0, 0], w * S), ox, oy, BLUE, 2.8, casing=2.4, g_case=g_case)
    hidden_line(g_line, silhouette([0, 0, 0], w * S), ox, oy, BLUE, 2.8, casing=2.4, g_case=g_case)
    # its intersection with the shell, the boundary of the contributing zone
    ab = alpha_b(w)
    if ab is not None:
        be = np.linspace(np.pi, 2 * np.pi, 361)
        hidden_line(g_line, shell_point(np.full_like(be, ab), be), ox, oy, BLUE, 4.4, casing=2.8, g_case=g_case)

    A = {k: CAM(p) + [ox, oy] for k, p in anchor_points(w).items()}
    # radii u and w along -y in the plane z = 0, and the arrow O -> C
    g_line.append(draw.Line(*A["C"], *A["pu"], stroke=C["slate"], stroke_width=2.4))
    polyline(g_line, [A["O"], A["pw"]], BLUE, 2.6, casing=2.4, g_case=g_case)
    arrow2d(g_line, A["O"], A["C"], INK, width=3.0, head=20)
    g_line.append(draw.Circle(*A["O"], 7, fill=INK, stroke="white", stroke_width=2))
    g_line.append(draw.Circle(*A["C"], 6, fill=C["slate"], stroke="white", stroke_width=2))
    d.append(g)
    return A


def panel_labels(d, i, w, A):
    ox, oy = i * PANEL_W, TOP
    text(d, "abc"[i], ox + 8, 30, size=40, weight="bold")
    math(d, [("w", "it"), (f" = {w} km/s", "rm")], ox + 60, 30, size=FS)
    reg = {100: [("w", "it"), (" < |", "rm"), ("u", "it"), (" \u2212 ", "rm"), ("v", "it"), ("E", "sub"), ("|", "rm")],
           400: [("|", "rm"), ("u", "it"), (" \u2212 ", "rm"), ("v", "it"), ("E", "sub"), ("| < ", "rm"), ("w", "it"),
                 (" < ", "rm"), ("u", "it"), (" + ", "rm"), ("v", "it"), ("E", "sub")],
           700: [("w", "it"), (" > ", "rm"), ("u", "it"), (" + ", "rm"), ("v", "it"), ("E", "sub")]}[w]
    math(d, reg, ox + 60, 30 + FS * 1.2, size=FS_SMALL, fill=C["slate"])

    xo, yo = A["O"]
    xc, yc = A["C"]
    text(d, "O", xo + 12, yo - 24, size=FS, weight="bold")
    text(d, "C", xc - 14, yc - 26, size=FS, anchor="end", weight="bold")
    if w == 400:
        xm, ym = A["vE_lab"]
        math(d, [("v", "it"), ("E", "sub")], xm, ym - 26, size=FS, anchor="middle", halo=True)
        xu, yu = A["u_mid"]
        math(d, [("u", "it")], xu - 14, yu, size=FS, anchor="end", fill=C["slate"], halo=True)
        xw, yw = A["w_mid"]
        math(d, [("w", "it")], xw + 14, yw, size=FS, fill=BLUE, halo=True)
    if w == 100:
        xw, yw = A["pw"]
        math(d, [("w", "it")], xw + 10, yw + 20, size=FS, fill=BLUE, halo=True)


def panel_callouts(d, i, w, A):
    """Callouts for the two zones and the circle |v| = w, placed in the panel margins."""
    ox, oy = i * PANEL_W, TOP
    bot = oy + PANEL_H + 12
    if w == 100:
        xz, yz = A["far"]
        leader(d, ox + 40, bot - 22, xz, yz, fill=C["terracotta"])
        math(d, [("|", "rm"), ("v", "it"), ("| > ", "rm"), ("w", "it"), (" on the whole shell", "rm")],
             ox + 14, bot + 8, size=FS_SMALL, fill=C["terracotta"])
    if w == 400:
        xz, yz = A["far"]
        leader(d, ox + 40, bot - 22, xz, yz, fill=C["terracotta"])
        math(d, [("|", "rm"), ("v", "it"), ("| > ", "rm"), ("w", "it")], ox + 14, bot + 8, size=FS, fill=C["terracotta"])
        xn, yn = A["near"]
        lx, ly = ox + PANEL_W - 128, bot + 8
        leader(d, lx + 40, ly - 30, xn, yn, fill=C["slate"])
        math(d, [("|", "rm"), ("v", "it"), ("| < ", "rm"), ("w", "it")], lx, ly, size=FS, fill=C["slate"])
    if w == 700:
        xz, yz = A["far"]
        leader(d, ox + 40, bot - 22, xz, yz, fill=C["slate"])
        math(d, [("|", "rm"), ("v", "it"), ("| < ", "rm"), ("w", "it"), (" on the whole shell", "rm")],
             ox + 14, bot + 8, size=FS_SMALL, fill=C["slate"])


# ---------------------------------------------------------------- panel d

def eta(w):
    lo, hi = abs(U - VE), U + VE
    if w < lo:
        return 1.0 / U
    if w > hi:
        return 0.0
    return (hi - w) / (2 * U * VE)


def panel_function(d):
    ax0, ay0 = D_X + 80, TOP + 20
    aw, ah = D_W - 90, PANEL_H - 100
    xmax, ymax = 800.0, 3.0  # km/s, 1e-3 s/km

    def X(v):
        return ax0 + v / xmax * aw

    def Y(e):
        return ay0 + ah - (e * 1e3) / ymax * ah

    text(d, "d", D_X + 8, 30, size=40, weight="bold")
    math(d, [("\u03b7", "it"), ("u", "sub"), ("(", "rm"), ("w", "it"), (") for ", "rm"), ("u", "it"), (" = 400 km/s", "rm")],
         D_X + 60, 30, size=FS)
    d.append(draw.Rectangle(X(0), ay0, X(VE) - X(0), ah, fill=C["shade"]))
    d.append(draw.Line(X(VE), ay0, X(VE), ay0 + ah, stroke=C["slate"], stroke_width=1.6, stroke_dasharray="6,6"))
    math(d, [("v", "it"), ("E", "sub")], X(VE) + 10, ay0 + ah - 28, size=FS_SMALL, fill=C["slate"])
    d.append(draw.Line(ax0, ay0 + ah, ax0 + aw, ay0 + ah, stroke=INK, stroke_width=2))
    d.append(draw.Line(ax0, ay0, ax0, ay0 + ah, stroke=INK, stroke_width=2))
    for v in (0, 200, 400, 600, 800):
        d.append(draw.Line(X(v), ay0 + ah, X(v), ay0 + ah + 8, stroke=INK, stroke_width=2))
        text(d, str(v), X(v), ay0 + ah + 26, size=FS_SMALL, anchor="middle")
    for e in (0, 1, 2, 3):
        d.append(draw.Line(ax0 - 8, Y(e * 1e-3), ax0, Y(e * 1e-3), stroke=INK, stroke_width=2))
        text(d, str(e), ax0 - 16, Y(e * 1e-3), size=FS_SMALL, anchor="end")
    math(d, [("w", "it"), (" (km/s)", "rm")], ax0 + aw / 2, ay0 + ah + 62, size=FS, anchor="middle")
    d.append(draw.Text("\u03b7 (10\u207b\u00b3 s/km)", FS, ax0 - 56, ay0 + ah / 2, font_family=FONT_CSS, fill=INK,
                       text_anchor="middle", dominant_baseline="middle",
                       transform=f"rotate(-90 {ax0 - 56} {ay0 + ah / 2})"))
    ws = [0, abs(U - VE), U + VE, xmax]
    pts = []
    for w in ws:
        pts += [X(w), Y(eta(w))]
    d.append(draw.Lines(*pts, fill="none", stroke=C["terracotta"], stroke_width=3.5, stroke_linejoin="round"))
    for w in (abs(U - VE), U + VE):
        d.append(draw.Circle(X(w), Y(eta(w)), 6, fill="white", stroke=C["terracotta"], stroke_width=2.5))
    k1, k2 = abs(U - VE), U + VE
    d.append(draw.Line(X(k1) - 2, Y(eta(k1)) + 8, X(k1) - 8, Y(eta(k1)) + 34, stroke=C["slate"], stroke_width=1.6))
    math(d, [("|", "rm"), ("u", "it"), (" \u2212 ", "rm"), ("v", "it"), ("E", "sub"), ("|", "rm")],
         X(k1) + 28, Y(eta(k1)) + 52, size=FS_SMALL, fill=C["slate"], anchor="end")
    d.append(draw.Line(X(k2) + 6, Y(0) - 8, X(k2) + 34, Y(0) - 62, stroke=C["slate"], stroke_width=1.6))
    math(d, [("u", "it"), (" + ", "rm"), ("v", "it"), ("E", "sub")],
         X(k2) + 20, Y(0) - 80, size=FS_SMALL, fill=C["slate"])
    for letter, w in zip("abc", W_VALUES):
        d.append(draw.Circle(X(w), Y(eta(w)), 9, fill=BLUE, stroke="white", stroke_width=2))
        text(d, letter, X(w), Y(eta(w)) - 30, size=FS, anchor="middle", weight="bold", fill=BLUE)


def main():
    d = draw.Drawing(W, H, origin=(0, 0))
    d.append(draw.Rectangle(0, 0, W, H, fill="white"))
    anchors = [panel_render(d, i, w) for i, w in enumerate(W_VALUES)]
    for i, (w, A) in enumerate(zip(W_VALUES, anchors)):
        panel_labels(d, i, w, A)
        panel_callouts(d, i, w, A)
    panel_function(d)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    svg = d.as_svg()
    (OUT / "shell.svg").write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(FIGDIR / "shell.pdf"))
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(FIGDIR / "shell.png"),
                     output_width=int(W), output_height=int(H))
    print("wrote", FIGDIR / "shell.pdf")


if __name__ == "__main__":
    main()
