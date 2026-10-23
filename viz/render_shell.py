"""Path-traced geometry of a boosted isotropic shell (Figure 1, panels a-c).

Lab-frame velocity space, 1 Blender unit = 100 km/s. A shell of Galactic speed u
is the sphere |v + v_E| = u, centred at C = -v_E; the detector is at rest at O.
Points of the shell farther than w = v_min from O contribute to eta(w). The
scene is cut at the plane z = 0 so the interior is visible: the shell bowl,
coloured where |v| > w, and the translucent bowl |v| = w about O.

Run with a bpy venv (Blender 4.2+ / 5.0 as a Python module):
  gtimeout 900 .venv-bpy/bin/python viz/render_shell.py --w 400 [--pct 25 --samples 64]
Writes viz/out/shell_w<w>_render.png (RGBA, shadow catcher) and
viz/out/shell_w<w>_points.json (projected anchors for compose_shell.py).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # noqa: F401  (must come before bmesh)
import bmesh
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "config"))
from figstyle import COLORS  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

V_E = 232.0
U = 400.0
S = 0.01  # Blender units per km/s

C_POS = Vector((-V_E * S, 0.0, 0.0))
FAR = Vector((-(V_E + U) * S, 0.0, 0.0))
NEAR = Vector(((U - V_E) * S, 0.0, 0.0))


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def srgb_to_linear(c):
    return tuple((v / 12.92) if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


def col(name):
    return (*srgb_to_linear(hex_to_rgb(COLORS[name])), 1.0)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def principled(name, rgb, rough=0.5, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = rgb
    b.inputs["Roughness"].default_value = rough
    b.inputs["Alpha"].default_value = alpha
    return m


def shell_material(w_units: float):
    """Terracotta where the world-space distance from O exceeds w, pale otherwise."""
    m = bpy.data.materials.new("shell")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.55
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    length = nt.nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    gt = nt.nodes.new("ShaderNodeMath")
    gt.operation = "GREATER_THAN"
    gt.inputs[1].default_value = w_units
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.inputs["Color1"].default_value = col("shade")
    mix.inputs["Color2"].default_value = col("terracotta")
    nt.links.new(geo.outputs["Position"], length.inputs[0])
    nt.links.new(length.outputs["Value"], gt.inputs[0])
    nt.links.new(gt.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(mix.outputs["Color"], b.inputs["Base Color"])
    return m


def bowl(name, radius, center, material, thickness=0.035, segments=192, rings=96):
    """Lower hemisphere (z <= 0) of a sphere, solidified so the rim reads."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=center,
                                         segments=segments, ring_count=rings)
    ob = bpy.context.object
    ob.name = name
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    # cut in world space: the plane z = 0 is z = -center.z in object space
    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                           plane_co=(0, 0, -center[2]), plane_no=(0, 0, 1), clear_outer=True)
    bm.to_mesh(ob.data)
    bm.free()
    for p in ob.data.polygons:
        p.use_smooth = True
    if thickness > 0:
        sol = ob.modifiers.new("sol", "SOLIDIFY")
        sol.thickness = thickness
        sol.offset = 0.0
    ob.data.materials.append(material)
    return ob


def ball(loc, r, material, name):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=48, ring_count=24)
    ob = bpy.context.object
    ob.name = name
    for p in ob.data.polygons:
        p.use_smooth = True
    ob.data.materials.append(material)
    return ob


def rod(p1, p2, r, material, name):
    p1, p2 = Vector(p1), Vector(p2)
    d = p2 - p1
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length, location=(p1 + p2) / 2, vertices=32)
    ob = bpy.context.object
    ob.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    ob.name = name
    for p in ob.data.polygons:
        p.use_smooth = True
    ob.data.materials.append(material)
    return ob


def arrow(p1, p2, r, material, name, head=0.16):
    p1, p2 = Vector(p1), Vector(p2)
    d = (p2 - p1).normalized()
    rod(p1, p2 - d * head, r, material, name + "_shaft")
    bpy.ops.mesh.primitive_cone_add(radius1=r * 2.6, radius2=0.0, depth=head,
                                    location=p2 - d * head / 2, vertices=32)
    ob = bpy.context.object
    ob.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    ob.name = name + "_head"
    ob.data.materials.append(material)


def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def build(w_kms: float):
    w = w_kms * S
    u = U * S
    ink = principled("ink", col("ink"), rough=0.4)
    slate = principled("slate", col("slate"), rough=0.4)
    blue = principled("wbowl", col("dustyblue"), rough=0.2, alpha=0.18)

    bowl("shell", u, C_POS, shell_material(w))
    bowl("wbowl", w, (0, 0, 0), blue, thickness=0.0)
    ball((0, 0, 0), 0.09, ink, "O")
    ball(C_POS, 0.08, slate, "C")
    arrow((0, 0, 0), C_POS, 0.022, ink, "vE")
    # radius u from C to the rim point on the camera side, radius w from O likewise
    pu = C_POS + Vector((0, -u, 0))
    pw = Vector((0, -w, 0))
    rod(C_POS, pu, 0.018, slate, "u_rod")
    rod((0, 0, 0), pw, 0.018, principled("wrod", col("dustyblue"), rough=0.3), "w_rod")

    # floor: shadow catcher just under the deepest bowl
    depth = -max(u, w) - 0.08
    bpy.ops.mesh.primitive_plane_add(size=60, location=(-2.0, 0, depth))
    floor = bpy.context.object
    floor.is_shadow_catcher = True
    floor.data.materials.append(principled("floor", (1, 1, 1, 1), rough=1.0))

    # camera: three-quarter view from the +y side, slightly above the rim
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 55
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = Vector((-1.4, -15.5, 8.6))
    look_at(cam, (-2.3, 0.3, -1.3))
    bpy.context.scene.camera = cam

    # lights: warm key front-left, cool rim behind, soft fill from the camera
    def area(name, loc, energy, size, color=(1, 1, 1), target=(-2.3, 0, -1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = energy
        ld.size = size
        ld.color = color
        lo = bpy.data.objects.new(name, ld)
        bpy.context.collection.objects.link(lo)
        lo.location = loc
        look_at(lo, target)
        return lo

    area("key", (-9, -8, 12), 2600, 6, (1.0, 0.96, 0.90))
    area("rim", (4, 9, 7), 1200, 5, (0.88, 0.93, 1.0))
    area("fill", (2, -14, 5), 700, 10)
    world = bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.9, 0.9, 0.9, 1)
    bg.inputs["Strength"].default_value = 0.35

    anchors = {
        "O": (0, 0, 0), "C": tuple(C_POS), "far": tuple(FAR), "near": tuple(NEAR),
        "pu": tuple(pu), "pw": tuple(pw), "u_mid": tuple((C_POS + pu) / 2), "w_mid": tuple(pw / 2),
        "vE_mid": tuple(C_POS / 2),
        # a point on the inner surface of the shell, far side (coloured when w < 632)
        "zone_far": tuple(C_POS + Vector((-u * math.sin(math.radians(55)), 0, -u * math.cos(math.radians(55))))),
        # a point on the inner surface, near side (coloured only when w < 168)
        "zone_near": tuple(C_POS + Vector((u * math.sin(math.radians(50)), 0.6, -u * math.cos(math.radians(50))))),
        "rim_far": tuple(C_POS + Vector((-u, 0, 0))),
        "wbowl_in": tuple(Vector((0.35 * w, -0.2 * w, -0.9 * w))),
    }
    return cam, anchors


def setup_render(pct: int, samples: int, preview: bool):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "METAL"
        sc.cycles.device = "GPU"
    except Exception:
        sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02 if preview else 0.004
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    try:
        sc.cycles.denoising_use_gpu = True
    except Exception:
        pass
    sc.cycles.transparent_max_bounces = 16
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    sc.render.film_transparent = True
    sc.render.resolution_x = 1800
    sc.render.resolution_y = 1350
    sc.render.resolution_percentage = pct
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "8"


def project(cam, anchors):
    sc = bpy.context.scene
    W = sc.render.resolution_x * sc.render.resolution_percentage / 100
    H = sc.render.resolution_y * sc.render.resolution_percentage / 100
    out = {}
    for k, p in anchors.items():
        co = world_to_camera_view(sc, cam, Vector(p))
        out[k] = [co.x * W, (1 - co.y) * H]
    return out, (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=400.0, help="v_min in km/s")
    ap.add_argument("--pct", type=int, default=100)
    ap.add_argument("--samples", type=int, default=512)
    args = ap.parse_args()
    preview = args.pct < 100
    reset()
    cam, anchors = build(args.w)
    setup_render(args.pct, args.samples, preview)
    tag = f"shell_w{int(args.w)}" + ("_preview" if preview else "")
    bpy.context.scene.render.filepath = str(OUT / f"{tag}_render.png")
    bpy.ops.render.render(write_still=True)
    pts, size = project(cam, anchors)
    (OUT / f"{tag}_points.json").write_text(json.dumps(
        {"w_kms": args.w, "u_kms": U, "vE_kms": V_E, "size": size, "points": pts}, indent=1))
    print("wrote", OUT / f"{tag}_render.png")


if __name__ == "__main__":
    main()
