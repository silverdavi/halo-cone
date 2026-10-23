"""Cover art: under isotropy the 8B neutrino fog in xenon stands only between m_lo and m_hi.

The floor is the dark-matter mass axis (log scale, K units per decade). The fog is a
volume whose mass extent is exactly the isotropic exact-degeneracy window of the
ideal single-isotope detector at the paper's threshold, [m_lo, m_hi(E_thr)], read from
paper/numbers.tex. Its walls in mass are sharp; height (cross section, free in the
degeneracy) and depth (no meaning) are soft. It is lit from behind by a low Sun, the
source of the 8B neutrinos. The any-halo extent, every mass >= m_lo, is drawn in the
compose step as a dashed outline, never rendered.

  gtimeout 900 ../../eilmer/.venv-bpy/bin/python hero/render_cover.py --view cover [--pct 25 --samples 64]
Writes hero/out/<view>_render.png and hero/out/<view>_points.json (projected anchors).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import bpy  # noqa: F401  (must come before bmesh)
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def paper_number(name: str) -> float:
    tex = (ROOT / "paper" / "numbers.tex").read_text()
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{([-0-9.]+)\}", tex)
    return float(m.group(1))


M_LO = paper_number("nMlo")      # 6.7 GeV, end-point edge (any halo and isotropic)
M_HI = paper_number("nMhi")      # 7.7 GeV, isotropic upper edge at the 0.51 keV threshold
K = 11.0                         # scene units per decade of mass
M_C = math.sqrt(M_LO * M_HI)     # window centre sits at x = 0
DEPTH = 2.6                      # half-depth of the fog slab (y), no physical meaning
HEIGHT = 15.0                    # fog height (z), the free cross-section direction
AXIS_Y = -DEPTH                  # the scale runs along the front edge of the fog


def X(m: float) -> float:
    return K * math.log10(m / M_C)


VIEWS = {
    # resolution, camera location, target, lens (mm, sensor 36 on the long side)
    "cover": dict(res=(2550, 3300), loc=(3.4, -13.0, 1.7), target=(0.9, 0.0, 5.4), lens=24.0),
    "wide": dict(res=(3840, 2160), loc=(3.0, -25.0, 1.9), target=(3.9, 0.0, 5.9), lens=24.0),
}
SUN_DIST = 140.0                       # y of the Sun, far behind the slab
SUN_AIM = Vector((0.0, 0.0, 2.05))      # the Sun sits on the line camera -> this point, so it backlights the fog


def sun_pos(view):
    c = Vector(VIEWS[view]["loc"])
    d = SUN_AIM - c
    return c + d * ((SUN_DIST - c.y) / d.y)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def link(ob):
    bpy.context.collection.objects.link(ob)
    return ob


def nd(nt, kind, **inputs):
    n = nt.nodes.new(kind)
    for k, v in inputs.items():
        n.inputs[k].default_value = v
    return n


def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def box(name, x0, x1, y0, y1, z0, z1, material):
    bpy.ops.mesh.primitive_cube_add(size=1, location=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    ob = bpy.context.object
    ob.name = name
    ob.scale = (x1 - x0, y1 - y0, z1 - z0)
    bpy.ops.object.transform_apply(scale=True)
    ob.data.materials.append(material)
    return ob


def mat_fog():
    m = bpy.data.materials.new("fog")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vol = nd(nt, "ShaderNodeVolumePrincipled", **{"Anisotropy": 0.72})
    vol.inputs["Color"].default_value = (1.0, 0.76, 0.50, 1)
    tc = nt.nodes.new("ShaderNodeNewGeometry")                 # world position
    mapn = nt.nodes.new("ShaderNodeMapping")
    mapn.inputs["Scale"].default_value = (2.4, 0.22, 0.30)     # strands run up and through the slab
    noise = nd(nt, "ShaderNodeTexNoise", Scale=1.7, Detail=10.0, Roughness=0.66, Distortion=0.6)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[0].color = (0.02, 0.02, 0.02, 1)
    ramp.color_ramp.elements[1].position = 0.66
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    # fade with height and toward the (meaningless) front and back faces; mass walls stay sharp
    zfade = nd(nt, "ShaderNodeMapRange", **{"From Min": 0.0, "From Max": HEIGHT, "To Min": 1.0, "To Max": 0.0})
    zfade.interpolation_type = "SMOOTHERSTEP"
    absy = nd(nt, "ShaderNodeMath")
    absy.operation = "ABSOLUTE"
    yfade = nd(nt, "ShaderNodeMapRange", **{"From Min": DEPTH * 0.35, "From Max": DEPTH, "To Min": 1.0, "To Max": 0.0})
    yfade.interpolation_type = "SMOOTHSTEP"
    m1 = nd(nt, "ShaderNodeMath")
    m1.operation = "MULTIPLY"
    m2 = nd(nt, "ShaderNodeMath")
    m2.operation = "MULTIPLY"
    m3 = nd(nt, "ShaderNodeMath")
    m3.operation = "MULTIPLY"
    m3.inputs[1].default_value = 1.3   # peak density per unit length
    L = nt.links.new
    L(tc.outputs["Position"], mapn.inputs["Vector"])
    L(mapn.outputs["Vector"], noise.inputs["Vector"])
    L(noise.outputs["Fac"], ramp.inputs["Fac"])
    L(tc.outputs["Position"], sep.inputs["Vector"])
    L(sep.outputs["Z"], zfade.inputs["Value"])
    L(sep.outputs["Y"], absy.inputs[0])
    L(absy.outputs["Value"], yfade.inputs["Value"])
    L(ramp.outputs["Color"], m1.inputs[0])
    L(zfade.outputs["Result"], m1.inputs[1])
    L(m1.outputs["Value"], m2.inputs[0])
    L(yfade.outputs["Result"], m2.inputs[1])
    L(m2.outputs["Value"], m3.inputs[0])
    L(m3.outputs["Value"], vol.inputs["Density"])
    L(vol.outputs["Volume"], out.inputs["Volume"])
    return m


def mat_floor():
    m = bpy.data.materials.new("floor")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.012, 0.014, 0.02, 1)
    b.inputs["Roughness"].default_value = 0.32
    b.inputs["Specular IOR Level"].default_value = 0.3
    return m


def mat_emit(rgb, strength, name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nd(nt, "ShaderNodeEmission", Strength=strength)
    em.inputs["Color"].default_value = (*rgb, 1)
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def build(view):
    SUN_POS = sun_pos(view)
    # floor, the mass axis
    bpy.ops.mesh.primitive_plane_add(size=600, location=(0, 200, 0))
    bpy.context.object.data.materials.append(mat_floor())

    # scale: a baseline and ticks every 1 GeV (4-10), every 10 GeV (10-100)
    tick_mat = mat_emit((0.80, 0.78, 0.74), 0.9, "tick")
    major = [5, 10, 20, 50, 100]
    masses = list(range(3, 10)) + list(range(10, 101, 10))
    x_min, x_max = X(3.0), X(100.0)
    box("baseline", x_min, x_max, AXIS_Y - 0.012, AXIS_Y + 0.012, 0, 0.004, tick_mat)
    for mm in masses:
        x = X(mm)
        ln = 0.55 if mm in major else 0.28
        box(f"tick{mm}", x - 0.014, x + 0.014, AXIS_Y - ln, AXIS_Y, 0, 0.004, tick_mat)

    # the fog: exact window in mass
    fog = box("fog", X(M_LO), X(M_HI), -DEPTH, DEPTH, 0.0, HEIGHT, mat_fog())
    fog_only = bpy.data.collections.new("fog_only")
    bpy.context.scene.collection.children.link(fog_only)
    fog_only.objects.link(fog)

    # the Sun, far behind and low, and the lamp that matches it
    bpy.ops.mesh.primitive_uv_sphere_add(radius=11.0, location=SUN_POS, segments=64, ring_count=32)
    sunob = bpy.context.object
    sunob.data.materials.append(mat_emit((1.0, 0.36, 0.11), 1.1, "sun"))
    sunob.visible_shadow = False
    ld = bpy.data.lights.new("sunlamp", "SUN")
    ld.energy = 4.2
    ld.angle = math.radians(2.0)
    ld.color = (1.0, 0.80, 0.58)
    lo = link(bpy.data.objects.new("sunlamp", ld))
    lo.rotation_euler = (Vector((0, 0, 0)) - SUN_POS).to_track_quat("-Z", "Y").to_euler()
    lo.light_linking.receiver_collection = fog_only     # the lamp lights the fog, not the floor
    # weak cool fill from the camera side so the scale and the front of the curtain read
    fd = bpy.data.lights.new("fill", "AREA")
    fd.energy = 500
    fd.size = 14
    fd.color = (0.62, 0.72, 1.0)
    fo = link(bpy.data.objects.new("fill", fd))
    fo.location = (4, -24, 10)
    look_at(fo, (0, 0, 3))

    # night sky: deep blue overhead, warmer toward the horizon behind the Sun
    world = bpy.data.worlds.new("sky")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.020, 0.030, 0.060, 1)
    ramp.color_ramp.elements[1].position = 0.30
    ramp.color_ramp.elements[1].color = (0.002, 0.004, 0.012, 1)
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0

    anchors = {"sun": tuple(SUN_POS)}
    for mm in masses:
        anchors[f"tick{mm}"] = (X(mm), AXIS_Y - (0.55 if mm in major else 0.28), 0.0)
    for tag, mm in (("lo", M_LO), ("hi", M_HI)):
        for zt, z in (("0", 0.0), ("h", HEIGHT)):
            for yt, y in (("f", -DEPTH), ("b", DEPTH)):
                anchors[f"win_{tag}_{yt}{zt}"] = (X(mm), y, z)
    for xt, mm in (("lo", M_LO), ("far", 100.0)):
        for zt, z in (("0", 0.0), ("h", HEIGHT)):
            for yt, y in (("f", -DEPTH), ("b", DEPTH)):
                anchors[f"ghost_{xt}_{yt}{zt}"] = (X(mm), y, z)
    for tag, mm in (("lo", M_LO), ("hi", M_HI)):
        anchors[f"axis_{tag}"] = (X(mm), AXIS_Y, 0.0)
    anchors["axis_far"] = (X(100.0), AXIS_Y, 0.0)
    for zz in (2.0, 4.0, 6.0, 8.0):
        anchors[f"winmid_z{int(zz)}"] = ((X(M_LO) + X(M_HI)) / 2, -DEPTH, zz)
    return anchors


def setup_render(view, pct, samples, preview):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "METAL"
    sc.cycles.device = "GPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.03 if preview else 0.005
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    sc.cycles.denoising_use_gpu = True
    sc.cycles.max_bounces = 12
    sc.cycles.volume_bounces = 2
    sc.cycles.volume_step_rate = 1.0 if not preview else 2.0
    sc.cycles.sample_clamp_indirect = 6.0
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Base Contrast"
    sc.render.resolution_x, sc.render.resolution_y = VIEWS[view]["res"]
    sc.render.resolution_percentage = pct
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "16"


def camera(view):
    v = VIEWS[view]
    cd = bpy.data.cameras.new("cam")
    cd.lens = v["lens"]
    cd.sensor_width = 36.0
    cd.sensor_fit = "AUTO"
    cd.clip_end = 2000
    cam = link(bpy.data.objects.new("cam", cd))
    cam.location = Vector(v["loc"])
    look_at(cam, v["target"])
    bpy.context.scene.camera = cam
    return cam


def project(cam, anchors):
    sc = bpy.context.scene
    W, H = sc.render.resolution_x, sc.render.resolution_y
    out = {}
    for k, p in anchors.items():
        co = world_to_camera_view(sc, cam, Vector(p))
        out[k] = [co.x * W, (1 - co.y) * H, co.z]
    return out, (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="cover", choices=list(VIEWS))
    ap.add_argument("--pct", type=int, default=100)
    ap.add_argument("--samples", type=int, default=384)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    preview = args.pct < 100
    reset()
    anchors = build(args.view)
    cam = camera(args.view)
    setup_render(args.view, args.pct, args.samples, preview)
    stem = f"{args.view}{args.tag}" + ("_preview" if preview else "")
    bpy.context.scene.render.filepath = str(OUT / f"{stem}_render.png")
    bpy.ops.render.render(write_still=True)
    pts, size = project(cam, anchors)
    full = VIEWS[args.view]["res"]
    meta = {"view": args.view, "size": list(full), "m_lo": M_LO, "m_hi": M_HI, "K": K, "M_C": M_C,
            "points": {k: [p[0] * full[0] / size[0], p[1] * full[1] / size[1], p[2]] for k, p in pts.items()}}
    (OUT / f"{stem}_points.json").write_text(json.dumps(meta, indent=1))
    print("wrote", OUT / f"{stem}_render.png")


if __name__ == "__main__":
    main()
