import math
import json
from pathlib import Path

import mitsuba as mi

mi.set_variant("scalar_rgb")
from mitsuba import ScalarTransform4f as T


materials = ["diffuse", "glossy"]


def setpaths(model_name):
    out_root = Path("/Users/27171653/Desktop/PhD/Specular-Highlights/data/train/") / model_name
    meta_dir = out_root / "metadata"
    return out_root, meta_dir


def ensure_dirs(model_name):
    out_root, meta_dir = setpaths(model_name)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    for mat in materials:
        (out_root / mat).mkdir(parents=True, exist_ok=True)


def normalise(v):
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-12:
        raise ValueError(f"Cannot normalise near-zero vector: {v}")
    return [x / n for x in v]


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def rotate_about_axis(v, axis, angle_rad):
    axis = normalise(axis)
    vx, vy, vz = v
    ax, ay, az = axis

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    dot = vx * ax + vy * ay + vz * az

    cx = ay * vz - az * vy
    cy = az * vx - ax * vz
    cz = ax * vy - ay * vx

    rx = vx * c + cx * s + ax * dot * (1 - c)
    ry = vy * c + cy * s + ay * dot * (1 - c)
    rz = vz * c + cz * s + az * dot * (1 - c)
    return [rx, ry, rz]


def get_camera_pose(i, n_views, target, orbit_radius, orbit_height):
    theta = 2.0 * math.pi * i / n_views
    return [
        target[0] + orbit_radius * math.cos(theta),
        target[1] + orbit_radius * math.sin(theta),
        orbit_height,
    ]


def get_camera_basis(cam_origin, target, up, ring_tilt_deg=0.0):
    forward = [
        target[0] - cam_origin[0],
        target[1] - cam_origin[1],
        target[2] - cam_origin[2],
    ]
    forward = normalise(forward)

    right = cross(forward, up)
    right_norm = math.sqrt(sum(x * x for x in right))
    if right_norm < 1e-12:
        raise ValueError("Camera forward vector is parallel to up vector.")
    right = [x / right_norm for x in right]

    true_up = normalise(cross(right, forward))

    if abs(ring_tilt_deg) > 1e-6:
        tilt = math.radians(ring_tilt_deg)
        true_up = normalise(rotate_about_axis(true_up, right, tilt))
        forward = normalise(rotate_about_axis(forward, right, tilt))

    return forward, right, true_up


def set_sensor_pose(params, cam_origin, target, up):
    params["sensor.to_world"] = T.look_at(
        origin=cam_origin,
        target=target,
        up=up,
    )


def set_ring_lights(params, cam_origin, forward, right, true_up, n_lights, ring_radius, ring_forward):
    ring_centre = [
        cam_origin[0] + ring_forward * forward[0],
        cam_origin[1] + ring_forward * forward[1],
        cam_origin[2] + ring_forward * forward[2],
    ]

    light_positions = []

    for k in range(n_lights):
        ang = 2.0 * math.pi * k / n_lights
        offset = [
            ring_radius * math.cos(ang) * right[0] + ring_radius * math.sin(ang) * true_up[0],
            ring_radius * math.cos(ang) * right[1] + ring_radius * math.sin(ang) * true_up[1],
            ring_radius * math.cos(ang) * right[2] + ring_radius * math.sin(ang) * true_up[2],
        ]
        pos = [
            ring_centre[0] + offset[0],
            ring_centre[1] + offset[1],
            ring_centre[2] + offset[2],
        ]
        params[f"ring{k}.position"] = pos
        light_positions.append(pos)

    return ring_centre, light_positions


def save_metadata(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_scene_and_params(scene_path):
    scene = mi.load_file(scene_path)
    params = mi.traverse(scene)
    return scene, params


def prepare_scene(params, cam_origin, scene_info):
    target = scene_info["target"]
    up = scene_info["up"]
    ring_tilt_deg = scene_info["ring_tilt_deg"]
    n_lights = scene_info["n_lights"]
    ring_radius = scene_info["ring_radius"]
    ring_forward = scene_info["ring_forward"]

    forward, right, true_up = get_camera_basis(
        cam_origin=cam_origin,
        target=target,
        up=up,
        ring_tilt_deg=ring_tilt_deg,
    )

    set_sensor_pose(params, cam_origin, target, up)

    ring_centre, light_positions = set_ring_lights(
        params=params,
        cam_origin=cam_origin,
        forward=forward,
        right=right,
        true_up=true_up,
        n_lights=n_lights,
        ring_radius=ring_radius,
        ring_forward=ring_forward,
    )

    params.update()

    return {
        "camera_origin": cam_origin,
        "target": target,
        "up": up,
        "forward": forward,
        "right": right,
        "true_up": true_up,
        "ring_centre": ring_centre,
        "ring_positions": light_positions,
        "orbit_radius": scene_info["orbit_radius"],
        "orbit_height": scene_info["orbit_height"],
        "ring_radius": ring_radius,
        "ring_forward": ring_forward,
        "ring_tilt_deg": ring_tilt_deg,
        "n_lights": n_lights,
        "spp": scene_info["spp"],
    }


def _write_exr_and_png(img, exr_path, png_path):
    bmp = mi.Bitmap(img)
    bmp.write(str(exr_path))

    ldr = bmp.convert(
        pixel_format=mi.Bitmap.PixelFormat.RGB,
        component_format=mi.Struct.Type.UInt8,
        srgb_gamma=True,
    )
    ldr.write(str(png_path))


def render_dataset_from_sceneinfo(
    model,
    diffuse_scene_path,
    glossy_scene_path,
    scene_info,
):
    required_keys = [
        "n_views",
        "target",
        "up",
        "orbit_radius",
        "orbit_height",
        "n_lights",
        "ring_radius",
        "ring_forward",
        "ring_tilt_deg",
        "spp",
    ]
    for k in required_keys:
        if k not in scene_info:
            raise KeyError(f"scene_info is missing required key: {k}")

    n_views = int(scene_info["n_views"])
    target = scene_info["target"]
    orbit_radius = float(scene_info["orbit_radius"])
    orbit_height = float(scene_info["orbit_height"])
    spp = int(scene_info["spp"])

    ensure_dirs(model_name=model)
    out_root, meta_dir = setpaths(model)

    diffuse_scene, diffuse_params = load_scene_and_params(diffuse_scene_path)
    glossy_scene, glossy_params = load_scene_and_params(glossy_scene_path)

    for i in range(n_views):
        cam_origin = get_camera_pose(
            i=i,
            n_views=n_views,
            target=target,
            orbit_radius=orbit_radius,
            orbit_height=orbit_height,
        )

        diffuse_meta = prepare_scene(diffuse_params, cam_origin, scene_info)
        glossy_meta = prepare_scene(glossy_params, cam_origin, scene_info)

        diffuse_img = mi.render(diffuse_scene, spp=spp)
        glossy_img = mi.render(glossy_scene, spp=spp)

        diffuse_exr = out_root / "diffuse" / f"position_{i:02d}.exr"
        diffuse_png = out_root / "diffuse" / f"position_{i:02d}.png"

        glossy_exr = out_root / "glossy" / f"position_{i:02d}.exr"
        glossy_png = out_root / "glossy" / f"position_{i:02d}.png"

        _write_exr_and_png(diffuse_img, diffuse_exr, diffuse_png)
        _write_exr_and_png(glossy_img, glossy_exr, glossy_png)

        base_meta = {
            "model": model,
            "position_index": i,
            "diffuse_scene_path": str(diffuse_scene_path),
            "glossy_scene_path": str(glossy_scene_path),
            "camera_origin": cam_origin,
            "target": scene_info["target"],
            "up": scene_info["up"],
            "orbit_radius": scene_info["orbit_radius"],
            "orbit_height": scene_info["orbit_height"],
            "ring_radius": scene_info["ring_radius"],
            "ring_forward": scene_info["ring_forward"],
            "ring_tilt_deg": scene_info["ring_tilt_deg"],
            "n_lights": scene_info["n_lights"],
            "spp": scene_info["spp"],
        }

        diffuse_out_meta = dict(base_meta)
        diffuse_out_meta["material"] = "diffuse"
        diffuse_out_meta["exr_path"] = str(diffuse_exr)
        diffuse_out_meta["png_path"] = str(diffuse_png)
        diffuse_out_meta["scene_meta"] = diffuse_meta

        glossy_out_meta = dict(base_meta)
        glossy_out_meta["material"] = "glossy"
        glossy_out_meta["exr_path"] = str(glossy_exr)
        glossy_out_meta["png_path"] = str(glossy_png)
        glossy_out_meta["scene_meta"] = glossy_meta

        save_metadata(meta_dir / f"position_{i:02d}_diffuse.json", diffuse_out_meta)
        save_metadata(meta_dir / f"position_{i:02d}_glossy.json", glossy_out_meta)

        print(f"Rendered position_{i:02d} | diffuse + glossy")

    print(f"Done. Rendered {n_views} paired positions.")