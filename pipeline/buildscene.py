from pathlib import Path
import warnings
import trimesh


def stl_to_ply(
    stl_path,
    ply_path=None,
    smooth_iterations=20,
    lambda_factor=0.5,
    subdivide=False,
    subdivisions=1,
):
    """
    Convert STL to PLY and optionally smooth the surface.

    Parameters
    ----------
    stl_path : str
        Path to input STL file.
    ply_path : str or None
        Path to output PLY file. If None, uses same name as STL.
    smooth_iterations : int
        Number of Laplacian smoothing iterations.
    lambda_factor : float
        Smoothing strength per iteration.
    subdivide : bool
        If True, subdivide before smoothing for a smoother surface.
    subdivisions : int
        Number of subdivision steps if subdivide=True.

    Returns
    -------
    ply_path : str
        Path to saved PLY.
    mesh : trimesh.Trimesh
        Processed mesh.
    """

    stl_path = Path(stl_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    if ply_path is None:
        ply_path = stl_path.with_suffix(".ply")
    else:
        ply_path = Path(ply_path)

    mesh = trimesh.load_mesh(str(stl_path), force="mesh")
    if mesh is None or mesh.is_empty:
        raise ValueError(f"Could not load a valid mesh from: {stl_path}")

    mesh = mesh.copy()
    original_mesh = mesh.copy()

    # Optional subdivision to give smoothing more vertices to work with
    if subdivide:
        for _ in range(subdivisions):
            mesh = mesh.subdivide()

    # Laplacian smoothing
    if smooth_iterations > 0:
        trimesh.smoothing.filter_laplacian(
            mesh,
            lamb=lambda_factor,
            iterations=smooth_iterations,
            implicit_time_integration=False,
            volume_constraint=True,
        )

        # Some meshes collapse under Laplacian smoothing; keep the raw conversion in that case.
        invalid_mesh = (
            mesh.is_empty
            or len(mesh.vertices) == 0
            or len(mesh.faces) == 0
            or not trimesh.util.is_shape(mesh.vertices, (-1, 3))
            or not trimesh.util.is_shape(mesh.faces, (-1, 3))
        )
        if not invalid_mesh:
            invalid_mesh = not (
                mesh.vertices.size
                and mesh.faces.size
                and mesh.bounds is not None
                and trimesh.util.allclose(mesh.vertices, mesh.vertices)
            )

        if invalid_mesh:
            warnings.warn(
                f"Smoothing produced an invalid mesh for {stl_path.name}; "
                "falling back to the unsmoothed mesh.",
                RuntimeWarning,
            )
            mesh = original_mesh

    # Recompute normals for smoother shading
    mesh.vertex_normals

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(ply_path))

    return str(ply_path), mesh

def buildscene(
    objectpath,
    material_mode="diffuse",   # "diffuse" or "glossy"
    template_xml_path="./scenes/object_template.xml",
    output_dir="./scenes/generated",
    n_views=10,
    n_lights=8,
    ring_tilt_deg=0.0,
    up=(0.0, 0.0, 1.0),
    spp=256,
    res=256,
    ring_intensity=40.0,
    auto_scale_ring_intensity=True,
    ring_intensity_reference_distance=27.0,
    orbit_radius_scale=4.0,
    orbit_height_scale=1.5,
    ring_radius_scale=0.15,
    ring_forward_scale=0.08,
    target_z_offset_scale=0.1,
    diffuse_reflectance=(0.35, 0.35, 0.35),
    glossy_diffuse_reflectance=(0.28, 0.28, 0.28),
    glossy_alpha=0.05,
    glossy_int_ior=1.49,
):
    """
    Build a Mitsuba XML scene for one mesh and one material mode.

    The defaults are chosen so that:
    - diffuse is not too bright
    - glossy has a similar base tone
    - glossy - diffuse gives a more meaningful highlight map
    """

    objectpath = Path(objectpath)
    template_xml_path = Path(template_xml_path)
    output_dir = Path(output_dir)

    if not objectpath.exists():
        raise FileNotFoundError(f"Object file not found: {objectpath}")

    if not template_xml_path.exists():
        raise FileNotFoundError(f"Template XML not found: {template_xml_path}")

    if objectpath.suffix.lower() != ".ply":
        raise ValueError(
            f"Current template uses <shape type='ply'>, but got {objectpath.suffix}. "
            "Convert the mesh to .ply or change the XML shape type."
        )

    mesh = trimesh.load(str(objectpath), force="mesh")
    if mesh is None or mesh.is_empty:
        raise ValueError(f"Loaded mesh is empty or invalid: {objectpath}")

    bounds = mesh.bounds
    mins = bounds[0]
    maxs = bounds[1]

    centre = ((mins + maxs) / 2.0).tolist()
    extent = (maxs - mins).tolist()
    size = float(max(extent))

    if size <= 0:
        raise ValueError(f"Invalid mesh size from bounds: {extent}")

    target = [
        float(centre[0]),
        float(centre[1]),
        float(centre[2] + target_z_offset_scale * extent[2]),
    ]

    orbit_radius = float(orbit_radius_scale * size)
    orbit_height = float(centre[2] + orbit_height_scale * size)
    ring_radius = float(ring_radius_scale * size)
    ring_forward = float(ring_forward_scale * size)

    cam_target_distance = float(
        ((orbit_radius ** 2) + (orbit_height - target[2]) ** 2) ** 0.5
    )
    light_target_distance = float(
        (((cam_target_distance - ring_forward) ** 2) + ring_radius ** 2) ** 0.5
    )
    if auto_scale_ring_intensity:
        effective_ring_intensity = float(
            ring_intensity * (light_target_distance / ring_intensity_reference_distance) ** 2
        )
    else:
        effective_ring_intensity = float(ring_intensity)

    if material_mode == "diffuse":
        bsdf_block = f"""        <bsdf type="diffuse" id="object_bsdf">
            <rgb name="reflectance" value="{diffuse_reflectance[0]}, {diffuse_reflectance[1]}, {diffuse_reflectance[2]}"/>
        </bsdf>"""
    elif material_mode == "glossy":
        bsdf_block = f"""        <bsdf type="roughplastic" id="object_bsdf">
            <rgb name="diffuse_reflectance" value="{glossy_diffuse_reflectance[0]}, {glossy_diffuse_reflectance[1]}, {glossy_diffuse_reflectance[2]}"/>
            <float name="alpha" value="{glossy_alpha}"/>
            <float name="int_ior" value="{glossy_int_ior}"/>
        </bsdf>"""
    else:
        raise ValueError("material_mode must be 'diffuse' or 'glossy'")

    xml = template_xml_path.read_text(encoding="utf-8")

    replacements = {
        "{{SPP}}": str(int(spp)),
        "{{RES}}": str(int(res)),
        "{{RING_INTENSITY}}": str(effective_ring_intensity),
        "{{MESH_FILE}}": objectpath.as_posix(),
        "{{TARGET_X}}": f"{target[0]:.6f}",
        "{{TARGET_Y}}": f"{target[1]:.6f}",
        "{{TARGET_Z}}": f"{target[2]:.6f}",
        "{{ORIGIN_X}}": f"{target[0]:.6f}",
        "{{ORIGIN_Y}}": f"{target[1] - orbit_radius:.6f}",
        "{{ORIGIN_Z}}": f"{orbit_height:.6f}",
        "{{UP_X}}": f"{float(up[0]):.6f}",
        "{{UP_Y}}": f"{float(up[1]):.6f}",
        "{{UP_Z}}": f"{float(up[2]):.6f}",
        "{{BSDF_BLOCK}}": bsdf_block,
    }

    missing = [k for k in replacements if k not in xml]
    if missing:
        raise ValueError(f"Template XML is missing placeholders: {missing}")

    for key, value in replacements.items():
        xml = xml.replace(key, value)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_scene_path = output_dir / f"{objectpath.stem}_{material_mode}.xml"
    saved_scene_path.write_text(xml, encoding="utf-8")

    scene_info = {
        "objectpath": str(objectpath),
        "saved_scene_path": str(saved_scene_path),
        "material_mode": material_mode,
        "mesh_bounds_min": mins.tolist(),
        "mesh_bounds_max": maxs.tolist(),
        "centre": centre,
        "extent": extent,
        "size": size,
        "n_views": int(n_views),
        "n_lights": int(n_lights),
        "ring_tilt_deg": float(ring_tilt_deg),
        "up": [float(x) for x in up],
        "target": target,
        "orbit_radius": orbit_radius,
        "orbit_height": orbit_height,
        "ring_radius": ring_radius,
        "ring_forward": ring_forward,
        "cam_target_distance": cam_target_distance,
        "light_target_distance": light_target_distance,
        "spp": int(spp),
        "res": int(res),
        "ring_intensity": effective_ring_intensity,
        "ring_intensity_base": float(ring_intensity),
        "auto_scale_ring_intensity": bool(auto_scale_ring_intensity),
        "ring_intensity_reference_distance": float(ring_intensity_reference_distance),
        "diffuse_reflectance": [float(x) for x in diffuse_reflectance],
        "glossy_diffuse_reflectance": [float(x) for x in glossy_diffuse_reflectance],
        "glossy_alpha": float(glossy_alpha),
        "glossy_int_ior": float(glossy_int_ior),
    }

    return str(saved_scene_path), scene_info
