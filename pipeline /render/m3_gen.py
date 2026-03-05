import math
import mitsuba as mi

"""
MITSUBA 3: Camera Orbit with Ring of Lights
This script demonstrates programmatically controls the camera and light positions in Mitsuba 3.
To create a series of renders from different viewpoints.
The camera orbits around a target point, 
while a ring of lights is placed in front of the camera to illuminate the scene from various angles.
ToDo:
- Export camera positions and light positions to a file for later use in analysis or other renderers.
- Render muiltiple view at the same camera position but with different light configurations to generate ground truth
- Export Hightlight positions (e.g. by rendering a mask).
"""

# Renderer settings

mi.set_variant("scalar_rgb")
from mitsuba import ScalarTransform4f as T

scene = mi.load_file("../data/scenes/moded_simple.xml")
params = mi.traverse(scene)

# Orbit settings
n_views = 10
target = [0.0, 0.0, 1.25]          # same target as your XML
up = [0.0, 0.0, 1.0]

orbit_radius = 12.0                # matches your original camera distance in Y
orbit_height = 5.0                 # matches your original Z height

# Ring settings (in camera space)
n_lights = 8
ring_radius = 0.35                 # metres in front of camera; adjust as needed
ring_forward = 0.20                # how far in front of camera the ring sits
ring_tilt_deg = 0.0                # optionally tilt ring relative to camera

def rotate_about_axis(v, axis, angle_rad):
    # Rodrigues' rotation formula to rotate vector v around axis by angle_rad in 3d space
    ax = axis
    vx, vy, vz = v
    axx, axy, axz = ax
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    dot = vx*axx + vy*axy + vz*axz
    cx = axy*vz - axz*vy
    cy = axz*vx - axx*vz
    cz = axx*vy - axy*vx
    rx = vx*c + cx*s + axx*dot*(1-c)
    ry = vy*c + cy*s + axy*dot*(1-c)
    rz = vz*c + cz*s + axz*dot*(1-c)
    return [rx, ry, rz]

for i in range(n_views):
    theta = 2.0 * math.pi * i / n_views

    # Camera orbit (circle around the target)
    cam_origin = [
        target[0] + orbit_radius * math.cos(theta),
        target[1] + orbit_radius * math.sin(theta),
        orbit_height
    ]

    cam_to_world = T.look_at(origin=cam_origin, target=target, up=up)
    params["sensor.to_world"] = cam_to_world

    # Compute camera basis vectors in world space:
    # forward points from camera to target
    forward = [
        target[0] - cam_origin[0],
        target[1] - cam_origin[1],
        target[2] - cam_origin[2],
    ]
    f_norm = math.sqrt(sum(x*x for x in forward))
    forward = [x / f_norm for x in forward]

    # right = forward x up
    right = [
        forward[1]*up[2] - forward[2]*up[1],
        forward[2]*up[0] - forward[0]*up[2],
        forward[0]*up[1] - forward[1]*up[0],
    ]
    r_norm = math.sqrt(sum(x*x for x in right))
    right = [x / r_norm for x in right]

    # true_up = right x forward
    true_up = [
        right[1]*forward[2] - right[2]*forward[1],
        right[2]*forward[0] - right[0]*forward[2],
        right[0]*forward[1] - right[1]*forward[0],
    ]

    # Optionally tilt the ring around the right axis
    if abs(ring_tilt_deg) > 1e-6:
        tilt = math.radians(ring_tilt_deg)
        true_up = rotate_about_axis(true_up, right, tilt)
        forward = rotate_about_axis(forward, right, tilt)

    # Ring centre in world space (a bit forward from the camera)
    ring_centre = [
        cam_origin[0] + ring_forward * forward[0],
        cam_origin[1] + ring_forward * forward[1],
        cam_origin[2] + ring_forward * forward[2],
    ]

    # Place N point lights around the ring in the camera plane (right/true_up)
    for k in range(n_lights):
        ang = 2.0 * math.pi * k / n_lights
        offset = [
            ring_radius * math.cos(ang) * right[0] + ring_radius * math.sin(ang) * true_up[0],
            ring_radius * math.cos(ang) * right[1] + ring_radius * math.sin(ang) * true_up[1],
            ring_radius * math.cos(ang) * right[2] + ring_radius * math.sin(ang) * true_up[2],
        ]
        pos = [ring_centre[0] + offset[0], ring_centre[1] + offset[1], ring_centre[2] + offset[2]]
        params[f"ring{k}.position"] = pos

    params.update()

    img = mi.render(scene)
    mi.Bitmap(img).write(f"../data/views/m3_gen_view_{i:02d}.exr")  # EXR is best for scientific work

print("Rendered m3_gen_view_00.exr ... m3_gen_view_09.exr")