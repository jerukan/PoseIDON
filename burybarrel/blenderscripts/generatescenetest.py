import os
from pathlib import Path
import json

import bpy
import mathutils
import numpy as np

bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 875

barrel_obj_path = Path("/Users/jerry/Projects/ms-stuff/barrel-playground/barrels/results/dive3-depthcharge-03-04-reconstr/openmvs-out/scene_dense_mesh_refine_texture.obj")
camposes_path = Path("/Users/jerry/Projects/ms-stuff/barrel-playground/barrels/results/dive3-depthcharge-03-04-reconstr/cam_poses.json")

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

bpy.ops.wm.obj_import(filepath=str(barrel_obj_path), up_axis="Z", forward_axis="Y")
barrelobj = bpy.context.selected_objects[0]
print(f"{barrelobj.name}: {barrelobj}")

with open(camposes_path, "rt") as f:
    camposes = json.load(f)

for i, campose in enumerate(camposes):
    bpy.ops.object.camera_add()
    cam = bpy.context.selected_objects[0]
    cam.data.lens_unit = "FOV"
    cam.data.angle = 2 * np.arctan2(1920, 2 * campose["K"][0][0])
    cam.rotation_mode = "QUATERNION"
    q = campose["R"]
    # ok blender follow opengl camera coordinates (cam faces -Z instead of +Z)
    blendq = mathutils.Quaternion([1, 0, 0, 0])
    blendq.rotate(mathutils.Euler((np.pi, 0, 0), "XYZ"))
    blendq.rotate(mathutils.Quaternion(q))
    cam.rotation_quaternion = blendq
    cam.location = mathutils.Vector(campose["t"])
print(bpy.data.objects)
