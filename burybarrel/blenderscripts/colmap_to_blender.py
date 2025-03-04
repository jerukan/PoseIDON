"""
This script loads camera poses from COLMAP into Blender and overlays the reference RGB
image for the camera on its background.

A mesh for the scene as reference should also be loaded in to make manual fitting easier.
The reference model is also loaded for the actual fitting process.
"""

import os
from pathlib import Path
import json

import bpy
import mathutils
import numpy as np

bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 875

# reference model mesh
# reference_model_path = Path("/Users/jerry/Projects/ms-stuff/barrel-playground/models3d/Munitions_Models/depth_charge_mark_9_mod_1-scaled.ply")
reference_model_path = Path("/Users/jerry/Projects/ms-stuff/barrel-playground/models3d/barrelsingle-scaled.ply")

# reconstruction paths
# reconstruct_dense_path = Path("/Users/jerry/Projects/ms-stuff/barrel-playground/barrels/results/dive3-depthcharge-03-04-reconstr/openmvs-out/scene_dense.ply")
# reconstruct_mesh_path = Path("/Users/jerry/Projects/ms-stuff/barrel-playground/barrels/results/dive3-depthcharge-03-04-reconstr/openmvs-out/scene_dense_mesh_refine_texture.obj")
# camposes_path = Path("/Users/jerry/Projects/ms-stuff/barrel-playground/barrels/results/dive3-depthcharge-03-04-reconstr/cam_poses.json")

reconstruct_mesh_path = Path("/Users/jerry/Projects/ms-stuff/barrel-playground/barrels/results/barrel2-reconstr/openmvs-out/scene_dense_mesh_refine_texture.obj")
camposes_path = Path("/Users/jerry/Projects/ms-stuff/barrel-playground/barrels/results/barrel2-reconstr/cam_poses.json")

# clearing every object in the current scene (don't accidentally run this in the wrong scene)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
for c in bpy.context.scene.collection.children:
    bpy.context.scene.collection.children.unlink(c)

bpy.ops.wm.ply_import(filepath=str(reference_model_path))
# bpy.ops.wm.ply_import(filepath=str(reconstruct_dense_path))
bpy.ops.wm.obj_import(filepath=str(reconstruct_mesh_path), up_axis="Z", forward_axis="Y")

camcollection = bpy.data.collections.new("ReconstructedCameras")
bpy.context.scene.collection.children.link(camcollection)

with open(camposes_path, "rt") as f:
    camposes = json.load(f)

for i, campose in enumerate(camposes):
    bpy.ops.object.camera_add()
    cam = bpy.context.selected_objects[0]
    bpy.context.scene.camera = cam
    camcollection.objects.link(cam)
    bpy.context.scene.collection.objects.unlink(cam)
    cam.data.lens_unit = "FOV"
    cam.data.angle = 2 * np.arctan2(1920, 2 * campose["K"][0][0])
    # attempt at changing principal point... fraction of width/height in pixels
    cam.data.shift_x = (campose["K"][0][2] - 960) / 1920
    cam.data.shift_y = (campose["K"][1][2] - 437.5) / 1920
    cam.rotation_mode = "QUATERNION"
    q = campose["R"]
    # ok blender follow opengl camera coordinates (cam faces -Z instead of +Z)
    blendq = mathutils.Quaternion([1, 0, 0, 0])
    blendq.rotate(mathutils.Euler((np.pi, 0, 0), "XYZ"))
    blendq.rotate(mathutils.Quaternion(q))
    cam.rotation_quaternion = blendq
    cam.location = mathutils.Vector(campose["t"])
    bpy.ops.view3d.camera_background_image_add(filepath=campose["img_path"], relative_path=False)
    cam.data.background_images[0].display_depth = "FRONT"
print(bpy.data.objects)
