"""
Prints out the transforms of reference obj and plane in the scene as the yaml format
I store the GTs in.
"""
import math
import os
from pathlib import Path
import json

import bpy
import mathutils


name = bpy.context.scene.name
refobj = bpy.data.objects.get(f"ref_object-{name}")
planeobj = bpy.data.objects.get(f"plane_seafloor-{name}")

rad2deg = 180.0 / math.pi
scale = refobj.scale[0]
refx, refy, refz = refobj.location[0], refobj.location[1], refobj.location[2]
refxr, refyr, refzr = refobj.rotation_euler[0] * rad2deg, refobj.rotation_euler[1] * rad2deg, refobj.rotation_euler[2] * rad2deg
planex, planey, planez = planeobj.location[0], planeobj.location[1], planeobj.location[2]
planexr, planeyr, planezr = planeobj.rotation_euler[0] * rad2deg, planeobj.rotation_euler[1] * rad2deg, planeobj.rotation_euler[2] * rad2deg

yamltxt = f"""
{name}:
  scalefactor: {scale:.6g}
  t: [{refx:.6g}, {refy:.6g}, {refz:.6g}]
  R: [{refxr:.6g}, {refyr:.6g}, {refzr:.6g}]
  t_floor: [{planex:.6g}, {planey:.6g}, {planez:.6g}]
  R_floor: [{planexr:.6g}, {planeyr:.6g}, {planezr:.6g}]
"""

print(yamltxt)
