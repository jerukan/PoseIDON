import os
from pathlib import Path

import bpy

barrel_obj_path = Path("barrels/data/models3d/barrel-nonozzle.obj")

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

bpy.ops.import_scene.obj(filepath=str(barrel_obj_path))
barrelobj = bpy.context.selected_objects[0]
print(f"{barrelobj.name}: {barrelobj}")
