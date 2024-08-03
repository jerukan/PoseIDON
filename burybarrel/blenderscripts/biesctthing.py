import bpy

# bpy.ops.wm.tool_set_by_id(name="builtin.bisect")
# bpy.ops.mesh.bisect(
#    plane_co=(0.0358142, -0.0578473, 0.412393),
#    plane_no=(0.0489209, -1.34198e-07, 0.998803),
#    use_fill=True,
#    clear_inner=False,
#    clear_outer=True,
#    xstart=1194,
#    xend=1684,
#    ystart=970,
#    yend=994,
#    flip=False
# )
bpy.ops.import_scene.obj(
    filepath="/Users/jerry/Projects/ms-stuff/modeling/models3d/barrel-nonozzle.obj"
)
