import os
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import quaternion
import sqlite3
import visu3d as v3d
import yaml

import burybarrel.colmap_util as cutil
from burybarrel.image import imgs_from_dir
from burybarrel.camera import save_v3dcams


def run(img_dir, out_dir, overwrite=False):
    ### colmap code ###
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    colmap_out = out_dir / "colmap-out"
    openmvs_out = out_dir / "openmvs-out"
    colmap_out.mkdir(parents=True, exist_ok=True)
    openmvs_out.mkdir(parents=True, exist_ok=True)
    database_path = colmap_out / "database.db"
    camposes_path = out_dir / "cam_poses.json"
    if overwrite and database_path.exists():
        database_path.unlink()
    mvs_dir = colmap_out / "mvs"

    imgpaths, imgs = imgs_from_dir(img_dir)
    # assume same size for all images (surely colmap will error if not)
    w, h = imgs[0].size
    camera = pycolmap.Camera(
        model=pycolmap.CameraModelId.RADIAL,
        width=1920,
        height=875,
        # f, cx, cy, k1, k2
        params=[1300, 960, 420, 0.0, 0.0]
    )
    pycolmap.extract_features(
        database_path,
        img_dir,
        camera_model=camera.model.name,
        reader_options={"camera_model": camera.model.name, "camera_params": camera.params_to_string()},
        sift_options={
            # DSP-SIFT is presumably better
            "domain_size_pooling": True,
            "edge_threshold":  5.0,
            "peak_threshold":  1 / 200,
            "max_num_orientations": 3,
            "num_octaves": 8,
            "octave_resolution": 6,
            "num_threads": 4,
            "estimate_affine_shape": True,
            "dsp_max_scale": 6.0,
            "dsp_min_scale": 0.08,
            "dsp_num_scales": 20,
        }
    )
    pycolmap.match_exhaustive(
        database_path,
        matching_options={"block_size": 10},
        verification_options={
            "detect_watermark": False
        }
    )
    maps = pycolmap.incremental_mapping(
        database_path, img_dir, colmap_out,
        options={
            "ba_refine_focal_length": True,
            "ba_refine_principal_point": True,
            "ba_refine_extra_params": True,
            "ba_global_function_tolerance": 1e-2,
            "ba_local_function_tolerance": 1e-2,
            "init_num_trials": 1000,
            "ba_global_max_num_iterations": 200,
            # just double all the thresholds lol
            # surely this won't go horribly
            "mapper": {
                "abs_pose_max_error": 24.0,
                "abs_pose_min_inlier_ratio": 0.1,
                "abs_pose_min_num_inliers": 10,
                "filter_max_reproj_error": 8.0,
                "filter_min_tri_angle": 3.0,
                "init_max_error": 8.0,
                "init_max_reg_trials": 4,
                "init_min_num_inliers": 20,
            }
        }
    )
    print(f"All reconstructed maps: {maps}")
    if len(maps) == 0:
        raise RuntimeError("No valid sparse reconstruction from COLMAP.")

    # dense reconstruction
    if overwrite and mvs_dir.exists():
        shutil.rmtree(mvs_dir)
    # not sure the pattern colmap stores sparse maps, I presume "0" is the best
    pycolmap.undistort_images(
        mvs_dir,
        colmap_out / "0",
        img_dir,
        output_type="COLMAP",
    )
    reconstruction = pycolmap.Reconstruction(colmap_out / "0")
    cams, camnames = cutil.get_cams_v3d(reconstruction, return_names=True)
    save_v3dcams(cams, [img_dir / name for name in camnames], camposes_path)

    ### openmvs code ###
    mvs_rel = os.path.relpath(mvs_dir, openmvs_out)  # openmvs converts colmap stuff to relative paths
    print(mvs_rel)
    subprocess.run(["InterfaceCOLMAP", "-i", mvs_rel, "-o", "scene.mvs"], cwd=openmvs_out, check=True)
    subprocess.run(["DensifyPointCloud", "scene.mvs"], cwd=openmvs_out, check=True)
    subprocess.run(["ReconstructMesh", "scene_dense.mvs", "-p", "scene_dense.ply"], cwd=openmvs_out, check=True)
    subprocess.run(["RefineMesh", "scene.mvs", "-m", "scene_dense_mesh.ply", "-o", "scene_dense_mesh_refine.mvs"], cwd=openmvs_out, check=True)
    # export as obj since openmvs exports ply textures in a format blender can't read
    subprocess.run(["TextureMesh", "scene_dense.mvs", "-m", "scene_dense_mesh_refine.ply", "-o", "scene_dense_mesh_refine_texture.mvs", "--export-type", "obj"], cwd=openmvs_out, check=True)
