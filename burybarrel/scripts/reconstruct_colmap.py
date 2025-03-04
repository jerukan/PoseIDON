import json
import os
from pathlib import Path
import shutil
import subprocess

import click
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import quaternion
import sqlite3
import trimesh
import visu3d as v3d
import yaml

import burybarrel.colmap_util as cutil
from burybarrel.image import imgs_from_dir
from burybarrel.camera import save_v3dcams, RadialCamera


@click.command()
@click.option(
    "-i",
    "--imgdir",
    "imgdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "-o",
    "--outdir",
    "outdir",
    required=True,
    type=click.Path(file_okay=False),
)
@click.option(
    "--sparse",
    "sparse",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Whether to run sparse reconstruction",
)
@click.option(
    "--dense",
    "dense",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Whether to run dense + mesh + texture reconstruction (requires sparse to be run first)",
)
@click.option(
    "--overwrite",
    "overwrite",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Overwrite existing COLMAP database if it exists (it complains by default)",
)
def reconstruct_colmap(img_dir, out_dir, sparse=True, dense=True, overwrite=False):
    ### colmap code ###
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    colmap_out = out_dir / "colmap-out"
    openmvs_out = out_dir / "openmvs-out"
    colmap_out.mkdir(parents=True, exist_ok=True)
    openmvs_out.mkdir(parents=True, exist_ok=True)
    database_path = colmap_out / "database.db"
    camposes_path = out_dir / "cam_poses.json"
    sparseply_path = out_dir / "sparse.ply"
    camintrinsics_path = out_dir / "camera.json"
    mvs_dir = colmap_out / "mvs"
    sparsetmp_dir = colmap_out / "sparse_models_tmp"
    sparsetmp_dir.mkdir(parents=True, exist_ok=True)

    if sparse:
        if overwrite and database_path.exists():
            database_path.unlink()
        imgpaths, imgs = imgs_from_dir(img_dir)
        # assume same size for all images (surely colmap will error if not)
        w, h = imgs[0].size
        # currently hardcoded, need to generalize it a little
        f_prior = 1300
        cx, cy = 960, 420
        camera = pycolmap.Camera(
            model=pycolmap.CameraModelId.RADIAL,
            width=w,
            height=h,
            # f, cx, cy, k1, k2
            params=[f_prior, cx, cy, 0.0, 0.0]
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
            verification_options={
                "detect_watermark": False,
                "ransac": {
                    "confidence": 0.999,
                }
            }
        )
        incrementalmapping_options = {
            "ba_refine_focal_length": True,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": True,
            "init_num_trials": 400,
            "ba_global_max_num_iterations": 100,
            "ba_global_function_tolerance": 1e-2,
            "ba_local_function_tolerance": 1e-2,
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
                "init_max_reg_trials": 4,
                "max_reg_trials": 6,
            }
        }
        maps = pycolmap.incremental_mapping(
            database_path, img_dir, sparsetmp_dir,
            options=incrementalmapping_options,
        )
        print(f"All reconstructed without prior intrinsics maps: {maps}")
        if len(maps) == 0:
            raise RuntimeError("No valid sparse reconstruction from COLMAP.")
        
        # rerun with fixed intrinsics estimated from first run (is this valid?)
        reconstruction = pycolmap.Reconstruction(sparsetmp_dir / "0")
        cams, camnames = cutil.get_cams_v3d(reconstruction, return_names=True)
        specs: RadialCamera = cams.spec
        med_f = np.median(specs.K[:, 0, 0])
        med_k1 = np.median(specs.k1k2[:, 0])
        med_k2 = np.median(specs.k1k2[:, 1])
        with pycolmap.Database(database_path) as db:
            ncams = db.num_cameras
            for i in range(ncams):
                fixedcam = pycolmap.Camera(
                    camera_id=i + 1,
                    has_prior_focal_length=True,
                    model=pycolmap.CameraModelId.RADIAL,
                    width=w,
                    height=h,
                    # f, cx, cy, k1, k2
                    params=[med_f, cx, cy, med_k1, med_k2]
                )
                db.update_camera(fixedcam)
        incrementalmapping_options["ba_refine_focal_length"] = False
        incrementalmapping_options["ba_refine_principal_point"] = False
        incrementalmapping_options["ba_refine_extra_params"] = False
        maps = pycolmap.incremental_mapping(
            database_path, img_dir, colmap_out,
            options=incrementalmapping_options
        )
        print(f"All reconstructed maps using fixed estimated intrinsics: {maps}")
        if len(maps) == 0:
            raise RuntimeError("No valid sparse reconstruction from COLMAP.")
        reconstruction = pycolmap.Reconstruction(colmap_out / "0")
        cams, camnames = cutil.get_cams_v3d(reconstruction, return_names=True)
        # saving relevant information from the reconstruction
        save_v3dcams(cams, [img_dir / name for name in camnames], camposes_path)
        camintrinsics = {
            "fx": med_f,
            "fy": med_f,
            "cx": cx,
            "cy": cy,
            "k1": med_k1,
            "k2": med_k2,
            "width": w,
            "height": h,
        }
        with open(camintrinsics_path, "wt") as f:
            json.dump(camintrinsics, f)
        pts, cols = cutil.get_pc(reconstruction)
        trimeshpc = trimesh.points.PointCloud(pts, colors=cols)
        trimeshpc.export(sparseply_path)

        if overwrite and mvs_dir.exists():
            shutil.rmtree(mvs_dir)
        # not sure the pattern colmap stores sparse maps, I presume "0" is the best
        pycolmap.undistort_images(
            mvs_dir,
            colmap_out / "0",
            img_dir,
            output_type="COLMAP",
        )

    if dense:
        # dense reconstruction
        ### openmvs code ###
        # holy crap openmvs generates so many log files
        for logpath in openmvs_out.glob("*.log"):
            logpath.unlink()
        mvs_rel = os.path.relpath(mvs_dir, openmvs_out)  # openmvs converts colmap stuff to relative paths
        print(mvs_rel)
        subprocess.run(["InterfaceCOLMAP", "-i", mvs_rel, "-o", "scene.mvs"], cwd=openmvs_out, check=True)
        subprocess.run(["DensifyPointCloud", "scene.mvs"], cwd=openmvs_out, check=True)
        subprocess.run(["ReconstructMesh", "scene_dense.mvs", "-p", "scene_dense.ply"], cwd=openmvs_out, check=True)
        subprocess.run(["RefineMesh", "scene.mvs", "-m", "scene_dense_mesh.ply", "-o", "scene_dense_mesh_refine.mvs"], cwd=openmvs_out, check=True)
        # export as obj since openmvs exports ply textures in a format blender can't read
        subprocess.run(["TextureMesh", "scene_dense.mvs", "-m", "scene_dense_mesh_refine.ply", "-o", "scene_dense_mesh_refine_texture.mvs", "--export-type", "obj"], cwd=openmvs_out, check=True)
