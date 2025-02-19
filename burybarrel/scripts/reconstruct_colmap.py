import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import sqlite3
import visu3d as v3d

import burybarrel.colmap_util as cutil


def run(img_dir, out_dir, overwrite=False):
    # camera = pycolmap.Camera(
    #     model="RADIAL",
    #     width=1920,
    #     height=875,
    #     params=[1246, 960, 420, -0.123, -0.015]
    # )
    # camera = pycolmap.Camera(
    #     model="SIMPLE_PINHOLE",
    #     width=1920,
    #     height=875,
    #     # params=[1246, 960, 420],
    #     params=[500, 960, 420],
    # )
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    database_path = out_dir / "database.db"
    if overwrite and database_path.exists():
        database_path.unlink()
    mvs_dir = out_dir / "mvs"
    camera = pycolmap.Camera(
        model="PINHOLE",
        width=1920,
        height=875,
        params=[1246, 1246, 960, 420],
        # params=[500, 500, 960, 420],
    )
    pycolmap.extract_features(
        database_path,
        img_dir,
        camera_model=camera.model.name,
        reader_options={"camera_model": camera.model.name, "camera_params": camera.params_to_string()},
        sift_options={
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
        database_path, img_dir, out_dir,
        options={
            "ba_global_function_tolerance": 1e-2,
            "ba_local_function_tolerance": 1e-2,
            "init_num_trials": 400,
            # "init_image_id1": 1,
            # "init_image_id2": 2,
        }
    )
    # dense reconstruction
    pycolmap.undistort_images(
        mvs_dir,
        out_dir / "0",
        img_dir,
        output_type="COLMAP",
    )
