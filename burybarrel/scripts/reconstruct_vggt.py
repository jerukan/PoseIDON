import json
import os
from pathlib import Path
import shutil
import subprocess

import click
import dataclass_array as dca
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import pycolmap
import quaternion
import sqlite3
from tqdm import tqdm
import torch
import torchvision.transforms as tvf
import trimesh
import visu3d as v3d
import yaml

from burybarrel import get_logger, add_file_handler, log_dir
from burybarrel.config import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR
import burybarrel.colmap_util as cutil
from burybarrel.image import imgs_from_dir
from burybarrel.camera import save_v3dcams, RadialCamera
from burybarrel.mesh import load_mesh


logger = get_logger(__name__)


@click.command()
@click.option(
    "-n",
    "--name",
    "dataset_names",
    required=True,
    type=click.STRING,
    help="Names of all datasets to process in data_dir. Use 'all' to run all valid datasets in data_dir",
    multiple=True,
)
@click.option(
    "-d",
    "--datadir",
    "data_dir",
    default=DEFAULT_DATA_DIR,
    required=True,
    type=click.Path(exists=True, file_okay=False),
    show_default=True,
    help="Directory containing all datasets",
)
@click.option(
    "-o",
    "--outdir",
    "out_dir",
    default=DEFAULT_RESULTS_DIR,
    required=True,
    type=click.Path(file_okay=False),
    show_default=True,
    help="Output directory for all results",
)
@click.option(
    "--checkpoint",
    "checkpoint_dir",
    default="/scratch/jeyan/vggt/VGGT-1B",
    required=True,
    type=click.STRING,
    show_default=True,
    help="Output directory for all results",
)
@click.option(
    "-d",
    "--device",
    "device",
    type=click.STRING,
    help="cuda device"
)
@click.option(
    "--img-limit",
    "img_limit",
    default=20,
    required=True,
    type=click.INT,
    show_default=True,
    help="unused rn probably (does VGGT have the same issues?)",
)
@click.option(
    "--overwrite",
    "overwrite",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Overwrite existing reconstructions if they exist",
)
def reconstruct_vggt(dataset_names, data_dir, out_dir, checkpoint_dir, device, img_limit, overwrite):
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    if "all" in [n.lower() for n in dataset_names]:
        dataset_names = []
        alldatapaths = data_dir.glob("*")
        for datapath in alldatapaths:
            if datapath.is_dir() and (datapath / "info.json").exists():
                dsname = datapath.name
                dataset_names.append(dsname)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained(checkpoint_dir).to(device)
    for dsname in tqdm(dataset_names):
        singledata_dir = data_dir / dsname
        colmapcam_path = singledata_dir / "camera.json"
        img_dir = singledata_dir / "rgb"
        res_dir = out_dir / dsname / "vggt-out"
        ply_dir = res_dir / "pc_ply"
        camposes_path = res_dir / "cam_poses.json"
        if camposes_path.exists() and not overwrite:
            logger.info(f"Skipping {dsname} since cam_poses.json already exist at {camposes_path}")
            continue
        res_dir.mkdir(parents=True, exist_ok=True)
        ply_dir.mkdir(parents=True, exist_ok=True)
        with open(colmapcam_path, "rt") as f:
            colmapcaminfo = yaml.safe_load(f)
        imgpaths, imgs = imgs_from_dir(img_dir, sortnames=True)
        orig_w, orig_h = imgs[0].width, imgs[0].height
        orig_cx, orig_cy = colmapcaminfo["cx"], colmapcaminfo["cy"]
        skip = 1
        # if len(imgpaths) > img_limit:
        #     skip = len(imgpaths) // img_limit
        imgpaths = imgpaths[::skip]
        imgs = imgs[::skip]
        imgnames =  [imgpath.stem for imgpath in imgpaths]
        imgpathsstr = list(map(str, imgpaths))
        vggt_imgs = load_and_preprocess_images(imgpathsstr).to(device)
        scale_w = vggt_imgs.shape[3]
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                predictions = model(vggt_imgs)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], predictions["images"].shape[-2:])
        # nx3x4 extrinsics (note this is world2cam, not cam2world, invert later)
        extrinsics = extrinsics[0].cpu().numpy()
        intrinsics = intrinsics[0].cpu().numpy()
        extrinsicbottom = np.zeros((extrinsics.shape[0], 1, 4))
        extrinsicbottom[..., -1] = 1
        extrinsics4 = np.concatenate([extrinsics, extrinsicbottom], axis=-2)
        cams = []
        for i in range(len(vggt_imgs)):
            idx = i
            img = vggt_imgs[i].cpu().numpy().transpose(1, 2, 0)
            rgb = img.reshape(-1, 3)
            rgb = (rgb * 255).astype(np.uint8)
            xyz = predictions["world_points"][0][i].cpu().numpy().reshape(-1, 3)
            trimeshpc = trimesh.PointCloud(vertices=xyz, colors=rgb)
            trimeshpc.export(ply_dir / f"{imgnames[idx]}.ply")
            # convert intrinsics to the original image size
            # since focal length is in pixels, it will be scaled by the ratio of the original
            # width (1920) to the vggt scaled width (518 px)
            scale_f = intrinsics[idx][0, 0]
            conv_f = (orig_w / scale_w) * scale_f
            spec = v3d.PinholeCamera(
                K=[
                    [conv_f, 0, orig_cx],
                    [0, conv_f, orig_cy],
                    [0, 0, 1]
                ],
                resolution=(orig_h, orig_w),
            )
            cam = v3d.Camera(
                spec=spec,
                # invert!
                world_from_cam=v3d.Transform.from_matrix(np.array(extrinsics4[idx])).inv
            )
            cams.append(cam)
        cams = dca.stack(cams)
        save_v3dcams(cams, imgpaths, camposes_path)
