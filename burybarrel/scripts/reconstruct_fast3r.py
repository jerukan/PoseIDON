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
    default="/scratch/jeyan/fast3r/Fast3R_ViT_Large_512",
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
    help="Max number of images to use to prevent OOM errors",
)
@click.option(
    "--overwrite",
    "overwrite",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Overwrite existing reconstructions if they exist",
)
def reconstruct_fast3r(dataset_names, data_dir, out_dir, checkpoint_dir, device, img_limit, overwrite):
    from fast3r.dust3r.inference_multiview import inference
    from fast3r.models.fast3r import Fast3R
    from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

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
    model = Fast3R.from_pretrained(checkpoint_dir)
    model = model.to(device)
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    model.eval()
    lit_module.eval()
    for dsname in dataset_names:
        singledata_dir = data_dir / dsname
        colmapcam_path = singledata_dir / "camera.json"
        img_dir = singledata_dir / "rgb"
        res_dir = out_dir / dsname / "fast3r-out"
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
        if len(imgpaths) > img_limit:
            skip = len(imgpaths) // img_limit + 1
        else:
            skip = 1
        imgpaths = imgpaths[::skip]
        imgs = imgs[::skip]
        imgnames =  [imgpath.stem for imgpath in imgpaths]
        imgpathsstr = list(map(str, imgpaths))
        fast3r_imgs = load_images(imgpathsstr, size=512, verbose=False)
        fast3r_imgs_nonorm = load_images(imgpathsstr, size=512, verbose=False, normalize=False)
        fast_w = fast3r_imgs[0]["img"][0].shape[2]
        output_dict, profiling_info = inference(
            fast3r_imgs,
            model,
            device,
            dtype=torch.float32,  # or use torch.bfloat16 if supported
            verbose=True,
            profiling=True,
        )
        poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
            output_dict["preds"],
            niter_PnP=100,
            focal_length_estimation_method="first_view_from_global_head"
        )
        # poses_c2w_batch is a list; the first element contains the estimated poses for each view.
        camera_poses = poses_c2w_batch[0]

        cams = []
        for i in range(len(fast3r_imgs)):
            idx = i
            img = fast3r_imgs_nonorm[idx]["img"][0].cpu().numpy().transpose(1, 2, 0)
            rgb = img.reshape(-1, 3)
            rgb = (rgb * 255).astype(np.uint8)
            xyz = output_dict["preds"][idx]["pts3d_in_other_view"].cpu().numpy().reshape(-1, 3)
            trimeshpc = trimesh.PointCloud(vertices=xyz, colors=rgb)
            trimeshpc.export(ply_dir / f"{imgnames[idx]}.ply")
            # convert intrinsics to the original image size
            # since focal length is in pixels, it will be scaled by the ratio of the original
            # width (1920) to the fast3r scaled width (512 px)
            fast_f = estimated_focals[0][idx]
            conv_f = (orig_w / fast_w) * fast_f
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
                world_from_cam=v3d.Transform.from_matrix(np.array(camera_poses[idx]))
            )
            cams.append(cam)
        cams = dca.stack(cams)
        save_v3dcams(cams, imgpaths, camposes_path)


"""
Stuff below was copied over from fast3r since there were some slight modifications needed
to the functions.
"""


try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False


ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(folder_or_list, size, square_ok=False, verbose=True, rotate_clockwise_90=False, crop_to_landscape=False, normalize=True):
    """open and convert all images in a list or folder to proper input format for DUSt3R"""
    if isinstance(folder_or_list, str):
        if verbose:
            print(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list

    else:
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert("RGB")
        if rotate_clockwise_90:
            img = img.rotate(-90, expand=True)
        if crop_to_landscape:
            # Crop to a landscape aspect ratio (e.g., 16:9)
            desired_aspect_ratio = 4 / 3
            width, height = img.size
            current_aspect_ratio = width / height

            if current_aspect_ratio > desired_aspect_ratio:
                # Wider than landscape: crop width
                new_width = int(height * desired_aspect_ratio)
                left = (width - new_width) // 2
                right = left + new_width
                top = 0
                bottom = height
            else:
                # Taller than landscape: crop height
                new_height = int(width / desired_aspect_ratio)
                top = (height - new_height) // 2
                bottom = top + new_height
                left = 0
                right = width

            img = img.crop((left, top, right, bottom))

        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not (square_ok) and W == H:
                halfh = 3 * halfw / 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        if verbose:
            print(f" - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")
        if normalize:
            imgs.append(
                dict(
                    img=ImgNorm(img)[None],
                    true_shape=np.int32([img.size[::-1]]),
                    idx=len(imgs),
                    instance=str(len(imgs)),
                )
            )
        else:
            imgs.append(
                dict(
                    img=tvf.ToTensor()(img)[None],
                    true_shape=np.int32([img.size[::-1]]),
                    idx=len(imgs),
                    instance=str(len(imgs)),
                )
            )
            
    assert imgs, "no images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")
    return imgs
