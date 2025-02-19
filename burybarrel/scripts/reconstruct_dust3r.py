import os
from pathlib import Path
import sys

import torch
from torch import nn
import trimesh
import visu3d as v3d

sys.path.append(os.path.abspath(os.path.join("dust3r")))
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from burybarrel.dust3r_utils import save_dust3r_outs, read_dust3r, resize_to_dust3r


def run(
    model_path,
    img_dir,
    out_dir,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(model_path)
    dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    imgpaths = sorted(list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png")))
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    dust3rout_path = out_dir / "dust3r_out.pth"
    ply_dir = out_dir / "plys"
    ply_dir.mkdir(exist_ok=True, parents=True)

    batch_size = 8
    schedule = "cosine"
    lr = 0.01
    niter = 300
    images = load_images(list(map(str, imgpaths)), size=512)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    # pairs = make_pairs(images, scene_graph="swin-5-noncyclic", prefilter=None, symmetrize=True)
    output = inference(pairs, dust3r_model, device, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=True)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    outdict = save_dust3r_outs(scene, dust3rout_path)

    pc_final, pcs_each, v3dcams = read_dust3r(dust3rout_path, flatten=False)

    pcd = trimesh.points.PointCloud(pc_final.p, pc_final.rgb)
    pcd.export(ply_dir / f"pts_agg.ply")
    for i, imgpc in enumerate(pcs_each):
        pcd = trimesh.points.PointCloud(imgpc.p.reshape(-1, 3), imgpc.rgb.reshape(-1, 3))
        pcd.export(ply_dir / f"{imgpaths[i].stem}_pts.ply")
