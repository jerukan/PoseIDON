import json
from pathlib import Path
import traceback

import click
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pyrender
from tqdm import tqdm
import transforms3d as t3d
import trimesh
import visu3d as v3d
import yaml

from burybarrel import config, get_logger
from burybarrel.transform import T_from_blender
from burybarrel.image import render_v3d, render_models, to_contour, delete_imgs_in_dir
from burybarrel.camera import load_v3dcams
from burybarrel.utils import add_to_json
from burybarrel.plotting import get_surface_line_traces, get_ray_trace


logger = get_logger(__name__)


@click.command()
@click.option(
    "-n",
    "--name",
    "names",
    required=True,
    type=click.STRING,
    multiple=True,
    help="dataset names to run. input multiple times for multiple datasets. Use 'all' to run all datasets in the input directory",
)
@click.option(
    "-i",
    "--indir",
    "datadir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    default=config.DEFAULT_DATA_DIR,
    show_default=True,
)
@click.option(
    "-o",
    "--outdir",
    "resdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    default=config.DEFAULT_RESULTS_DIR,
    show_default=True,
)
@click.option(
    "-m",
    "--modeldir",
    "objdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    default=config.DEFAULT_MODEL_DIR,
    show_default=True,
)
@click.option(
    "--render",
    "render_overlays",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Visualize the ground truth via overlays on image and primitive renderings",
)
@click.option(
    "--overwrite",
    "overwrite",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Overwrite gt data if it already exists",
)
def gt_from_blender(names, datadir, resdir, objdir, render_overlays, overwrite):
    """
    Generate ground truth information from Blender labeling. Will correspond directly
    to the COLMAP reconstruction camera poses and point cloud. If the reconstruction is
    generated again, GT will have to be labeled again.
    """
    datadir = Path(datadir)
    resdir = Path(resdir)
    objdir = Path(objdir)
    if "all" in [n.lower() for n in names]:
        alldatadirs = filter(lambda x: x.is_dir() and (x / "info.json").exists(), datadir.glob("*"))
        names = [x.name for x in alldatadirs]
    logger.info(f"Running gt_from_blender for {names}")
    for name in tqdm(names, desc="Overall datasets processing"):
        try:
            logger.info(f"Generating gt for {name}")
            generate_gt_single(name, datadir, resdir, objdir, render_overlays=render_overlays, overwrite=overwrite)
        except Exception as e:
            logger.error(f"Error in {name}: {e}\n{traceback.format_exc()}")


def generate_gt_single(name, datadir, resdir, objdir, render_overlays=True, overwrite=True):
    datadir = Path(datadir) / name
    resdir = Path(resdir) / name
    camposes_path = resdir / "colmap-out/cam_poses.json"
    gt_obj2cam_path = datadir / "gt_obj2cam.json"
    if not overwrite and gt_obj2cam_path.exists():
        logger.info(f"GT data already exists for {name}, skipping")
        return

    datasetinfopath = datadir / "info.json"
    with open(datasetinfopath, "rt") as f:
        datasetinfo = yaml.safe_load(f)
    infopath = Path("configs/blender_gt_info.yaml")
    with open(infopath, "rt") as f:
        allinfo = yaml.safe_load(f)
    info = allinfo[name]
    model_path = Path(objdir) / datasetinfo["object_name"]
    cams, imgpaths = load_v3dcams(camposes_path, img_parent=datadir / "rgb")
    scalefactor = info["scalefactor"]

    T_gt = T_from_blender(info["R"], info["t"], scalefactor)
    T_floor_gt = T_from_blender(info["R_floor"], info["t_floor"], scalefactor)
    imgs = np.array([cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB) for imgpath in imgpaths])

    mesh: trimesh.Trimesh = trimesh.load(model_path)
    meshvol = mesh.volume
    vtxs = np.array(mesh.vertices)
    rgb = np.zeros_like(vtxs, dtype=np.uint8)
    rgb[:, 0] = 255
    vtxs_p3d = v3d.Point3d(p=vtxs, rgb=rgb)
    # scaled 3d
    camscaled = cams.replace(world_from_cam=cams.world_from_cam.replace(t=cams.world_from_cam.t * (1 / scalefactor)))
    # visualization of GT
    if render_overlays:
        gtoverlaydir = datadir / "gt-overlays"
        if gtoverlaydir.exists():
            delete_imgs_in_dir(gtoverlaydir)
        gtoverlaydir.mkdir(exist_ok=True)
        plane = trimesh.creation.box(extents=(10, 10, 0.01))
        for i, img in enumerate(tqdm(imgs, desc="Rendering ground truth overlays")):
            imgpath = imgpaths[i]
            vtxs_trf = T_gt @ vtxs_p3d
            rgb, _, _ = render_models(camscaled[i], mesh, T_gt, light_intensity=200.0)
            overlayimg = to_contour(rgb, color=(255, 0, 0), background=img)
            # these won't actually be used, just for visual reference, jpg so it's smaller
            Image.fromarray(overlayimg).save(gtoverlaydir / f"{imgpath.stem}.jpg")
            rgb_primitives, _, _ = render_models(camscaled[i], [mesh, plane], [T_gt, T_floor_gt], light_intensity=200.0, flags=pyrender.RenderFlags.NONE)
            Image.fromarray(rgb_primitives).save(gtoverlaydir / f"{imgpath.stem}_primitives.jpg")
    
    # idk
    floornorm = T_floor_gt.apply_to_dir(np.array([0, 0, 1]))
    zup_mesh_T = T_floor_gt.inv @ T_gt
    zup_mesh = mesh.copy().apply_transform(zup_mesh_T.matrix4x4)
    mesh_zvals = zup_mesh.vertices[:, 2]
    zmin, zmax = np.min(mesh_zvals), np.max(mesh_zvals)
    if zmin >= 0:
        burial_ratio_z = 0
    else:
        burial_ratio_z = abs(zmin) / (abs(zmin) + zmax)
    slicedmesh = trimesh.intersections.slice_mesh_plane(zup_mesh, [0, 0, 1], [0, 0, 0], cap=True)
    burial_ratio_vol = 1 - slicedmesh.volume / meshvol
    burial_depth = abs(zmin)
    logger.info(f"vol burial: {burial_ratio_vol}, z level burial: {burial_ratio_z}, depth: {burial_depth}")

    xx, yy = np.meshgrid(np.linspace(-0.2, 0.2, 10), np.linspace(-0.2, 0.2, 10))
    zz = np.zeros_like(xx)
    raycent = np.mean(zup_mesh.vertices, axis=0)
    plane = go.Surface(x=xx, y=yy, z=zz, opacity=0.2)
    # v3d.make_fig(v3d.Point3d(p=zup_mesh_T @ mesh.vertices), plane, *get_surface_line_traces(xx, yy, zz), get_ray_trace(raycent, [0, 0, 1], color="#ff4d00", length=0.1, width=5, markersize=10))

    obj2cams_truth = [cam.world_from_cam.inv @ T_gt for cam in camscaled]
    floor2cams_truth = [cam.world_from_cam.inv @ T_floor_gt for cam in camscaled]
    gt_data_list = []
    for i, (T, T_floor) in enumerate(zip(obj2cams_truth, floor2cams_truth)):
        truthdata = {
            "img_path": str(imgpaths[i]),
            "R": T.R.tolist(),
            "t": T.t.tolist(),
            "R_floor": T_floor.R.tolist(),
            "t_floor": T_floor.t.tolist(),
        }
        gt_data_list.append(truthdata)
    with open(gt_obj2cam_path, "wt") as f:
        json.dump(gt_data_list, f, indent=4)
    add_to_json(
        {
            "burial_ratio_vol": burial_ratio_vol,
            "burial_ratio_z": burial_ratio_z,
            "burial_depth": burial_depth
        },
        datadir / "info.json"
    )
