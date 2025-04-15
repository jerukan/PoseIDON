import json
import math
import os
from pathlib import Path
from typing import List
import sys

import click
import cv2
import dataclass_array as dca
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from PIL import Image
import pandas as pd
import pycolmap
import pyrender
import tqdm
import trimesh
import visu3d as v3d
import yaml

from bop_toolkit.bop_toolkit_lib.pose_error import vsd, mssd, mspd
from bop_toolkit.bop_toolkit_lib.misc import get_symmetry_transformations
from bop_toolkit.bop_toolkit_lib.renderer import create_renderer

from burybarrel import config
import burybarrel.colmap_util as cutil
from burybarrel.image import render_v3d, imgs_from_dir


@click.command()
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
    "--best-hyp",
    "rankbest_hyp",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="choose the best hypothesis with ground truth knowledge (for raw foundpose results) (this will obviously skew to better performance); otherwise, just choose the 0th hypothesis",
)
def get_metrics(datadir, resdir, objdir, rankbest_hyp=False):
    # info about rankbest_hyp:
    # if estimations have multiple hypotheses per image, setting this true will choose the
    # best hypothesis with ground truth knowledge (this will obviously skew to better performance)
    # otherwise, just choose the 0th hypothesis

    # refactor?
    objectdir = objdir

    with open(objectdir / "model_info.json", "rt") as f:
        allobjectinfo: dict = json.load(f)
    allresdirs = list(filter(lambda x: x.is_dir(), resdir.glob("*")))

    renderer = create_renderer(1920, 875, renderer_type="vispy", mode="depth")
    for modelname in allobjectinfo.keys():
        renderer.add_object(modelname, objectdir / modelname)

    allestmetrics = []
    for singleresdir in tqdm.tqdm(allresdirs):
        print(f"processing results for following dataset: {singleresdir}")
        dataname = singleresdir.name
        singledatadir = datadir / dataname
        fitoutdir = singleresdir / "fit-output"
        if not fitoutdir.exists():
            print(f"Skipping {dataname} because fit-output does not exist")
            continue

        with open(singledatadir / "gt_obj2cam.json", "rt") as f:
            gtposes = yaml.safe_load(f)
        with open(singledatadir / "info.json", "rt") as f:
            datainfo = yaml.safe_load(f)
        with open(singledatadir / "camera.json", "rt") as f:
            caminfo = yaml.safe_load(f)
        # gt masks (may not exist)
        masksdir = singledatadir / "mask"
        maskpaths = None
        masks = None
        if masksdir.exists():
            maskpaths, masks = imgs_from_dir(masksdir, grayscale=True, asarray=True)
            if len(masks) != len(gtposes):
                maskpaths = None
                masks = None
            else:
                masks = masks / 255
        # predicted SAM masks
        sammasksdir = singleresdir / "sam-masks"
        sammaskpaths, sammasks = imgs_from_dir(sammasksdir, grayscale=True, asarray=True)
        sammasks = sammasks / 255

        gt_Rs = np.array([gtpose["R"] for gtpose in gtposes])
        gt_ts = np.array([gtpose["t"] for gtpose in gtposes])[..., None]
        imgnames = [Path(gtpose["img_path"]).name for gtpose in gtposes]

        fx, fy, cx, cy = caminfo["fx"], caminfo["fy"], caminfo["cx"], caminfo["cy"]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1.0]
        ], dtype=float)

        object_name = datainfo["object_name"]
        symTs = get_symmetry_transformations(allobjectinfo[object_name], 0.01)

        mesh: trimesh.Trimesh = trimesh.load(objectdir / object_name)
        vtxs = np.array(mesh.vertices)

        # raw foundpose metrics
        print(f"evaluating raw foundpose results")
        posepath = singleresdir / "foundpose-output/inference/estimated-poses.json"
        with open(posepath, "rt") as f:
            ests = yaml.safe_load(f)
        foundposecoarsevsd = []
        foundposecoarsemssd = []
        foundposecoarsemspd = []
        foundposerefvsd = []
        foundposerefmssd = []
        foundposerefmspd = []
        for i, imgname in enumerate(imgnames):
            imgmatches = get_imgmatches(ests, imgname)
            # will fail if masking in foundpose fails
            if len(imgmatches) == 0:
                continue
            R_gt = gt_Rs[i]
            t_gt = gt_ts[i]
            coarsevsd, coarsemssd, coarsemspd = evaluate_singleest(
                ests, imgname, R_gt, t_gt, K, renderer, vtxs, object_name, coarse=True, syms=symTs, rankbest_hyp=rankbest_hyp, mask_gt=masks[i] if masks is not None else None
            )
            refvsd, refmssd, refmspd = evaluate_singleest(
                ests, imgname, R_gt, t_gt, K, renderer, vtxs, object_name, coarse=False, syms=symTs, rankbest_hyp=rankbest_hyp, mask_gt=masks[i] if masks is not None else None
            )
            foundposecoarsevsd.append(coarsevsd)
            foundposecoarsemssd.append(coarsemssd)
            foundposecoarsemspd.append(coarsemspd)
            foundposerefvsd.append(refvsd)
            foundposerefmssd.append(refmssd)
            foundposerefmspd.append(refmspd)
        coarseestmetrics = {
            "dataset": dataname,
            "avg_vsd": float(np.mean(foundposecoarsevsd)),
            "avg_mssd": float(np.mean(foundposecoarsemssd)),
            "avg_mspd": float(np.mean(foundposecoarsemspd)),
            "multiview_fitted": False,
            "pose_type": "coarse",
            "use_icp": False,
            "burial_error_vol": -1,
            "burial_error_z": -1,
            "burial_error_depth": -1,
            "burial_ratio_vol": -1,
            "burial_ratio_z": -1,
            "burial_depth": -1,
            "burial_ratio_vol_gt": -1,
            "burial_ratio_z_gt": -1,
            "burial_depth_gt": -1,
            "resdir": str(singleresdir / "foundpose-output/inference"),
            "datadir": str(singledatadir),
            "description": datainfo["description"],
            "lat": datainfo["lat"],
            "lon": datainfo["lon"],
            "depth": datainfo["depth"],
            "object_name": datainfo["object_name"],
        }
        refestmetrics = {
            "dataset": dataname,
            "avg_vsd": float(np.mean(foundposerefvsd)),
            "avg_mssd": float(np.mean(foundposerefmssd)),
            "avg_mspd": float(np.mean(foundposerefmspd)),
            "multiview_fitted": False,
            "pose_type": "refined",
            "use_icp": False,
            "burial_error_vol": -1,
            "burial_error_z": -1,
            "burial_error_depth": -1,
            "burial_ratio_vol": -1,
            "burial_ratio_z": -1,
            "burial_depth": -1,
            "burial_ratio_vol_gt": -1,
            "burial_ratio_z_gt": -1,
            "burial_depth_gt": -1,
            "resdir": str(singleresdir / "foundpose-output/inference"),
            "datadir": str(singledatadir),
            "description": datainfo["description"],
            "lat": datainfo["lat"],
            "lon": datainfo["lon"],
            "depth": datainfo["depth"],
            "object_name": datainfo["object_name"],
        }
        allestmetrics.append(coarseestmetrics)
        allestmetrics.append(refestmetrics)

        # multiview fit metrics
        allfitdirs = list(filter(lambda x: x.is_dir(), fitoutdir.glob("*")))
        for fitdir in allfitdirs:
            print(f"evaluating fitted results in {fitdir}")
            posepath = fitdir / "estimated-poses.json"
            estinfopath = fitdir / "reconstruction-info.json"
            with open(posepath, "rt") as f:
                ests = yaml.safe_load(f)
            with open(estinfopath, "rt") as f:
                estinfo = yaml.safe_load(f)
            allvsd = []
            allmssd = []
            allmspd = []
            for i, imgname in enumerate(imgnames):
                imgmatches = get_imgmatches(ests, imgname)
                if len(imgmatches) == 0:
                    continue
                R_gt = gt_Rs[i]
                t_gt = gt_ts[i]
                # rankbest_hyp is false since multiview fit already uses all hypotheses
                # thus there is only 1 hypothesis per image
                imgvsd, imgmssd, imgmspd = evaluate_singleest(
                    ests, imgname, R_gt, t_gt, K, renderer, vtxs, object_name, syms=symTs, rankbest_hyp=False, mask_gt=masks[i] if masks is not None else None
                )
                allvsd.append(imgvsd)
                allmssd.append(imgmssd)
                allmspd.append(imgmspd)
            estmetrics = {
                "dataset": dataname,
                "avg_vsd": float(np.mean(allvsd)),
                "avg_mssd": float(np.mean(allmssd)),
                "avg_mspd": float(np.mean(allmspd)),
                "multiview_fitted": True,
                "pose_type": "coarse" if estinfo["use_coarse"] else "refined",
                "use_icp": estinfo["use_icp"],
                "burial_error_vol": abs(estinfo["burial_ratio_vol"] - datainfo["burial_ratio_vol"]),
                "burial_error_z": abs(estinfo["burial_ratio_z"] - datainfo["burial_ratio_z"]),
                "burial_error_depth": abs(estinfo["burial_depth"] - datainfo["burial_depth"]),
                "burial_ratio_vol": float(estinfo["burial_ratio_vol"]),
                "burial_ratio_z": float(estinfo["burial_ratio_z"]),
                "burial_depth": float(estinfo["burial_depth"]),
                "burial_ratio_vol_gt": float(datainfo["burial_ratio_vol"]),
                "burial_ratio_z_gt": float(datainfo["burial_ratio_z"]),
                "burial_depth_gt": float(datainfo["burial_depth"]),
                "resdir": str(fitdir),
                "datadir": str(singledatadir),
                "description": datainfo["description"],
                "lat": datainfo["lat"],
                "lon": datainfo["lon"],
                "depth": datainfo["depth"],
                "object_name": datainfo["object_name"],
                # "all_vsd": np.array(allvsd).tolist(),
                # "all_mssd": np.array(allmssd).tolist(),
                # "all_mspd": np.array(allmspd).tolist(),
            }
            allestmetrics.append(estmetrics)

    df = pd.DataFrame.from_records(allestmetrics)
    with open(resdir / "allmetrics.csv", "wt") as f:
        df.to_csv(f, index=False)


def get_imgmatches(ests, imgname):
    return list(filter(lambda x: Path(x["img_path"]).name == imgname, ests))


def evaluate_singleest(ests, imgname, R_gt, t_gt, K, renderer, vtxs, object_name, coarse=False, syms=None, rankbest_hyp=False, mask_gt=None):
    if syms is None:
        syms = []
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # no test depth, just use ground truth with ground truth mask
    depth_test = renderer.render_object(object_name, R_gt, t_gt, fx, fy, cx, cy)["depth"]
    if mask_gt is not None:
        depth_test = mask_gt * depth_test
    imgmatches = get_imgmatches(ests, imgname)
    imgvsd = []
    imgmssd = []
    imgmspd = []
    for j, imgmatch in enumerate(imgmatches):
        if not rankbest_hyp:
            if imgmatch["hypothesis_id"] != "0":
                continue
        if coarse:
            R_est = np.array(imgmatch["R_coarse"])
            t_est = np.array(imgmatch["t_coarse"])
        else:
            R_est = np.array(imgmatch["R"])
            t_est = np.array(imgmatch["t"])
        vsdres = vsd(R_est, t_est, R_gt, t_gt, depth_test, K, 0.2, [0.2], False, None, renderer, object_name, "step")
        mssdres = mssd(R_est, t_est, R_gt, t_gt, vtxs, syms)
        mspdres = mspd(R_est, t_est.reshape(3, 1), R_gt, t_gt, K, vtxs, syms)
        imgvsd.append(vsdres[0])
        imgmssd.append(mssdres)
        imgmspd.append(mspdres)
    # choose hypothesis with majority best metric between vsd, mssd, mspd
    winnings = np.zeros(len(imgmatches), dtype=int)
    winnings[np.argmin(imgvsd)] += 1
    winnings[np.argmin(imgmssd)] += 1
    winnings[np.argmin(imgmspd)] += 1
    probablybest = np.argmax(winnings)
    return imgvsd[probablybest], imgmssd[probablybest], imgmspd[probablybest]
