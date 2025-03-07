import json
import os
from pathlib import Path
import sys
from typing import List, Dict, Tuple

import dataclass_array as dca
import jax.numpy as jnp
from jax import grad
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import quaternion
from tqdm import tqdm
import trimesh
import visu3d as v3d
import yaml

sys.path.append(os.path.abspath(os.path.join("bop_toolkit")))
from bop_toolkit.bop_toolkit_lib.misc import get_symmetry_transformations

import burybarrel.colmap_util as cutil
from burybarrel.image import render_v3d, render_model, to_contour, imgs_from_dir
from burybarrel.plotting import get_axes_traces
from burybarrel.camera import load_v3dcams
from burybarrel.transform import icp, scale_T_translation, qangle, qmean, closest_quat_sym
from burybarrel.mesh import segment_pc_from_masks
from burybarrel.estimators import ransac
from burybarrel.utils import match_lists


def scale_cams(scale: float, cams: v3d.Camera):
    T = cams.world_from_cam
    return cams.replace(world_from_cam=scale_T_translation(T, scale))


def variance_from_scale(scale, data):
    """
    Implemented in JAX.

    This derivative 100% has a closed form but i'm too lazy to solve for it
    so screw it, just do gradient descent.
    """
    camTs = jnp.array(data[0])
    objTs = jnp.array(data[1])
    scaledcamTs = camTs.at[:, 0:3, 3].multiply(scale)
    centershom = scaledcamTs @ objTs @ jnp.array([0, 0, 0, 1.0])
    centers = centershom[:, :3]
    # trace of cov matrix for now, i guess
    return jnp.sum(jnp.var(centers, axis=0))


class ScaleCentroidModel():
    def __init__(self):
        self.scale = None
        self.mean = None

    def __call__(self, data):
        return self.predict(data)
    
    def fit(self, data):
        varfunc_data = lambda x: variance_from_scale(x, data)
        grad_cost = grad(varfunc_data)
        scaleinit = 1.0
        currscale = scaleinit
        currgrad = grad_cost(scaleinit)
        rate = 0.01
        eps = 1e-3
        while jnp.abs(currgrad) > eps:
            currgrad = grad_cost(currscale)
            currscale -= rate * currgrad
        self.scale = float(currscale)
        centroids = self.predict(data)
        self.mean = np.mean(centroids, axis=0)
        return self

    def predict(self, data):
        cam2worlds = data[0]
        obj2cams = data[1]
        scaledcamTs = np.copy(cam2worlds)
        scaledcamTs[:, 0:3, 3] *= self.scale
        centershom = scaledcamTs @ obj2cams @ jnp.array([0, 0, 0, 1.0])
        centers = centershom[:, :3]
        return centers


# data = (cam2world nx4x4, obj2cam nx4x4)
def fitcams(data):
    model = ScaleCentroidModel()
    model.fit(data)
    return model


def camloss(model, data):
    cents = model(data)
    return np.linalg.norm(cents - model.mean, axis=1)


def camcost(model, data):
    cents = model(data)
    return jnp.sum(jnp.var(cents, axis=0))


# data = (n quaternions)
def qloss(model, qs):
    qs = np.reshape(qs, -1)
    return np.array([qangle(model, q) for q in qs])


def qcost(model, qs):
    return np.sum(qloss(model, qs))


def fit_foundpose_multiview(
    foundpose_estimates: List[Dict],
    names: List[str],
    cameras: v3d.Camera,
    objectmesh: trimesh.Trimesh,
    masks: NDArray = None,
    scenepts: v3d.Point3d = None,
    objectsymmetries: List[Dict] = None,
    use_coarse: bool = False,
    use_icp: bool = False,
) -> Tuple[List[Dict], v3d.Transform, float]:
    cams = cameras
    # names, cameras are already assumed to be ordered and filtered
    # depending on failed registration in COLMAP or SAM masks
    name2cam = {name: cam for name, cam in zip(names, cameras)}
    # foundpose results are used as reference for image ids
    name2imgid = {}
    name2imgpath = {}
    obj2cams: v3d.Transform = []
    # camera for each hypothesis. If multiple hypotheses for each image, there will be
    # duplicate cameras in here
    camhyps: v3d.Camera = []
    # colmap usually can't reconstruct every camera pose, so we can only fit the
    # foundpose results with a camera pose
    for fres in foundpose_estimates:
        imgpath = Path(fres["img_path"])
        name = imgpath.stem
        if name not in name2imgid.keys():
            name2imgid[name] = fres["img_id"]
        if name not in name2imgpath.keys():
            name2imgpath[name] = imgpath
        # could just use the "best" hypothesis. pretty often though, this hypothesis sucks.
        # valid_hyp = name in name2cam.keys() and fres["hypothesis_id"] == "0"
        valid_hyp = name in names
        if valid_hyp:
            camhyps.append(name2cam[name])
            if use_coarse:
                R = fres["R_coarse"]
                t = fres["t_coarse"]
            else:
                R = fres["R"]
                t = fres["t"]
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = np.reshape(t, -1)
            obj2cams.append(v3d.Transform.from_matrix(T))
    obj2cams = dca.stack(obj2cams)
    camhyps = dca.stack(camhyps)

    model, inlieridxs = ransac(camhyps.world_from_cam.matrix4x4, obj2cams.matrix4x4, fit_func=fitcams, loss_func=camloss, cost_func=camcost, samp_min=5, inlier_min=5, inlier_thres=0.15, max_iter=50)

    scalefactor = model.scale
    camscaled = scale_cams(scalefactor, cams)
    camhypsscaled = scale_cams(scalefactor, camhyps)
    obj2worlds = camhypsscaled.world_from_cam @ obj2cams
    obj2worldsinlier: v3d.Transform = obj2worlds[inlieridxs]

    if use_icp:
        if scenepts is None:
            raise ValueError()
        sceneptsscaled = scenepts.replace(p=scenepts.p * scalefactor)
        segidxs = segment_pc_from_masks(sceneptsscaled, masks, camscaled, min_ratio=1/3)
        meshsamp, _ = trimesh.sample.sample_surface(objectmesh, count=len(segidxs))
        objinscenepts = sceneptsscaled[segidxs]
        tmpT = []
        for i, obj2world in enumerate(tqdm(obj2worldsinlier, desc="Running ICP on inlier poses")):
            samp_trf = obj2world @ meshsamp
            icpT = v3d.Transform.from_matrix(icp(objinscenepts.p, samp_trf))
            tmpT.append(icpT.inv @ obj2world)
        obj2worldsinlier = dca.stack(tmpT)

    quatsinlier = quaternion.from_rotation_matrix(obj2worldsinlier.R)
    ref = quatsinlier[0]
    quatssymd = [ref]
    for otherquat in quatsinlier[1:]:
        best = closest_quat_sym(ref, otherquat, objectsymmetries)
        quatssymd.append(best)
    quatssymd = np.array(quatssymd)
    obj2worldsinliersym = obj2worldsinlier.replace(R=quaternion.as_rotation_matrix(quatssymd))
    v3d.make_fig(*get_axes_traces(obj2worldsinliersym))

    qmeanransac, qinliers = ransac(quatssymd, fit_func=qmean, loss_func=qloss, cost_func=qcost, samp_min=5, inlier_min=5, inlier_thres=0.2, max_iter=50)
    qmeanransac, qinliers

    meanT = v3d.Transform(R=quaternion.as_rotation_matrix(qmeanransac), t=np.mean(obj2worldsinliersym.t, axis=0))

    obj2camfit = camscaled.world_from_cam.inv @ meanT[..., None]
    estposes = []
    for name, obj2cam in zip(names, obj2camfit):
        posedata = {
            "img_path": str(name2imgpath[name]),
            "img_id": name2imgid[name],
            "hypothesis_id": "0",
            "R": obj2cam.R.tolist(),
            "t": obj2cam.t.tolist(),
        }
        estposes.append(posedata)
    return estposes, meanT, scalefactor


def load_fit_write(datadir: Path, resdir: Path, objdir: Path, use_coarse: bool=False, use_icp: bool=False):
    datadir = Path(datadir)
    resdir = Path(resdir)
    objdir = Path(objdir)
    datainfo_path = datadir / "info.json"
    camposes_path = resdir / "colmap-out/cam_poses.json"
    scene_path = resdir / "openmvs-out/scene_dense_trimeshvalid.ply"
    foundpose_res_path = resdir / "foundpose-output/inference/estimated-poses.json"
    imgdir = datadir / "rgb"
    maskdir = resdir / "sam-masks"

    with open(objdir / "model_info.json", "rt") as f:
        objinfo = yaml.safe_load(f)
    with open(datainfo_path, "rt") as f:
        datainfo = yaml.safe_load(f)
    symTs = get_symmetry_transformations(objinfo[datainfo["object_name"]], 0.01)
    obj_path = objdir / datainfo["object_name"]
    mesh = trimesh.load(obj_path)
    trimeshpc: trimesh.PointCloud = trimesh.load(scene_path)
    scenevtxs, scenecols = trimeshpc.vertices, trimeshpc.colors[:, :3]
    scenepts = v3d.Point3d(p=scenevtxs, rgb=scenecols)
    cams, imgpaths = load_v3dcams(camposes_path, img_parent=imgdir)
    imgs = np.array([np.array(Image.open(imgpath).convert("RGB")) for imgpath in imgpaths])
    # camera names are just the filenames without extension
    imgnames = [imgpath.stem for imgpath in imgpaths]
    maskpaths, masks = imgs_from_dir(maskdir, asarray=True, grayscale=True)
    masks = masks / 255
    masknames = [maskpath.stem for maskpath in maskpaths]
    # COLMAP and SAM may not succeed for all images, so only keep registered images
    imgidxs, maskidxs = match_lists(imgnames, masknames)
    filtnames = [imgnames[i] for i in imgidxs]
    filtcams = cams[imgidxs]
    filtmasks = masks[maskidxs]
    filtimgs = imgs[imgidxs]

    # v3d.make_fig([cams, scenepts])
    with open(foundpose_res_path, "rt") as f:
        foundpose_res = yaml.safe_load(f)
    
    results, meanT, scalefactor = fit_foundpose_multiview(
        foundpose_res,
        filtnames,
        filtcams,
        mesh,
        masks=filtmasks,
        scenepts=scenepts,
        objectsymmetries=symTs,
        use_coarse=use_coarse,
        use_icp=use_icp,
    )
    camscaled = scale_cams(scalefactor, filtcams)

    coarsestr = "coarse" if use_coarse else "refine"
    icpstr = "icp" if use_icp else "noicp"
    estimate_dir = resdir / f"fit-output/est-{coarsestr}-{icpstr}"
    estimate_dir.mkdir(exist_ok=True, parents=True)
    overlaydir = estimate_dir / f"fit-overlays"
    overlaydir.mkdir(exist_ok=True)
    for i, img in enumerate(tqdm(filtimgs, desc="Rendering fit overlay results")):
        imgname = filtnames[i]
        rgb, _, _ = render_model(camscaled[i], mesh, meanT, light_intensity=40.0)
        overlayimg = to_contour(rgb, color=(255, 0, 0), background=img)
        Image.fromarray(overlayimg).save(overlaydir / f"{imgname}.png")
        # Image.fromarray(render_v3d(camscaled[i], meanT @ meshpts, radius=4, background=img)).save(overlaydir / f"{imgpaths[i].stem}.png")
    with open(estimate_dir / f"estimated-poses.json", "wt") as f:
        json.dump(results, f)
    with open(estimate_dir / f"reconstruction-info.json", "wt") as f:
        json.dump({"scalefactor": scalefactor, "obj2world": meanT.matrix4x4.tolist()}, f)
