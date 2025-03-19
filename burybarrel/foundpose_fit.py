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
import open3d as o3d
from PIL import Image
import pyransac3d as pyrsc
import quaternion
from tqdm import tqdm
import trimesh
import visu3d as v3d
import yaml

sys.path.append(os.path.abspath(os.path.join("bop_toolkit")))
from bop_toolkit.bop_toolkit_lib.misc import get_symmetry_transformations

import burybarrel.colmap_util as cutil
from burybarrel.image import render_v3d, render_models, to_contour, imgs_from_dir
from burybarrel.plotting import get_axes_traces
from burybarrel.camera import load_v3dcams
from burybarrel.transform import icp, scale_T_translation, qangle, qmean, closest_quat_sym, get_axes_rot, T_from_translation
from burybarrel.mesh import segment_pc_from_masks
from burybarrel.estimators import ransac
from burybarrel.utils import match_lists, invert_idxs


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
    seed = None,
    resdir = None,
) -> Tuple[List[Dict], v3d.Transform, float, float]:
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

    model, inlieridxs = ransac(camhyps.world_from_cam.matrix4x4, obj2cams.matrix4x4, fit_func=fitcams, loss_func=camloss, cost_func=camcost, samp_min=5, inlier_min=5, inlier_thres=0.15, max_iter=50, seed=seed)

    scalefactor = model.scale
    camscaled = scale_cams(scalefactor, cams)
    camhypsscaled = scale_cams(scalefactor, camhyps)
    obj2worlds = camhypsscaled.world_from_cam @ obj2cams
    obj2worldsinlier: v3d.Transform = obj2worlds[inlieridxs]

    sceneptsscaled = scenepts.replace(p=scenepts.p * scalefactor)
    segidxs = segment_pc_from_masks(sceneptsscaled, masks, camscaled, min_ratio=1/2)

    # plane fit to seafloor
    floormask = np.ones(len(sceneptsscaled), dtype=bool)
    floormask[segidxs] = False
    plane1 = pyrsc.Plane()
    best_eq, best_inliers = plane1.fit(sceneptsscaled[floormask].p, thresh=0.005)
    a, b, c, d = best_eq
    normal = np.array([a, b, c])
    R = get_axes_rot([0, 0, 1], normal)
    planecent = [0, 0, -d / c]
    Tmat = np.eye(4)
    Tmat[:3, :3] = R
    Tmat[:3, 3] = planecent
    planeT = v3d.Transform.from_matrix(Tmat)
    camzup = camscaled.apply_transform(planeT.inv)
    if np.mean(camzup.world_from_cam.t[:, 2]) < 0:
        planeT = planeT @ v3d.Transform.from_matrix(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
        camzup = camscaled.apply_transform(planeT.inv)
        normal = -normal

    if use_icp:
        icpdebugdir = resdir / "icp-debug"
        icpdebugdir.mkdir(exist_ok=True, parents=True)
        # 3x points from sfm point cloud
        meshsamp, _ = trimesh.sample.sample_surface(objectmesh, count=len(segidxs))
        objinscenepts = sceneptsscaled[segidxs]
        # simple outlier removal to see if this works
        o3dobj = o3d.geometry.PointCloud()
        o3dobj.points = o3d.utility.Vector3dVector(objinscenepts.p)
        _, inlieridx = o3dobj.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
        tmpT = []
        for i, obj2world in enumerate(tqdm(obj2worldsinlier, desc="Running ICP on inlier poses")):
            samp_trf = obj2world @ meshsamp
            # source is sfm point cloud, since it is incomplete
            icpT = v3d.Transform.from_matrix(icp(objinscenepts.p[inlieridx], samp_trf, outlier_std=2.0))
            tmpT.append(icpT.inv @ obj2world)
            # debug figure (this takes a while to generate)
            meshpts = planeT.inv @ v3d.Point3d(p=samp_trf, rgb=[0, 0, 255])
            meshicppts = planeT.inv @ v3d.Point3d(p=icpT.inv @ samp_trf, rgb=[0, 255, 0])
            sceneobjzuppts = planeT.inv @ objinscenepts[inlieridx]
            sceneobjzupoutlierpts = planeT.inv @ objinscenepts[invert_idxs(inlieridx, len(objinscenepts))]
            sceneobjzupoutlierpts = sceneobjzupoutlierpts.replace(rgb=[255, 0, 0])
            xycent = np.mean(sceneobjzuppts.p, axis=0)[:2]
            centT = T_from_translation(-xycent[0], -xycent[1], 0)
            icpfig = v3d.make_fig(
                [centT @ meshpts, centT @ meshicppts, centT @ sceneobjzuppts, centT @ sceneobjzupoutlierpts],
            num_samples_point3d=1000)
            icpfig.write_image(icpdebugdir / f"icpout_{str(i).zfill(4)}.png")
        obj2worldsinlier = dca.stack(tmpT)

    quatsinlier = quaternion.from_rotation_matrix(obj2worldsinlier.R)
    ref = quatsinlier[0]
    quatssymd = [ref]
    for otherquat in quatsinlier[1:]:
        best = closest_quat_sym(ref, otherquat, objectsymmetries)
        quatssymd.append(best)
    quatssymd = np.array(quatssymd)
    obj2worldsinliersym = obj2worldsinlier.replace(R=quaternion.as_rotation_matrix(quatssymd))

    qmeanransac, qinliers = ransac(quatssymd, fit_func=qmean, loss_func=qloss, cost_func=qcost, samp_min=5, inlier_min=5, inlier_thres=0.2, max_iter=50, relax_on_fail=True, seed=seed)

    meanT = v3d.Transform(R=quaternion.as_rotation_matrix(qmeanransac), t=np.mean(obj2worldsinliersym.t, axis=0))
    quatfig = v3d.make_fig(*get_axes_traces(obj2worldsinliersym, scale=0.5), *get_axes_traces(meanT, linewidth=10))
    quatfig.write_image(resdir / "quaternion_fit.png")

    # burial ratio by fitting plane to floor point cloud
    T_zup = planeT.inv @ meanT
    meshzup = objectmesh.copy().apply_transform(T_zup.matrix4x4)
    meshzup = trimesh.intersections.slice_mesh_plane(meshzup, [0, 0, 1], [0, 0, 0], cap=True)
    burial_ratio = 1 - meshzup.volume / objectmesh.volume

    # visualization of fitted scene
    meshzuppts = v3d.Point3d(p=meshzup.vertices)
    scenezuppts = planeT.inv @ sceneptsscaled
    aggfig = v3d.make_fig([scenezuppts, meshzuppts, camzup])
    aggfig.write_image(resdir / "scene-aggregate-fit.png")

    plane2cam = camscaled.world_from_cam.inv @ planeT[..., None]
    obj2camfit = camscaled.world_from_cam.inv @ meanT[..., None]
    estposes = []
    for name, obj2cam in zip(names, obj2camfit):
        posedata = {
            "img_path": str(name2imgpath[name]),
            "img_id": name2imgid[name],
            "hypothesis_id": "0",
            "R": obj2cam.R.tolist(),
            "t": obj2cam.t.tolist(),
            "R_floor": plane2cam.R.tolist(),
            "t_floor": plane2cam.t.tolist(),
        }
        estposes.append(posedata)
    return estposes, meanT, scalefactor, planeT, burial_ratio


def load_fit_write(datadir: Path, resdir: Path, objdir: Path, use_coarse: bool=False, use_icp: bool=False, seed=None):
    # existing dirs
    datadir = Path(datadir)
    resdir = Path(resdir)
    objdir = Path(objdir)
    datainfo_path = datadir / "info.json"
    camposes_path = resdir / "colmap-out/cam_poses.json"
    scene_path = resdir / "openmvs-out/scene_dense_trimeshvalid.ply"
    foundpose_res_path = resdir / "foundpose-output/inference/estimated-poses.json"
    imgdir = datadir / "rgb"
    maskdir = resdir / "sam-masks"
    # dir setup
    coarsestr = "coarse" if use_coarse else "refine"
    icpstr = "icp" if use_icp else "noicp"
    estimate_dir = resdir / f"fit-output/est-{coarsestr}-{icpstr}"
    estimate_dir.mkdir(exist_ok=True, parents=True)

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
    
    results, meanT, scalefactor, planeT, burial_ratio = fit_foundpose_multiview(
        foundpose_res,
        filtnames,
        filtcams,
        mesh,
        masks=filtmasks,
        scenepts=scenepts,
        objectsymmetries=symTs,
        use_coarse=use_coarse,
        use_icp=use_icp,
        seed=seed,
        resdir=estimate_dir
    )
    camscaled = scale_cams(scalefactor, filtcams)

    overlaydir = estimate_dir / f"fit-overlays"
    overlaydir.mkdir(exist_ok=True)
    plane = trimesh.creation.box(extents=(10, 10, 0.01))
    for i, img in enumerate(tqdm(filtimgs, desc="Rendering fit overlay results")):
        imgname = filtnames[i]
        rgb, _, _ = render_models(camscaled[i], mesh, meanT, light_intensity=40.0)
        overlayimg = to_contour(rgb, color=(255, 0, 0), background=img)
        Image.fromarray(overlayimg).save(overlaydir / f"{imgname}.jpg")
        # Image.fromarray(render_v3d(camscaled[i], meanT @ meshpts, radius=4, background=img)).save(overlaydir / f"{imgpaths[i].stem}.png")
        rgb_primitives, _, _ = render_models(camscaled[i], [mesh, plane], [meanT, planeT], light_intensity=200.0)
        Image.fromarray(rgb_primitives).save(overlaydir / f"{imgname}_primitives.jpg")
    with open(estimate_dir / f"estimated-poses.json", "wt") as f:
        json.dump(results, f, indent=4)
    otherresults = {
        "scalefactor": scalefactor,
        "burial_ratio": burial_ratio,
        "obj2world": meanT.matrix4x4.tolist(),
        "use_coarse": use_coarse,
        "use_icp": use_icp
    }
    with open(estimate_dir / f"reconstruction-info.json", "wt") as f:
        json.dump(otherresults, f, indent=4)
