"""
Synthetic data inference test. Need to run in background since ICP is omega slow.
"""

import os
from pathlib import Path

import dill as pickle
import numpy as np
import torch.utils.data
from tqdm import tqdm

from burybarrel.transform import icp_translate
from burybarrel.barrelnet.barrelnet import BarrelNet
from burybarrel.barrelnet.data import pts2inference_format
from burybarrel.synthbarrel import Cylinder


def run():
    print("opening test dataset. this might take a while")
    with open("data/synthbarrel/testbarrels_1000_fixed.pkl", "rb") as f:
        synthdict = pickle.load(f)
    print(synthdict.keys())

    ## Load Model
    model_path = "checkpoints/pointnet_iter80_fixed.pth"
    pointnet = BarrelNet(k=5, normal_channel=False)
    pointnet.load_state_dict(torch.load(model_path))
    pointnet.cuda().eval()

    # cylnp = random_cylinder_surf([0, 0, 0], [0, 0, height_ratio], 1, 5000).astype(np.float32)
    # radius predicted: fraction of height
    # normalized space: height is fixed at 1
    # height_ratio = 2.5  # height / radius ratio
    cylh = 1
    ntrials = synthdict["radii"].shape[0]

    trialresults = []
    for i in tqdm(range(ntrials)):
        results = {}
        cylnp = synthdict["pts"][i].numpy()
        axtruth = synthdict["axis_vectors"][i]
        rtruth = synthdict["radii"][i].numpy()
        # height in generated data is fixed at 1
        yoffsettruth = synthdict["burial_offsets"][i]
        cyltruth = Cylinder.from_axis(axtruth, rtruth, 1, c=[0, yoffsettruth, 0])
        
        results["cyltruth"] = cyltruth
        results["burialtruth"] = cyltruth.get_volume_ratio_monte(5000, planecoeffs=[0, 1, 0, 0])

        axis_pred, r, h, y = pointnet.predict_np(cylnp, height_radius_ratio=1/rtruth)
        
        cylpred = Cylinder.from_axis(axis_pred, r, h, c=[0, y, 0])
        predsurfpts = cylpred.get_random_pts_surf(5000)
        translation = icp_translate(cylnp, predsurfpts, max_iters=5, ntheta=0, nphi=0)
        cylpred = cylpred.translate(-translation)
        
        results["cylpred"] = cylpred
        results["burialpred"] = cylpred.get_volume_ratio_monte(5000, planecoeffs=[0, 1, 0, 0])

        # print("ahAHSFHJKSADHJKFSDHJKDFSHJKFSAD")
        # print(axis_pred, r, h, y)
        # print(axtruth, rtruth, h, yoffsettruth / h)
        
        trialresults.append(results)

        # print("TRUTH")
        # print(f"axis: {cylax}\nradius: {cylr}\nheight: {cylh}\nz-offset: {cylz}")
        # print(f"burial percentage: {burialtruth}")
        # print("PREDICTED")
        # print(radius_pred, zshift_pred, axis_pred)
        # print(f"axis: {axis_pred}\nradius: {r}\nheight: {h}\nz-offset: {z}")
        # print(f"burial percentage: {burialpred}")

        # truthray = v3d.Ray(pos=[0,0,0], dir=cylax)
        # predray = v3d.Ray(pos=[0,0,0], dir=axis_pred)
        # v3d.make_fig([v3d.Point3d(p=cylnp), truthray, predray])
    with open("results/pointnet_synth_results.pkl", "wb") as f:
        pickle.dump(trialresults, f)
