"""
Everything 3D model. Meshes, point clouds, and more.
"""
from pathlib import Path
from typing import Union

import numpy as np
from plyfile import PlyData
import trimesh
import visu3d as v3d


def load_mesh(path) -> trimesh.Geometry:
    """
    Loads a mesh from a file using trimesh.

    I wrote this specifically because OpenMVS creates funky dense point cloud plys that
    trimesh can't read.
    """
    path = Path(path)
    if path.suffix == ".ply":
        try:
            return trimesh.load(path)
        except ValueError:
            print(f"Trimesh thinks {path} is corrupt. Trying to load with plyfile...")
            with open(path, "rb") as f:
                plydata = PlyData.read(f)
            if "face" in plydata:
                raise ValueError("I didn't implement loading corrupt ply meshes yet, only point clouds")
            vertexdata = plydata["vertex"]
            vertices = np.vstack([vertexdata["x"], vertexdata["y"], vertexdata["z"]]).T
            colors = np.vstack([vertexdata["red"], vertexdata["green"], vertexdata["blue"]]).T
            return trimesh.PointCloud(vertices=vertices, colors=colors)
    return trimesh.load(path)


def segment_pc_from_mask(pc: Union[v3d.Point3d, np.ndarray], mask: np.ndarray, v3dcam: v3d.Camera):
    """
    Segments a point cloud given a mask and camera.

    Args:
        pc (mx3 array): point cloud
        mask (hxw array): binary mask (float or uint8 should work)
        v3dcam (v3d.Camera): single camera

    Returns:
        np.ndarray: The indices of the segmented points.
    """
    if not isinstance(pc, v3d.Point3d):
        pc = v3d.Point3d(p=pc)
    idxs = np.arange(pc.size)
    H, W = v3dcam.spec.resolution
    if H != mask.shape[0] or W != mask.shape[1]:
        raise ValueError(f"Mask HxW {mask.shape} does not match camera HxW {H}x{W}")
    pxpts = v3dcam.px_from_world @ pc
    uvs = pxpts.p
    valid = (uvs[:, 0] >= 0) & (uvs[:, 0] <= W) & (uvs[:, 1] >= 0) & (uvs[:, 1] <= H)
    validuvs = uvs[valid].astype(int)
    uvmaskvals = mask[validuvs.T[1], validuvs.T[0]] > 0
    segidxs = idxs[valid][uvmaskvals]
    return segidxs


def segment_pc_from_masks(pc: Union[v3d.Point3d, np.ndarray], masks: np.ndarray, v3dcams: v3d.Camera, min_ratio=1/3):
    """
    Segments a point cloud given a collection of masks and cameras.

    Args:
        pc (mx3 array): point cloud
        masks (nxhxw array): n binary masks (float or uint8 should work)
        v3dcams (n v3d.Camera): n cameras
        min_ratio (float): minimum ratio of masks that must contain a point to be segmented

    Returns:
        np.ndarray: The indices of the segmented points.
    """
    if not isinstance(pc, v3d.Point3d):
        pc = v3d.Point3d(p=pc)
    scores = np.zeros(pc.size)
    for mask, cam in zip(masks, v3dcams):
        segidxs = segment_pc_from_mask(pc, mask, cam)
        scores[segidxs] += 1
    scores_valid = scores > min_ratio * v3dcams.size
    return np.arange(pc.size)[scores_valid]
