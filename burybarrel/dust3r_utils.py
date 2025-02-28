import os
from pathlib import Path
import sys
from typing import List

import dataclass_array as dca
import numpy as np
from PIL import Image
import torch
import visu3d as v3d


def _resize_pil_image(img, long_edge_size):
    """Ripped from dust3r"""
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    elif S <= long_edge_size:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_to_dust3r(img, size, square_ok=False, verbose=True):
    """Ripped from dust3r"""
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
        print(f"Resized image to {W2}x{H2}")
    return img


def save_dust3r_outs(scene, savepath):
    """
    Code to save output of dust3r after global alignment into a dictionary

    Args:
        scene: Output of `global_aligner`

    Returns:
        dict
        saves a .pth file, can be loaded using torch.load()
    """
    imgs = scene.imgs
    focals = scene.get_focals()  # Optimized Focal length of the N cameras [N,1]
    poses = scene.get_im_poses()  # Optimized Camera Poses [N,4,4]
    pts3d = scene.get_pts3d()  # Point clouds as seen from each camera.
    out_dict = {}
    pts3d = [pts.cpu().detach() for pts in pts3d]
    out_dict["imgs"] = imgs  # list of np arrays, not tensors
    out_dict["focals"] = focals.cpu().detach()
    out_dict["poses"] = poses.cpu().detach()
    out_dict["pts3d"] = pts3d
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    torch.save(out_dict, savepath)
    print(f"Saved Dust3r outputs to {savepath}")
    return out_dict


def read_dust3r(path, flatten=True):
    """
    Reads dust3r pickled dictionary.

    Returns:
        pts_final (v3d.Point3d): [N*P, 3]
        pts_each (List[v3d.Point3d]): List, [P, 3]
        v3dcams: List of visu3d.Camera
    """
    outs = torch.load(path)
    H, W, _ = outs["imgs"][0].shape
    # points separated by camera
    pts_each = []
    cols_each = []
    # concatenated points
    pts_final = []
    cols_final = []
    v3dcams: List[v3d.Camera] = []
    for i in range(len(outs["poses"])):
        imguint8 = (outs["imgs"][i] * 255).astype(np.uint8)
        f = outs["focals"][i, 0].numpy()
        cam_spec = v3d.PinholeCamera.from_focal(
            resolution=(H, W),
            focal_in_px=f,
        )
        pose = outs["poses"][i]
        T = pose.numpy()
        R, t = T[:3, :3], T[:3, 3:].T
        pts = outs["pts3d"][i]
        # pts = outs["pts3d"][i]@R.T + t
        if flatten:
            pts_each.append(pts.numpy().reshape(-1, 3))
            cols_each.append(imguint8.reshape(-1, 3))
        else:
            pts_each.append(pts.numpy())
            cols_each.append(imguint8)
        pts_final.append(pts)
        cols_final.append(imguint8)
        v3dcam = v3d.Camera(spec=cam_spec, world_from_cam=v3d.Transform.from_matrix(T))
        v3dcams.append(v3dcam)
    v3dcams: v3d.Camera = dca.stack(v3dcams)
    pts_final = torch.stack(pts_final).view(-1, 3).numpy()
    cols_final = np.stack(cols_final).reshape(-1, 3)
    pcs_each = [v3d.Point3d(p=pts, rgb=cols) for pts, cols in zip(pts_each, cols_each)]
    pc_final = v3d.Point3d(p=pts_final, rgb=cols_final)
    return pc_final, pcs_each, v3dcams
