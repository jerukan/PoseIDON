"""
Everything images.
"""
import json
import os
from pathlib import Path
from typing import List, Union, Tuple

import cv2
import dataclass_array as dca
import numpy as np
from PIL import Image
import pyrender
import quaternion
import torch
import trimesh
import visu3d as v3d
import yaml

from burybarrel.utils import ext_pattern


def imgs_from_dir(imgdir, sortnames=True, patterns=None, asarray=False, grayscale=False) -> Tuple[List[Path], Union[np.ndarray, List[Image.Image]]]:
    """
    Loads 3-channel RGB or 1-channel grayscale images.

    Made so I don't have to rewrite this in every notebook.

    Args:
        asarray (bool): if true, load image array as RGB arrays. Otherwise, load as PIL Images.

    Returns:
        (imgpaths, imgs): list of img paths and list of loaded images
    """
    imgdir = Path(imgdir)
    if not imgdir.exists():
        raise FileNotFoundError(f"Directory {imgdir} not found.")
    if patterns is None:
        patterns = [ext_pattern("png"), ext_pattern("jpg"), ext_pattern("jpeg")]
    imgpaths = []
    for pattern in patterns:
        imgpaths.extend(list(imgdir.glob(pattern)))
    if sortnames:
        imgpaths = sorted(imgpaths)
    if asarray:
        if grayscale:
            imgs = np.array([cv2.imread(str(imgpath), cv2.IMREAD_GRAYSCALE) for imgpath in imgpaths])
        else:
            imgs = np.array([cv2.cvtColor(cv2.imread(str(imgpath)), cv2.COLOR_BGR2RGB) for imgpath in imgpaths])
    else:
        imgs = [Image.open(imgpath) for imgpath in imgpaths]
        if grayscale:
            imgs = [img.convert("L") for img in imgs]
        else:
            imgs = [img.convert("RGB") for img in imgs]
    return imgpaths, imgs


def get_bbox_mask(bbox, W, H):
    """
    Sets values inside bounding box to 255.

    Args:
        bbox: [x_min, y_min, x_max, y_max]
    """
    bbox = np.array(bbox, dtype=int)
    boxmask = np.zeros((H, W), dtype=np.uint8)
    boxmask = cv2.rectangle(boxmask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
    return boxmask


def get_local_plane_mask(bbox, expandratio_in, expandratio_out, W, H):
    """
    Takes the difference between a larger bbox and an even larger bbox mask to get
    a 'frame' of the local plane around the barrel.

    Args:
        bbox: [x_min, y_min, x_max, y_max]
        expandratio_in: expansion ratio of bbox sides for inner bbox
        expandratio_out: expansion ratio of bbox sides for outer bbox
    """
    newbboxout = np.zeros(4, dtype=int)
    newbboxin = np.zeros(4, dtype=int)
    expandratioin = expandratio_in
    expandratioout = expandratio_out
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    cx = bbox[0] + width // 2
    cy = bbox[1] + height // 2
    newbboxin[0] = max(cx - (expandratioin * width) // 2, 0)
    newbboxin[1] = max(cy - (expandratioin * height) // 2, 0)
    newbboxin[2] = min(cx + (expandratioin * width) // 2, W)
    newbboxin[3] = min(cy + (expandratioin * height) // 2, H)
    newbboxout[0] = max(cx - (expandratioout * width) // 2, 0)
    newbboxout[1] = max(cy - (expandratioout * height) // 2, 0)
    newbboxout[2] = min(cx + (expandratioout * width) // 2, W)
    newbboxout[3] = min(cy + (expandratioout * height) // 2, H)
    return get_bbox_mask(newbboxout, W, H) - get_bbox_mask(newbboxin, W, H)


def apply_clahe(img, clipLimit=None, tileGridSize=None):
    """
    Applies CLAHE to the image in CIELAB colorspace, as specified in the following link

    https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def render_v3d(cam: v3d.Camera, points: v3d.Point3d, radius=1, background=None) -> np.ndarray:
    """
    A modified version of v3d.Camera.render() to allow the sizes of each of the
    points in a point cloud to be changed, since you literally couldn't see
    anything for sparse point clouds renders (each points is literally a pixel).
    Currently doesn't scale point size for distance. I may implement this later.

    Project 3d points to the camera screen.

    Args:
      points: 3d points.
      background: background image to use; must be same dimensions as camera width/height

    Returns:
      img: The projected 3d points.
    """
    # TODO(epot): Support float colors and make this differentiable!
    if not isinstance(points, v3d.Point3d):
        raise TypeError(
            f'Camera.render expect `v3d.Point3d` as input. Got: {points}.'
        )

    # Project 3d -> 2d coordinates
    points2d = cam.px_from_world @ points

    # Flatten pixels
    points2d = points2d.flatten()
    px_coords = points2d.p
    rgb = points2d.rgb

    # Compute the valid coordinates
    w_coords = px_coords[..., 0]
    h_coords = px_coords[..., 1]
    valid_coords_mask = (
        (0 <= h_coords)
        & (h_coords < cam.h - 1)
        & (0 <= w_coords)
        & (w_coords < cam.w - 1)
        & (points2d.depth[..., 0] > 0)  # Filter points behind the camera
    )
    rgb = rgb[valid_coords_mask]
    px_coords = px_coords[valid_coords_mask]
    px_coords = np.astype(np.round(px_coords), np.int32)

    # px_coords is (h, w)
    img = np.zeros((*cam.resolution, 3), dtype=np.uint8)
    if background is not None:
        if background.shape[0] == cam.resolution[0] and background.shape[1] == cam.resolution[1]:
            img[...] = background
        else:
            raise ValueError(f"Invalid background img shape {background.shape}")
    for i, coord in enumerate(px_coords):
        img = cv2.circle(img, coord, radius, tuple(int(ch) for ch in rgb[i]), -1)
    return img


def render_models(cam: v3d.Camera, meshes: Union[trimesh.Trimesh, List[trimesh.Trimesh]], transforms: v3d.Transform, light_intensity=2.4, flags=pyrender.RenderFlags.NONE, device=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Renders meshes in given poses for a given camera view.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): uint8 color (hxwh3), depth (hxw), mask (hxw)
    """
    if torch.cuda.is_available():
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        if device is not None:
            if ":" in device:
                devicenum = device.split(":")[1]
                os.environ["EGL_DEVICE_ID"] = devicenum
    renderer = pyrender.OffscreenRenderer(cam.w, cam.h)
    pyrendercam = pyrender.IntrinsicsCamera(
        fx=cam.spec.K[0, 0],
        fy=cam.spec.K[1, 1],
        cx=cam.spec.K[0, 2],
        cy=cam.spec.K[1, 2],
        znear=0.05,
        zfar=3000.0  
    )
    if isinstance(meshes, trimesh.Geometry):
        meshes = [meshes]
    if isinstance(transforms, list):
        transforms = dca.stack(transforms)
    transforms = transforms.reshape((-1,))
    pyrendermeshes = [pyrender.Mesh.from_trimesh(mesh) for mesh in meshes]
    camrot = v3d.Transform.from_angle(x=np.pi, y=0, z=0)
    oglcamT: v3d.Transform = cam.world_from_cam @ camrot

    ambient_light = np.array([0.02, 0.02, 0.02, 1.0])
    scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=ambient_light)
    meshnodes = []
    for pyrendermesh, transform in zip(pyrendermeshes, transforms):
        meshnode = pyrender.Node(mesh=pyrendermesh, matrix=transform.matrix4x4)
        scene.add_node(meshnode)
        meshnodes.append(meshnode)
    camnode = pyrender.Node(camera=pyrendercam, matrix=oglcamT.matrix4x4)
    # light = pyrender.SpotLight(
    #     color=np.ones(3),
    #     intensity=light_intensity,
    #     innerConeAngle=np.pi / 16.0,
    #     outerConeAngle=np.pi / 6.0,
    # )
    light = pyrender.PointLight(color=np.ones(3), intensity=light_intensity)
    lightnode = pyrender.Node(light=light, matrix=oglcamT.matrix4x4)
    scene.add_node(camnode)
    scene.add_node(lightnode)
    color, depth = renderer.render(scene, flags=flags)
    mask = depth > 0
    renderer.delete()
    return color, depth, mask


def to_contour(img: np.ndarray, color=(255, 255, 255), dilate_iterations=1, outline_only=False, background=None):
    """
    Get contour of an object rendering (only object in frame, black background).

    Args:
        background (np.ndarray): RGB background image to overlay contour onto
    """
    if outline_only:
        mask = np.zeros_like(img)
        mask[img > 0] = 255
        mask = np.max(mask, axis=-1)
        mask_bool = mask.numpy().astype(np.bool_)

        mask_uint8 = (mask_bool.astype(np.uint8) * 255)[:, :, None]
        mask_rgb = np.concatenate((mask_uint8, mask_uint8, mask_uint8), axis=-1)
    else:
        mask_rgb = img

    canny = cv2.Canny(mask_rgb, threshold1=30, threshold2=100)

    kernel = np.ones((3, 3), np.uint8)
    canny = cv2.dilate(canny, kernel, iterations=dilate_iterations)

    if background is not None:
        img_contour = np.copy(background)
    else:
        img_contour = np.zeros_like(img)
    img_contour[canny > 0] = color

    return img_contour
