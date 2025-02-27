import os
from pathlib import Path
import shutil
import struct
from typing import Union, List, Tuple

import dataclass_array as dca
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import sqlite3
import visu3d as v3d


def get_images(database_path: Union[str, Path]) -> List[pycolmap.Image]:
    db = pycolmap.Database(database_path)
    images = db.read_all_images()
    return images


def get_features(
    database_path: Union[str, Path]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    images = get_images(database_path)

    all_keypoints = []
    all_descriptors = []

    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()

        for img in images:
            image_id = img.image_id
            image_name = img.name

            cursor.execute("SELECT data FROM keypoints WHERE image_id=?;", (image_id,))
            row = next(cursor)
            if row[0] is None:
                keypoints = np.zeros((0, 6), dtype=np.float32)
                descriptors = np.zeros((0, 128), dtype=np.uint8)
            else:
                keypoints = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 6)
                cursor.execute(
                    "SELECT data FROM descriptors WHERE image_id=?;", (image_id,)
                )
                row = next(cursor)
                descriptors = np.frombuffer(row[0], dtype=np.uint8).reshape(-1, 128)
            all_keypoints.append(keypoints)
            all_descriptors.append(descriptors)
        return all_keypoints, all_descriptors


def get_pc(reconstruction: pycolmap.Reconstruction) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        points in 3d space, RGB color for each point
    """
    pts = []
    cols = []
    for pair in reconstruction.points3D.items():
        pts.append(pair[1].xyz)
        cols.append(pair[1].color)
    pts = np.array(pts)
    cols = np.array(cols)
    return pts, cols


def get_cams_v3d(reconstruction: pycolmap.Reconstruction, sortkey="name", return_names=False) -> v3d.Camera:
    """
    Retrieves camera parameters from every image in a reconstruction as a visu3d Camera
    dataclass array. COLMAP has these cameras out of order, so it's probably best to
    sort by the filename of the image.
    """
    imgs: List[pycolmap.Image] = list(reconstruction.images.values())
    if sortkey is None:
        pass
    elif sortkey == "name":
        imgs = sorted(imgs, key=lambda x: x.name)
    else:
        raise ValueError(f"sortkey='{sortkey}' is invalid or hasn't been implemented yet")
    camlisttmp: List[v3d.Camera] = []
    names: List[str] = []
    for img in imgs:
        spec = v3d.PinholeCamera.from_focal(resolution=(img.camera.height, img.camera.width), focal_in_px=img.camera.focal_length)
        T = v3d.Transform.from_matrix(img.cam_from_world.matrix()).inv
        camlisttmp.append(v3d.Camera(spec=spec, world_from_cam=T))
        names.append(img.name)
    cams: v3d.Camera = dca.stack(camlisttmp)
    if return_names:
        return cams, names
    return cams


"""
Below read_array and write_array were ripped from
https://github.com/colmap/colmap/blob/main/scripts/python/read_write_dense.py
"""


def read_array(path):
    """Read a colmap depth map."""
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_array(array, path):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(endian_character + format_char_sequence, *data_list)
        fid.write(byte_data)
