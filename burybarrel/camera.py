import json
from pathlib import Path
from typing import List, Union, Tuple

import cv2
import dataclass_array as dca
from etils import enp
from etils.array_types import FloatArray
import numpy as np
from PIL import Image
import quaternion
import visu3d as v3d
from visu3d.utils import np_utils
import yaml

from burybarrel.utils import ext_pattern
from burybarrel.transform import scale_T_translation


class RadialCamera(v3d.PinholeCamera):
    """Simple radial camera model as defined in COLMAP.

    f, cx, cy, k1, k2

    Attributes:
        K: Camera intrinsics parameters.
        resolution: (h, w) resolution
        k1k2: (k1, k2) distortion coefficients
    """

    k1k2: FloatArray["*shape 2"] = (0.0, 0.0)

    @classmethod
    def from_jsonargs(cls, fx, fy, cx, cy, k1, k2, height, width):
        return cls(
            K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            resolution=(height, width),
            k1k2=(k1, k2),
        )

    def _px_and_depth_from_cam(
        self,
        points3d,
    ):
        """TODO: implement with distortion."""
        if points3d.shape[-1] != 3:
            raise ValueError(f"Expected cam coords {points3d.shape} to be (..., 3)")

        # K @ [X,Y,Z] -> s * [u, v, 1]
        # (3, 3) @ (..., 3) -> (..., 3)
        points2d = self.xnp.einsum("ij,...j->...i", self.K, points3d)
        # Normalize: s * [u, v, 1] -> [u, v, 1]
        # And only keep [u, v]
        depth = points2d[..., 2:3]
        points2d = points2d[..., :2] / (depth + 1e-8)
        return points2d, depth

    def _cam_from_px(
        self,
        points2d: FloatArray["*d 2"],
    ) -> FloatArray["*d 3"]:
        """TODO: implement with distortion."""
        assert not self.shape  # Should be vectorized
        points2d = dca.utils.np_utils.asarray(points2d, xnp=self.xnp)
        if points2d.shape[-1] != 2:
            raise ValueError(f"Expected pixel coords {points2d.shape} to be (..., 2)")

        # [u, v] -> [u, v, 1]
        # Concatenate (..., 2) with (..., 1) -> (..., 3)
        points2d = np_utils.append_row(points2d, 1.0, axis=-1)

        # [X,Y,Z] / s = K-1 @ [u, v, 1]
        # (3, 3) @ (..., 3) -> (..., 3)
        k_inv = enp.compat.inv(self.K)
        points3d = self.xnp.einsum("ij,...j->...i", k_inv, points2d)

        # TODO(epot): Option to return normalized rays ?
        # Set z to -1
        # [X,Y,Z] -> [X, Y, Z=1]
        points3d = points3d / enp.compat.expand_dims(points3d[..., 2], axis=-1)
        return points3d


def scale_cams(scale: float, cams: v3d.Camera):
    T = cams.world_from_cam
    return cams.replace(world_from_cam=scale_T_translation(T, scale))


def save_v3dcams(cams: v3d.Camera, imgpaths: List[Union[str, Path]], outpath: Union[str, Path], format="json"):
    """
    {
        "img_path": str,
        "R": [w, x, y, z],
        "t": [x, y, z],
        "K": 3x3 float array,
        "k1k2": [k1, k2],
        "width": int,
        "height": int,
    }
    """
    camposedata = []
    for cam, imgpath in zip(cams, imgpaths):
        quat = quaternion.from_rotation_matrix(cam.world_from_cam.R)
        t = cam.world_from_cam.t
        if hasattr(cam.spec, "k1k2"):
            k1k2 = cam.spec.k1k2.tolist()
        else:
            k1k2 = [0.0, 0.0]
        singlepose = {
            "img_path": str(imgpath.absolute()),
            "R": quaternion.as_float_array(quat).tolist(),
            "t": t.tolist(),
            "K": cam.spec.K.tolist(),
            "k1k2": k1k2,
            "width": cam.spec.w,
            "height": cam.spec.h,
        }
        camposedata.append(singlepose)
    outpath = Path(outpath)
    with open(outpath, "wt") as f:
        if format == "json":
            json.dump(camposedata, f, indent=4)
        elif format == "yaml":
            yaml.dump(camposedata, f)
    return camposedata


def load_v3dcams(path, img_parent=None) -> Tuple[v3d.Camera, List[Path]]:
    """
    JSON fields:
    ```
    {
        "img_path": str,
        "R": [w, x, y, z],
        "t": [x, y, z],
        "K": 3x3 float array,
        "k1k2": [k1, k2],
        "width": int,
        "height": int,
    }
    ```

    Args:
        path (path-like)
        img_parent (path-like): parent directory of images if on a different machine (simply
            replaces the parent directory of the image names in the JSON)

    Returns:
        (v3d.Camera, List[Path]): cameras and image paths
    """
    with open(path, "rt") as f:
        # yaml can load jsons
        camposedata = yaml.safe_load(f)
    camposedata = sorted(camposedata, key=lambda x: x["img_path"])
    camlisttmp: List[v3d.Camera] = []
    imgpaths: List[Path] = []
    for i, posedata in enumerate(camposedata):
        spec = RadialCamera(K=posedata["K"], resolution=(posedata["height"], posedata["width"]), k1k2=posedata["k1k2"])
        quat = quaternion.from_float_array(posedata["R"])
        R = quaternion.as_rotation_matrix(quat)
        t = np.array(posedata["t"])
        T = v3d.Transform(R=R, t=t)
        camlisttmp.append(v3d.Camera(spec=spec, world_from_cam=T))
        if img_parent is not None:
            imgpaths.append(Path(img_parent) / Path(posedata["img_path"]).name)
        else:
            imgpaths.append(Path(posedata["img_path"]))
    cams: v3d.Camera = dca.stack(camlisttmp)
    return cams, imgpaths
