from pathlib import Path

import numpy as np
import trimesh
import visu3d as v3d
import yaml

from burybarrel.camera import load_v3dcams, RadialCamera


class Reconstruction:
    def __init__(self, data_dir, res_dir):
        self.data_dir = Path(data_dir)
        self.res_dir = Path(res_dir)
    
    def get_dense(self, **kwargs):
        pass

    def get_cameras(self, **kwargs):
        pass


class ReconstructionVGGT(Reconstruction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstr_dir = self.res_dir / "vggt-out"

    def get_dense(self, idx=0):
        plydir = self.reconstr_dir / "pc_ply"
        plypaths = list(sorted(plydir.glob("*.ply")))
        trimeshpc: trimesh.PointCloud = trimesh.load(plypaths[idx])
        scenevtxs, scenecols = trimeshpc.vertices, trimeshpc.colors[:, :3]
        scenepts = v3d.Point3d(p=scenevtxs, rgb=scenecols)
        return scenepts
    
    def get_cameras(self, use_colmap_K=True):
        colmapcam_path = self.data_dir / "camera.json"
        camposes_path = self.reconstr_dir / "cam_poses.json"
        with open(camposes_path, "rt") as f:
            camposes = yaml.safe_load(f)
        if use_colmap_K:
            with open(colmapcam_path, "rt") as f:
                colmapcaminfo = yaml.safe_load(f)
            spec = RadialCamera.from_jsonargs(**colmapcaminfo)
        else:
            allK = np.array([campose["K"] for campose in camposes])
            spec = v3d.PinholeCamera(
                K=np.median(allK, axis=0),
                resolution=(camposes[0]["height"], camposes[0]["width"]),
            )
        cams, imgpaths = load_v3dcams(camposes_path, img_parent=self.data_dir / "rgb")
        cams = cams.replace(spec=spec)
        return cams, imgpaths
        

class ReconstructionFast3r(ReconstructionVGGT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstr_dir = self.res_dir / "fast3r-out"

class ReconstructionCOLMAP(Reconstruction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colmap_dir = self.res_dir / "colmap-out"
        self.openmvs_dir = self.res_dir / "openmvs-out"

    def get_dense(self):
        scene_path = self.openmvs_dir / "scene_dense_trimeshvalid.ply"
        trimeshpc: trimesh.PointCloud = trimesh.load(scene_path)
        scenevtxs, scenecols = trimeshpc.vertices, trimeshpc.colors[:, :3]
        scenepts = v3d.Point3d(p=scenevtxs, rgb=scenecols)
        return scenepts
    
    def get_cameras(self):
        camposes_path = self.colmap_dir / "cam_poses.json"
        cams, imgpaths = load_v3dcams(camposes_path, img_parent=self.data_dir / "rgb")
        return cams, imgpaths
