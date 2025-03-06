from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
import trimesh


def load_mesh(path) -> trimesh.Geometry:
    """
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
