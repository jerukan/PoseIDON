from pathlib import Path
from typing import List

import cv2
import dataclass_array as dca
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import KDTree
import torch
import visu3d as v3d


def get_surface_line_traces(
    x,
    y,
    z,
    color="#101010",
    width=1,
    step=1,
    include_vertical=True,
    include_horizontal=True,
) -> List[go.Scatter3d]:
    """
    Generates plotly traces for grid lines on a 3D surface, akin to to what 3D surfaces look
    like when plotted in MATLAB.
    """
    line_marker = dict(color=color, width=width)
    traces = []
    if include_horizontal:
        for xl, yl, zl in list(zip(x, y, z))[::step]:
            traces.append(
                go.Scatter3d(x=xl, y=yl, z=zl, mode="lines", line=line_marker, name="")
            )
    if include_vertical:
        for xl, yl, zl in list(zip(x.T, y.T, z.T))[::step]:
            traces.append(
                go.Scatter3d(x=xl, y=yl, z=zl, mode="lines", line=line_marker, name="")
            )
    return traces


def get_ray_trace(
    pos,
    raydir,
    length=1,
    width=1,
    color="#101010",
    markersize=6,
    markersymbol="diamond",
) -> go.Scatter3d:
    """Generates a plotly trace for a 3D ray given position and direction."""
    line_marker = dict(color=color, width=width)
    pos = np.array(pos)
    raydir = np.array(raydir)
    raydir = raydir / np.linalg.norm(raydir)
    pts = np.array([pos, pos + raydir * length])
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="lines+markers",
        line=line_marker,
        marker=go.scatter3d.Marker(
            size=[0, markersize, 0], symbol=markersymbol, opacity=1
        ),
        name="",
    )
