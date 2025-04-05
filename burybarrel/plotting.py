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


def get_axes_traces(transform, scale=1.0, linewidth=1.0):
    """
    Generates plotly traces for the axes of 3D transformation(s). Colors for each axis
    are red, green, and blue for x, y, and z, respectively, like in Blender.
    """
    if isinstance(transform, np.ndarray):
        tshape = transform.shape
        # nice code lol xd
        if len(tshape) == 2 and tshape[0] == 4 and tshape[1] == 4:
            transform = v3d.Transform.from_matrix(transform)
        elif len(tshape) == 3 and tshape[1] == 4 and tshape[2] == 4:
            transform = v3d.Transform.from_matrix(transform)
        else:
            raise ValueError("bad")
    xcol = "#d91616"
    ycol = "#22eb17"
    zcol = "#1929e0"
    traces = []
    origin = transform @ np.array([[0.0, 0, 0]])
    x = transform @ np.array([[1.0, 0, 0]])
    y = transform @ np.array([[0.0, 1, 0]])
    z = transform @ np.array([[0.0, 0, 1]])
    for i, singleorgn in enumerate(origin):
        traces.extend([
            get_ray_trace(singleorgn, x[i] - singleorgn, length=scale, width=linewidth, color=xcol),
            get_ray_trace(singleorgn, y[i] - singleorgn, length=scale, width=linewidth, color=ycol),
            get_ray_trace(singleorgn, z[i] - singleorgn, length=scale, width=linewidth, color=zcol)
        ])
    return traces


def get_line3d_trace(
    points, markersize=0, markercolor=None, markersymbol=None, linecolor=None, linewidth=1,
):
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        marker=dict(
            size=markersize,
            color=markercolor,
            # colorscale='Viridis',
            symbol=markersymbol,
        ),
        line=dict(
            color=linecolor,
            width=linewidth,
        )
    )
