from pathlib import Path
from typing import List, Tuple

import cartopy
import cartopy.crs as ccrs
import cv2
import dataclass_array as dca
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.neighbors import KDTree
import torch
import trimesh
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


def get_plane_zup(pts, n=10, z=0, square_grid=False):
    pts = np.array(pts)
    xmin = np.min(pts[:, 0])
    xmax = np.max(pts[:, 0])
    ymin = np.min(pts[:, 1])
    ymax = np.max(pts[:, 1])
    xdiff = xmax - xmin
    ydiff = ymax - ymin
    nx, ny = n, n
    if square_grid:
        if xdiff > ydiff:
            gridsize = xdiff / n
            ny = int(ydiff // gridsize + 1)
            diff = (ny * gridsize) - ydiff
            ymin = ymin - diff / 2
            ymax = ymax + diff / 2
        else:
            gridsize = ydiff / n
            nx = int(xdiff // gridsize + 1)
            diff = (nx * gridsize) - xdiff
            xmin = xmin - diff / 2
            xmax = xmax + diff / 2
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    zz = np.zeros_like(xx)
    zz.fill(z)
    return xx, yy, zz


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


def get_trimesh_traces(mesh: trimesh.Trimesh, surfcolor: str=None, wirecolor: str=None) -> Tuple[go.Mesh3d, go.Scatter3d]:
    """
    Gets the plotly traces for mesh surface and mesh wireframe by ripping them from
    the figure made by figure factory.

    Args:
        surfcolor (str): single color for the mesh surface, either hex or rgb()/rgba() string
        wirecolor (str): single color for wireframe
    
    Returns:
        (mesh trace, wireframe trace)
    """
    if surfcolor is not None:
        surfcolor = [surfcolor] * len(mesh.faces)
    meshfig = ff.create_trisurf(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        simplices=mesh.faces, color_func=surfcolor, show_colorbar=False
    )
    traces = list(meshfig.select_traces())
    meshtrace: go.Mesh3d = traces[0]
    wireframetrace: go.Scatter3d = traces[1]
    if wirecolor is not None:
        wireframetrace = wireframetrace.update({"line": {"color": wirecolor}})
    return meshtrace, wireframetrace


def generate_domain(lats, lons, padding=0):
    """Will have funky behavior if the coordinate range loops around back to 0."""
    lat_rng = (np.min(lats), np.max(lats))
    lon_rng = (np.min(lons), np.max(lons))
    return dict(
        S=lat_rng[0] - padding,
        N=lat_rng[1] + padding,
        W=lon_rng[0] - padding,
        E=lon_rng[1] + padding,
    )


def get_carree_axis(domain=None, projection=None, land=False, fig=None, pos=None):
    """
    Args:
        fig: exiting figure to add to if desired
        pos: position on figure subplots to add axes to
    """
    if projection is None:
        projection = ccrs.PlateCarree()
    if fig is None:
        fig = plt.figure()
    if pos is None:
        pos = 111
    if isinstance(pos, int):
        ax = fig.add_subplot(pos, projection=projection)
    else:
        ax = fig.add_subplot(*pos, projection=projection)
    if domain is not None:
        ext = [domain["W"], domain["E"], domain["S"], domain["N"]]
        ax.set_extent(ext, crs=projection)
    if land:
        ax.add_feature(cartopy.feature.COASTLINE)
    return fig, ax


def get_carree_gl(ax, labels=True):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.top_labels, gl.right_labels = (False, False)
    if not labels:
        gl.bottom_labels, gl.left_labels = (False, False)
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    return gl
