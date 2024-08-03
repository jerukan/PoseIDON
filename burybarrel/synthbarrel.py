"""Utilities related to synthetic barrel/cylinder generation."""

import numpy as np
import scipy.linalg

from burybarrel.utils import random_unitvec3


class Cylinder:
    """Everything cylinder."""

    def __init__(self, x1, x2, r):
        """
        Args:
            x1: endpoint 1
            x2: endpoint 2
            r : radius
        """
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.r = r
        self.axis = self.x2 - self.x1
        self.h = np.linalg.norm(self.axis)
        self.axis = self.axis / self.h
        # centroid
        self.c = (self.x1 + self.x2) / 2

    @classmethod
    def from_mat_params(cls, params):
        params = np.array(params)
        if len(params.shape) == 2:
            params = params[0]
        return cls(params[:3], params[3:6], params[6])

    @classmethod
    def from_axis(cls, axis, r, h, c=None):
        """
        By default initializes a cylinder centered at (0, 0, 0).
        """
        axis = np.array(axis)
        if c is None:
            c = np.zeros(3)
        c = np.array(c)
        axis = axis / np.linalg.norm(axis)
        # force vector to point up
        if axis[2] < 0:
            axis = -axis
        x1 = -axis * (h / 2)
        x2 = axis * (h / 2)
        return cls(x1 + c, x2 + c, r)

    def __repr__(self):
        return f"Cylinder(x1={self.x1}, x2={self.x2}, r={self.r})"

    def transform(self, T):
        T = np.array(T)
        x1hom = np.hstack([self.x1, 1])
        x2hom = np.hstack([self.x2, 1])
        return Cylinder((T @ x1hom)[:3], (T @ x2hom)[:3], self.r)

    def translate(self, t):
        t = np.array(t)
        return Cylinder(self.x1 + t, self.x2 + t, self.r)

    def get_random_pts_vol(self, npoints, sigma=0):
        return random_cylinder_vol(self.x1, self.x2, self.r, npoints, sigma=sigma)

    def get_random_pts_surf(self, npoints, sigma=0, includecap=True):
        return random_cylinder_surf(
            self.x1, self.x2, self.r, npoints, sigma=sigma, includecap=includecap
        )

    def get_volume_ratio_monte(self, npoints, planecoeffs=None):
        return monte_carlo_volume_ratio(
            npoints, self.x1, self.x2, self.r, planecoeffs=planecoeffs
        )

    def get_pts_surf(self, nt=100, nv=5):
        return get_cylinder_surf(self.x1, self.x2, self.r, nt=nt, nv=nv)


def random_cylinder_vol(x1, x2, r, npoints, sigma=0):
    """
    Generates uniformly distributed points within the volume of a defined cylinder
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    axis = x2 - x1
    h = np.linalg.norm(axis)
    axis = axis / h
    axnull = scipy.linalg.null_space(np.array([axis]))
    axnull1 = axnull[:, 0]
    axnull2 = axnull[:, 1]
    # sqrt radius to get uniform distribution
    rand_r = r * np.sqrt(np.random.random(npoints))
    rand_theta = np.random.random(npoints) * 2 * np.pi
    rand_h = np.random.random(npoints) * h

    cosval = (
        np.tile(axnull1, (npoints, 1))
        * rand_r[..., None]
        * np.cos(rand_theta)[..., None]
    )
    sinval = (
        np.tile(axnull2, (npoints, 1))
        * rand_r[..., None]
        * np.sin(rand_theta)[..., None]
    )
    xyzs = (
        cosval
        + sinval
        + np.tile(x1, (npoints, 1))
        + rand_h[..., None] * np.tile(axis, (npoints, 1))
    )
    if sigma > 0:
        xyzs += np.random.normal(0, sigma, xyzs.shape)
    return xyzs


def random_cylinder_surf(x1, x2, r, npoints, sigma=0, includecap=True):
    """
    Generates uniformly distributed points within the surface of a defined cylinder
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    axis = x2 - x1
    h = np.linalg.norm(axis)
    axis = axis / h
    axnull = scipy.linalg.null_space(np.array([axis]))
    axnull1 = axnull[:, 0]
    axnull2 = axnull[:, 1]
    # top and bottom caps
    cap_area = 2 * np.pi * r**2
    side_area = h * np.pi * r * 2
    # for a single cap
    if includecap:
        ncap = int((cap_area / (cap_area + side_area)) * npoints / 2)
    else:
        ncap = 0
    nside = npoints - 2 * ncap
    # sqrt radius to get uniform distribution
    rand_r = np.full(npoints, r, dtype=float)
    rand_r[: 2 * ncap] = r * np.sqrt(np.random.random(2 * ncap))
    rand_theta = np.random.random(npoints) * 2 * np.pi
    rand_h = np.random.random(npoints) * h
    rand_h[:ncap] = 0.0
    rand_h[ncap : 2 * ncap] = h

    cosval = (
        np.tile(axnull1, (npoints, 1))
        * rand_r[..., None]
        * np.cos(rand_theta)[..., None]
    )
    sinval = (
        np.tile(axnull2, (npoints, 1))
        * rand_r[..., None]
        * np.sin(rand_theta)[..., None]
    )
    xyzs = (
        cosval
        + sinval
        + np.tile(x1, (npoints, 1))
        + rand_h[..., None] * np.tile(axis, (npoints, 1))
    )
    if sigma > 0:
        xyzs += np.random.normal(0, sigma, xyzs.shape)
    return xyzs


def classify_points(points, a, b, c, d):
    """Function to classify points based on the plane"""
    results = np.array([a, b, c]) @ points.T + d
    above_plane = results >= 0
    return (above_plane).sum(), (~above_plane).sum()


def monte_carlo_volume_ratio(npoints, x1, x2, r, planecoeffs=None):
    """
    Monte Carlo method to estimate buried volume ratio (percent buried as decimal).

    Args:
        planecoeffs: [a, b, c, d]
    """
    if planecoeffs is None:
        a, b, c, d = 0.0, 0.0, 1.0, 0.0
    else:
        a, b, c, d = planecoeffs
    points = random_cylinder_vol(x1, x2, r, npoints)
    above_plane, below_plane = classify_points(points, a, b, c, d)
    ratio = below_plane / npoints
    return ratio


def generate_oriented_barrel(r, h, npoints, sigma=0, zlims=None, includecap=True):
    """Uniformly random oriented barrel, with random height from cylinder centroid z=0"""
    if zlims is None:
        zlims = [0, 0]
    heightchange = np.random.uniform(zlims[0], zlims[1])
    cylax = random_unitvec3(1)[0]
    cyl = Cylinder.from_axis(cylax, r, h, c=[0, 0, heightchange])

    points_all = cyl.get_random_pts_surf(npoints, sigma=sigma, includecap=includecap)

    # filter out buried points
    points = points_all[points_all[:, 2] >= 0, :]
    return points, points_all, cyl


def get_cyl_endpoints(cylax, h, offset, axidx=2):
    """
    Get axis endpoints of a cylinder based on centroid with given axis up at 0.

    axidx is 0 (x), 1 (y), 2 (z) up

    Offset z input.
    """
    cylax = cylax / np.linalg.norm(cylax)
    # force vector to point up
    if cylax[axidx] < 0:
        cylax = -cylax
    x1 = -cylax * (h / 2)
    x1[axidx] += offset
    x2 = cylax * (h / 2)
    x2[axidx] += offset
    return x1, x2


def get_cylinder_surf(x1, x2, r, nt=100, nv=5):
    """
    Returns x,y,z meshgrids for plotting a cylinder surface.

    Parametrize the cylinder of radius r, and endpoints x1, x2.

    Args:
        nt: number of angles
        nv: number of heights

    Returns:
        x, y, z meshgrids
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    axis = x2 - x1
    h = np.linalg.norm(axis)
    axis = axis / h
    axnull = scipy.linalg.null_space(np.array([axis]))
    axnull1 = axnull[:, 0]
    axnull2 = axnull[:, 1]

    theta = np.linspace(0, 2 * np.pi, nt)
    v = np.linspace(0, h, nv)
    theta, v = np.meshgrid(theta, v)
    gridshape = theta.shape
    gridsize = theta.size
    cosval = np.tile(axnull1, (gridsize, 1)) * r * np.cos(theta.reshape(-1))[..., None]
    sinval = np.tile(axnull2, (gridsize, 1)) * r * np.sin(theta.reshape(-1))[..., None]
    xyzs = (
        cosval
        + sinval
        + np.tile(x1, (gridsize, 1))
        + v.reshape(-1)[..., None] * np.tile(axis, (gridsize, 1))
    )
    x = xyzs[:, 0].reshape(gridshape)
    y = xyzs[:, 1].reshape(gridshape)
    z = xyzs[:, 2].reshape(gridshape)
    return x, y, z
