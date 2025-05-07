"""
Everything transforms, rotations, and fun!
"""
from typing import List, Dict

import numpy as np
from numpy.typing import NDArray
import quaternion
from sklearn.neighbors import KDTree
import torch
import transforms3d as t3d
import visu3d as v3d

from burybarrel import get_logger


logger = get_logger(__name__)


def apply_T_xyz(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, T: v3d.Transform):
    """
    Applies a transform to x, y, z coordinates with arbitrary shape.
    """
    original_shape = xx.shape
    pts = np.array([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T
    pts = T @ pts
    xx = pts[:, 0].reshape(original_shape)
    yy = pts[:, 1].reshape(original_shape)
    zz = pts[:, 2].reshape(original_shape)
    return xx, yy, zz


def T_from_blender(Reuler, t, scale=1) -> v3d.Transform:
    """
    Converts Blender translation and euler angle rotation to a 4x4 v3d Transform.

    Args:
        Reuler: a euler angle rotation [x, y, z] in degrees (sxyz order)
        t: [x, y, z] translation
        scale: multiply translation by 1 / scale
    """
    R = t3d.euler.euler2mat(*(np.array(Reuler).reshape(3) * np.pi / 180))
    t = np.array(t) * (1 / scale)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    T = v3d.Transform.from_matrix(T)
    return T


def scale_T_translation(T: v3d.Transform, scale) -> v3d.Transform:
    """Specifically scale translation component of SE(3) transform"""
    return T.replace(t=T.t * scale)


def T_from_translation(*translation) -> v3d.Transform:
    """
    Converts a translation vector to a 4x4 v3d Transform.

    Args:
        translation: a single translation [x, y, z]
    """
    translation = np.array(translation).reshape(3)
    T = np.eye(4)
    T[:3, 3] = translation
    return v3d.Transform.from_matrix(T)


def scale_pc(pc: v3d.Point3d, scale) -> v3d.Point3d:
    return pc.replace(p=pc.p * scale)


def scale_cams(cams: v3d.Camera, scale) -> v3d.Camera:
    return cams.replace(world_from_cam=scale_T_translation(cams.world_from_cam, scale))


def qangle(q1: quaternion.quaternion, q2: quaternion.quaternion) -> float:
    """
    Angle in radians between 2 quaternions.
    https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames
    """
    qerr = q1 * q2.conjugate()
    if qerr.w < 0:
        qerr *= -1
    err = np.arctan2(np.sqrt(qerr.x ** 2 + qerr.y ** 2 + qerr.z ** 2), qerr.w)
    return err


def qmean(qs: NDArray[quaternion.quaternion], weights: List[float]=None) -> quaternion.quaternion:
    """https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions"""
    if weights is None:
        weights = np.ones(len(qs))
    qs = np.squeeze(qs)
    Q = quaternion.as_float_array(qs * weights).T
    # symmetric real matrix, and PSD
    QQ = Q @ Q.T
    # eigh should prevent complex eigenvectors from being selected, I think
    vals, vecs = np.linalg.eigh(QQ)
    avg = vecs[:, np.argmax(np.abs(vals))]
    avg = avg / np.linalg.norm(avg)
    return quaternion.from_float_array(avg)


def closest_quat_sym(q1: quaternion.quaternion, q2: quaternion.quaternion, syms: List[Dict]=None) -> quaternion.quaternion:
    """
    Use q1 as reference, brute force rotate q2 using symmetry info and return the closest
    rotation to q1.

    Args:
        q1 (quaternion): Reference quaternion.
        q2 (quaternion): Quaternion to rotate.
        syms (dict): List of symmetry transformations, each given by a dictionary with:
            - 'R': 3x3 ndarray with the rotation matrix.
            - 't': 3x1 ndarray with the translation vector.
            The BOP toolkit has a function to generate this info.

    Returns:
        quaternion: q2 rotated to a symmetric rotation closest to q1
    """
    if syms is None:
        return q2
    errs = []
    q2_syms = []
    for sym in syms:
        q2_sym = q2 * quaternion.from_rotation_matrix(sym["R"])
        errs.append(qangle(q1, q2_sym))
        q2_syms.append(q2_sym)
    return q2_syms[np.argmin(np.abs(errs))]


def get_axes_rot(axis, target):
    """
    Get rotation matrix that rotates from given axis to target axis.
    """
    normal = np.array(axis, dtype=float)
    target = np.array(target, dtype=float)
    ang = np.arccos(
        (target @ normal) / (np.linalg.norm(target) * np.linalg.norm(normal))
    )
    rotax = np.cross(normal, target)
    eta = rotax / np.linalg.norm(rotax)
    theta = eta * ang
    thetahat = np.array(
        [[0, -theta[2], theta[1]], [theta[2], 0, -theta[0]], [-theta[1], theta[0], 0]]
    )
    R = (
        np.eye(3)
        + (np.sin(ang) / ang) * thetahat
        + ((1 - np.cos(ang)) / ang**2) * (thetahat @ thetahat)
    )
    return R


def rotate_pts_to_ax(pts, normal, target, ret_R=False):
    R = get_axes_rot(normal, target)
    rotscenexyz = (R @ pts.T).T
    if ret_R:
        return rotscenexyz, R
    return rotscenexyz


def rotate_pts_to_ax_torch(pts, normal, target):
    """Given a point cloud with normal vector, rotate it such that the new normal matches the target vector
    Args:
                pts: (torch.tensor) [N, 3] point cloud
                normal (torch.tensor)[3,] normal vector
                target (torch.tensor) [3,] target normal vector
    Return:
                rotated_pts (torch.tensor) [N, 3] rotated point cloud
    """
    ang = torch.arccos(
        (target @ normal) / (torch.linalg.norm(target) * torch.linalg.norm(normal))
    )
    rotax = torch.cross(normal, target)
    eta = rotax / torch.linalg.norm(rotax)
    theta = eta * ang
    thetahat = torch.tensor(
        [[0, -theta[2], theta[1]], [theta[2], 0, -theta[0]], [-theta[1], theta[0], 0]]
    )
    R = (
        torch.eye(3)
        + (torch.sin(ang) / ang) * thetahat
        + ((1 - torch.cos(ang)) / ang**2) * (thetahat @ thetahat)
    )
    rotscenexyz = pts @ R.T
    return rotscenexyz


def icp(src_pc: np.ndarray, trg_pc: np.ndarray, T_init=None, max_iters=20, tol=1e-3, verbose=False, ret_err=False, outlier_std=0.0):
    """
    Standard iterative closest point.

    Assumes src_pc is a subset of trg_pc. Otherwise, this may have questionable performance.

    Args:
        src_pc (nx3 array): source point cloud; transform T will be applied to this; n<=m should
            hold true
        trg_pc (mx3 array): target point cloud
        T_init (4x4 transform): initial guess for transform
        outlier_std (float): standard deviation nearest neighbor distance for outlier rejection
            of source point cloud; 0.0 means no rejection

    Returns:
        (4x4 transform, Optional[distance mse]): T @ src -> trg
    """
    src_mean = np.mean(src_pc, axis=0)
    trg_mean = np.mean(trg_pc, axis=0)
    src_cent = src_pc - src_mean
    # trg_cent = trg_pc - trg_mean

    # src_kd = KDTree(src_pc)
    target_kd = KDTree(trg_pc)

    if T_init is not None:
        T0 = np.array(T_init)
    else:
        T0 = np.eye(4)
    prevT = T0
    T = T0
    for i in range(max_iters):
        R = T[:3, :3]
        p = T[:3, 3]
        distances, close_idxs = target_kd.query((R @ src_pc.T).T + p)
        distances = np.reshape(distances, -1)
        close_idxs = np.reshape(close_idxs, -1)
        if outlier_std > 0.0:
            # reject based on distance stddev
            inliermask = np.abs(distances - np.mean(distances)) < outlier_std * np.std(distances)
            close_idxs = close_idxs[inliermask]
            # recompute source mean and centered source pc
            src_mean = np.mean(src_pc[inliermask], axis=0)
            src_cent = src_pc[inliermask] - src_mean
        # src_mean_filt = np.mean(src_pc[close_idxs], axis=0)
        trg_mean_filt = np.mean(trg_pc[close_idxs], axis=0)
        z = src_cent
        m = trg_pc[close_idxs] - trg_mean_filt
        Q = m.T @ z
        U, S, V = np.linalg.svd(Q)  # V is returned already transposed
        R = U @ np.diag([1, 1, np.linalg.det(U @ V)]) @ V
        # p = np.mean(target_pc[close_idxs], axis=0) - R @ src_mean
        p = trg_mean_filt - R @ src_mean
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        if np.allclose(prevT, T, atol=tol):
            break
        prevT = T
        if i == max_iters - 1:
            if verbose:
                print(f"max iters {max_iters} reached before tolerance {tol}")
    if ret_err:
        # mean square error
        dists, _ = target_kd.query((R @ src_cent.T).T + p)
        err = np.mean(dists ** 2)
        return T, err
    return T


def icp_translate(
    source_pc, target_pc, max_iters=20, tol=1e-3, verbose=False, ntheta=3, nphi=3
):
    """
    Extremely jank implementation of iterative closest point for only translation.

    Initializes guesses of translation by sampling points on a sphere around the
    target point cloud.

    source_pc assumed to be smaller than target_pc

    Returns:
        translation: 3d numpy array
    """
    src_mean = np.mean(source_pc, axis=0)
    targ_mean = np.mean(target_pc, axis=0)
    scale = np.max(np.linalg.norm(target_pc - targ_mean, axis=1))
    target_kd = KDTree(target_pc)

    if ntheta > 0 and nphi > 0:
        thetas = np.linspace(0, 2 * np.pi, ntheta + 1)[:-1]
        phis = np.linspace(0, np.pi, nphi + 2)[1:-1]
        alltheta, allphi = np.meshgrid(thetas, phis)
        alltheta = alltheta.reshape(-1)
        allphi = allphi.reshape(-1)
        offset_choices = (
            scale
            * np.array(
                [
                    np.sin(allphi) * np.cos(alltheta),
                    np.sin(allphi) * np.sin(alltheta),
                    np.cos(allphi),
                ]
            ).T
        )
    else:
        offset_choices = np.array([None])
    alltranslations = np.zeros((len(offset_choices), 3))
    allmeandists = np.zeros(len(offset_choices))
    for j, offset in enumerate(offset_choices):
        # p = targ_mean - src_mean
        if offset is None:
            p = np.array([0.0, 0.0, 0.0])
        else:
            p = (targ_mean + offset) - src_mean
        prevp = p
        prevdist = np.inf
        K = max_iters
        for i in range(K):
            dists, close_idxs = target_kd.query(source_pc + p)
            meandist = np.mean(dists)
            targ_mean_filt = np.mean(target_pc[close_idxs], axis=0)
            p = targ_mean_filt - src_mean
            if np.abs(prevdist - meandist) < tol:
                if verbose:
                    print(f"converged at iter {i}")
                break
            prevp = p
            prevdist = meandist
            if i == K - 1:
                if verbose:
                    print(f"max iters {K} reached before tolerance {tol}")
        allmeandists[j] = np.mean(meandist)
        alltranslations[j, :] = p
    bestidx = np.argmin(allmeandists)
    return alltranslations[bestidx]
