import numpy as np
from sklearn.neighbors import KDTree
import torch


def rotate_pts_to_ax(pts, normal, target, ret_R=False):
    normal = np.array(normal, dtype=float)
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
