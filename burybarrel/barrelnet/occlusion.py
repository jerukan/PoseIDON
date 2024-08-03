### Creating partially occluded point cloud.
import mitsuba as mi
import numpy as np
import roma
import torch
import drjit as dr
from matplotlib import pyplot as plt
import pyrender
import trimesh
import os
import gc

os.environ["PYOPENGL_PLATFORM"] = "egl"
mi.set_variant("cuda_ad_rgb")


def rotvec_to_matrix(target: torch.Tensor, device="cuda"):
    """
    Creates a rotation matrix to rotate a point cloud oriented y-axis up to align with
    a given axis using PyTorch.

    Args:
        axis (torch.Tensor): Target axis to rotate to, should be of shape (3,).
        device (str): Device to perform the computation ('cpu' or 'cuda').
    Returns:
        torch.Tensor (3, 3): Rotation matrix to transform points from y-axis up to target axis.
    """
    ## TODO: make this batched, to rotate the point cloud to different configurations at once.
    # Normalize the axis
    target = target.to(device)
    target = target / torch.norm(target)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
    rotation_vector = torch.cross(y_axis, target)
    rotation_vector = rotation_vector / torch.norm(rotation_vector)
    # angle = torch.arccos((y_axis @ target) / (torch.norm(y_axis) * torch.norm(target)))
    angle = torch.arccos(torch.dot(y_axis, target))

    # Convert rotation vector and angle to a PyTorch tensor
    theta = rotation_vector * angle

    # Create the rotation matrix using torch-roma
    rotation_matrix = roma.rotvec_to_rotmat(theta)
    return rotation_matrix


def get_posed_split_cylinder_mesh(radius, height, axis, burial_percent=0.5):
    """Args
    radius (float)
    height (float)
    axis (np.array (3,))
    burial_percent : what percentage of the visible cylinder from origin should be buried / visible
    """
    # Generate Cylinder Mesh and align it's axis with Y axis
    capped_cylinder = trimesh.creation.cylinder(
        radius=radius, height=height, sections=64, cap=True
    )
    capped_cylinder.apply_transform(
        trimesh.transformations.rotation_matrix(angle=np.pi / 2, direction=[1, 0, 0])
    )

    # Compute Rotation matrix to rotate cylinder to given axis
    axis = torch.from_numpy(axis).float().cuda()
    axis = axis / torch.linalg.norm(axis)
    rotmat = rotvec_to_matrix(axis).cpu().numpy()
    pose = np.eye(4)
    pose[:3, :3] = rotmat
    capped_cylinder.apply_transform(pose)

    # Compute Burial offset and bury cylinder by that much.
    yrange = (
        capped_cylinder.vertices[:, 1].max() - capped_cylinder.vertices[:, 1].min()
    ) / 2
    burial_offset = burial_percent * yrange
    relative_burial = burial_offset / height
    pose = np.eye(4)
    pose[1, 3] = burial_offset
    capped_cylinder.apply_transform(pose)

    # Split the mesh based on burial
    plane_normal = np.array([0.0, 1.0, 0.0])
    plane_point = np.array([0.0, 0.0, 0.0])

    # Perform the split
    sliced_cylinder = capped_cylinder.slice_plane(
        plane_normal=plane_normal, plane_origin=plane_point
    )
    return sliced_cylinder, relative_burial


def get_KRT(fov=np.pi / 3, HW=[400, 400], cam_origin=[5.0, 0.0, 0.0]):
    """Compute K matrix and camera pose from mitsuba3 scene. Returns torch.tensor K (3x3), and RT [4x4]"""
    H, W = HW
    fx = W / (2 * np.tan(fov / 2))
    fy = fx
    cx, cy = W / 2, H / 2
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).float().cuda()
    RT = mi.Transform4f.look_at(
        origin=cam_origin,
        target=(1.1 * np.array(cam_origin)).tolist(),
        up=[0.0, 1.0, 0.0],
    ).matrix.numpy()[0]
    RT = RT
    return K, RT


def pyrender_depth(renderer, HW, cyl_mesh, fov, cam_origin):
    """takes a cylinder mesh posed and centered at origin, location of camera, fov, image size
    and returns depth, K, RT"""
    mesh = pyrender.Mesh.from_trimesh(cyl_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=HW[1] / HW[0])
    K, RT = get_KRT(fov, HW, cam_origin)

    scene.add(camera, pose=RT)
    color, depth = renderer.render(scene)
    return (
        torch.from_numpy(depth).float().cuda(),
        K,
        torch.from_numpy(RT).float().cuda(),
    )


def pc_from_depth(depthmap, K, cam2world=torch.eye(4).cuda()):
    """Compute point cloud in the world frame from depthmap and K matrix
    Args:
                depthmap (torch.tensor) M,N
                K (torch.tensor) [3,3]
                cam2world (torch.tensor) [4,4] pose matrix that goes from camera to world
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    height, width = depthmap.shape[:2]
    v, u = torch.meshgrid(
        torch.arange(width).cuda(), torch.arange(height).cuda(), indexing="ij"
    )
    u = u.flatten()
    v = v.flatten()
    depth = depthmap.flatten()
    X = (u - cx) * depth / fx
    Y = -(v - cy) * depth / fy
    Z = depth
    idx = depth > 0
    points = torch.stack([X[idx], Y[idx], -Z[idx]], dim=-1).reshape(-1, 3)
    points = points @ cam2world[:3, :3].T + cam2world[:3, 3:].T
    return points


def render_occluded_point_cloud(
    axisvec, fov, imsize, radius, height, cam_origin, burial_percent, renderer
):
    """Given Camera location, params and cylinder params, render the occluded and buried point cloud in world frame
    Args:
        axisvec (np.array) : (3,) axis vector of the cylinder
        fov (float) : camera fov
        imsize (list) : [H,W] size of the image
        radius (float) : radius of the cylinder
        height (float) : height of the cylinder
        cam_origin (list) : Position of the camera on a hemisphere (Y up) looking at origin
        burial_percent (float) : percent of the barrel to be buried. (can be +ve or negative)
    returns:
        points [N,3] - point cloud rendered from the camera frame in an occluded viewpoint.
        The points are in world coordinate system with Y up.
        relative_burial_percent : relative percentage of the buried barrel w.r.t actual height of the cylinder.
    """
    cyl_mesh, relative_burial_percent = get_posed_split_cylinder_mesh(
        radius, height, axisvec, burial_percent
    )
    depth, K, RT = pyrender_depth(renderer, imsize, cyl_mesh, fov, cam_origin)
    points = pc_from_depth(depth, K, RT)
    return points, relative_burial_percent


# axis = np.array([0., 0.5, 0.5])
# fov = np.pi/3
# HW = [400,400]
# radius = 1.0
# height = 2.5
# cam_origin = [0.0, 0.0,5.0]
