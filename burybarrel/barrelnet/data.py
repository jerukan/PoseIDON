from pathlib import Path

import torch
from tqdm import tqdm
import roma
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import trimesh
import pyrender
from scipy.spatial.transform import Rotation as R
from burybarrel.barrelnet.occlusion import *
import os
import gc

os.environ["PYOPENGL_PLATFORM"] = "egl"


def pad_point_cloud(point_cloud, max_points=1000):
    """
    Pads the input point cloud with zeros to ensure it has exactly max_points.
    Args:
        point_cloud (torch.Tensor): The input point cloud tensor of shape (N, C).
        max_points (int): The desired number of points in the output tensor.
    Returns:
        torch.Tensor: The padded point cloud tensor of shape (max_points, C).
    """
    num_points, num_channels = point_cloud.shape
    padded_cloud = torch.zeros(
        (max_points, num_channels), dtype=point_cloud.dtype, device=point_cloud.device
    )
    length = min(num_points, max_points)
    perm_idx = torch.randperm(point_cloud.shape[0])
    pc = point_cloud[perm_idx]
    padded_cloud[:length, :] = pc[:length, :]
    return padded_cloud


def generate_cylinder_pts(radius, height, sections=64, num_pts=1000, noise_level=0):
    """
    Create a capped cylinder using trimesh.

    Args:
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.
        sections (int): Number of sections for the cylinder's roundness.

    Returns:
        points (torch.tensor) : [num_pts, 3]
    """
    # Create the capped cylinder with cap=True
    capped_cylinder = trimesh.creation.cylinder(
        radius=radius, height=height, sections=sections, cap=True
    )
    points, _ = trimesh.sample.sample_surface(capped_cylinder, num_pts)
    points = torch.from_numpy(points).float().cuda()
    noise = torch.randn_like(points).cuda() * noise_level
    points = points + noise
    return points


def rotate_to_axis(point_cloud, axis, device="cuda"):
    """
    Rotate the point cloud to align its z-axis with the given axis using PyTorch.

    Args:
        point_cloud (torch.Tensor): Input point cloud of shape (N, 3).
        axis (torch.Tensor): Target axis to align with, should be of shape (3,).
        device (str): Device to perform the computation ('cpu' or 'cuda').
    Returns:
        torch.Tensor: Rotated point cloud of shape (N, 3).
    """
    ## TODO: make this batched, to rotate the point cloud to different configurations at once.
    # Normalize the axis
    axis = axis / torch.linalg.norm(axis)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device="cuda")

    # Compute the rotation axis and angle
    rotation_vector = torch.cross(z_axis, axis)
    angle = torch.arccos(torch.dot(z_axis, axis))

    # Convert rotation vector and angle to a PyTorch tensor
    rotation_vector = torch.tensor(rotation_vector, dtype=torch.float32, device=device)
    angle = torch.tensor(angle, dtype=torch.float32, device=device)

    # Create the rotation matrix using torch-roma
    rotation_matrix = roma.rotvec_to_rotmat(rotation_vector * angle)

    # Apply rotation to the point cloud
    point_cloud_rotated = torch.matmul(point_cloud, rotation_matrix.T)

    return point_cloud_rotated


def prepare_point_cloud(
    points, normal, hr_ratio_range=[1 / 4, 1 / 2], max_burial_percent=0.7
):
    ## TODO : Maybe Subsume this in the point cloud class
    """
        points (torch.tensor) [N,3] : sampled surface of full cylinder in standard XYZ coordinate system.
        normal (torch.tensor) [3,]: normal vector to rotate
        hr_ratio_range (list) : [min, max] range of the height ratio
    Returns:
        valid_pts (torch.tensor) : Rotated point cloud, and clipped so that only points above ocean floor are visible
        scale (float) - factor by which radius is to be scaled, for easier regression by pointnet.
        radius (float) - randomly sampled radius of the cylinder . It's unscaled
        burial_z (float) - Relative Displacement of Barrel Center along Z axis inside / outside the ocean floor w.r.t height of the cylinder
    """
    point_cloud = points.clone().to(points.device)
    radius = hr_ratio_range[0] + torch.rand(1)[0] * (
        hr_ratio_range[1] - hr_ratio_range[0]
    )
    point_cloud[
        :, :2
    ] *= radius  # Scaling height of the cylinder to be 2.5 - 3.5 times the radius
    point_cloud_rot = rotate_to_axis(point_cloud, normal)

    ## TODO: Double Check this, the burial part is kinda weird
    ## Total Z extent spanned by the barrel after rotation.
    z_range = (
        torch.max(points[:, -1]) - torch.min(points[:, -1])
    ).item()  # z_range is going to dot product of normal_range and Z axis, since height=1.0
    z_rot_range = (
        torch.max(point_cloud_rot[:, -1]) - torch.min(point_cloud_rot[:, -1])
    ).item()

    burial_z = (
        max_burial_percent * (torch.rand(1)[0] - 0.5) * z_rot_range / z_range
    )  # Relative Displacement of Barrel Center along Z axis inside / outside the ocean floor w.r.t height of the cylinder
    burial_offset = burial_z * z_range
    point_cloud_rot[:, -1] += burial_offset
    valid_pts = point_cloud_rot[point_cloud_rot[:, -1] > 0]

    ## TODO : Add code here to compute burial fraction, for synthetic data.
    return valid_pts, radius, burial_z


def normalize_pc(valid_pts):
    """
    Normalize standard N,C point clouds
    Args:
        valid_pts (torch.tensor) N,C - Point Cloud (batch, num_channels, num_pts)
    Returns:
        pts (torch.tensor) N,C normalized Point Cloud
        scale (float) - value with which to divide the actual radius, for input to network.
    """
    ## Point Cloud Prep for Feeding it to pointnet
    ## Estimating the scale that needs to be given to the point cloud as input
    pts = valid_pts - valid_pts.mean(dim=0, keepdim=True)
    scale = torch.max(torch.linalg.norm(pts, dim=-1, keepdim=True), dim=0).values.item()
    pts = pts / scale
    return pts, scale


def normalize_pc_train(pts):
    """
    Normalize the point cloud B,C,N to feed in to point net. Also estimate the scaling factor.
    Args:
        pts (torch.tensor) B,C,N - Point Cloud (batch, num_channels, num_pts)
    Returns:
        pts (torch.tensor) B,C,N normalized Point Cloud
        scale (float) (B,1,1) - value with which to divide the actual radius, for input to network.
    """
    ## Point Cloud Prep for Feeding it to pointnet
    ## Estimating the scale that needs to be given to the point cloud as input
    pts = pts - pts.mean(dim=-1, keepdim=True)
    scale = torch.max(
        torch.linalg.norm(pts, dim=1, keepdim=True), dim=-1, keepdim=True
    ).values
    pts = pts / scale
    return pts, scale.squeeze(-1).squeeze(-1)


def pts2inference_format(points, max_points=1000):
    """Convert pts in the right format for input to pointnet
    Args:
        points: [N,3] torch tensor

    Returns:
        pts [1,3,N]
        scale
    """
    pts, scale = normalize_pc(points)
    pts = pad_point_cloud(pts, max_points)
    pts = pts.permute(1, 0)
    pts = pts.unsqueeze(0)
    return pts, scale


class CylinderData(Dataset):
    def __init__(
        self,
        num_poses=10000,
        num_surface_samples=3000,
        max_points=1000,
        transform=None,
        max_burial_percent=0.7,
        noise_level=0.0,
    ):
        """
        Args:
            num_poses : number of axis vectors to sample
            num_surface_samples : number of points on the surface of the cylinder to sample
            max_points : maximum number of points in a batch of the point cloud
            transform (callable, optional) : Optional transform to be applied
                on a sample.
        """
        self.points = generate_cylinder_pts(
            1.0, 1.0, num_pts=num_surface_samples, noise_level=noise_level
        )
        self.max_points = max_points
        self.normals = self.sample_normals(num_poses)
        self.radii = []
        self.burial_offsets = []
        self.scales = []
        self.pts = []

        ## TODO: Later remove this for loop, and make this one shot.
        for i in tqdm(range(num_poses)):
            valid_pts, radius, burial_offset = prepare_point_cloud(
                self.points, self.normals[i], max_burial_percent=max_burial_percent
            )
            if valid_pts.shape[0] <= 0:
                continue
            pts, scale = normalize_pc(valid_pts)
            self.radii.append(radius)
            self.burial_offsets.append(burial_offset)
            self.scales.append(scale)
            self.pts.append(pts)

        self.radii = torch.stack(self.radii)
        self.burial_offsets = torch.stack(self.burial_offsets)
        self.scales = torch.tensor(self.scales)

        self.transform = transform  ## Adding noise etc

    def pad_point_cloud(self, point_cloud):
        """
        Pads the input point cloud with zeros to ensure it has exactly max_points.
        Args:
            point_cloud (torch.Tensor): The input point cloud tensor of shape (N, C).
            max_points (int): The desired number of points in the output tensor.
        Returns:
            torch.Tensor: The padded point cloud tensor of shape (max_points, C).
        """
        num_points, num_channels = point_cloud.shape
        padded_cloud = torch.zeros(
            (self.max_points, num_channels),
            dtype=point_cloud.dtype,
            device=point_cloud.device,
        )
        length = min(num_points, self.max_points)
        perm_idx = torch.randperm(point_cloud.shape[0])
        pc = point_cloud[perm_idx]
        padded_cloud[:length, :] = pc[:length, :]
        return padded_cloud

    def sample_normals(self, num_samples, device="cuda"):
        """
        Sample random unit vectors uniformly distributed on the upper hemisphere.

        Args:
            num_samples (int): Number of unit vectors to sample.
            device (str): Device to perform the computation ('cpu' or 'cuda').

        Returns:
            torch.Tensor: Tensor of shape (num_samples, 3) containing the sampled unit vectors.
        """
        phi = torch.rand(num_samples, device=device) * 2 * torch.pi
        theta = torch.acos(torch.rand(num_samples, device=device))

        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        return torch.stack((x, y, z), dim=1)

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.pts)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            sample: (data, label) pair for the given index
        """
        pts = self.pad_point_cloud(self.pts[idx])
        sample = {
            "pts": pts.permute(1, 0),
            "scale_gt": self.scales[idx],
            "radius_gt": self.radii[idx],
            "axis_vec": self.normals[idx],
            "burial_z": self.burial_offsets[idx],
        }

        if self.transform:
            sample["pts"] = self.transform(sample["pts"])

        return sample


class CylinderDataOccluded(Dataset):
    def __init__(
        self,
        num_poses=10000,
        max_points=1000,
        max_burial_percent=0.7,
        noise_level=0.03,
        camera_radius_range=[1.5, 6.0],
        hr_ratio_range=[1 / 4, 1 / 2],
        theta_range=[0.18 * np.pi, 0.41 * np.pi],
        cam_views_per_pose=20,
        camera_fov=np.pi / 3,
        imsize=[512, 512],
        dataset_loadpath=None,
        dataset_savepath=None,
    ):
        """
        Args:
            num_poses : number of axis vectors to sample
            num_surface_samples : number of points on the surface of the cylinder to sample
            max_points : maximum number of points in a batch of the point cloud
            transform (callable, optional) : Optional transform to be applied
                on a sample.
        """
        self.num_poses = num_poses
        self.max_points = max_points
        self.theta_range = theta_range
        self.camera_radius_range = camera_radius_range
        self.cam_views_per_pose = cam_views_per_pose
        self.camera_fov = camera_fov
        self.imsize = imsize
        self.hr_ratio_range = hr_ratio_range
        self.noise_level = noise_level
        self.max_burial_percent = max_burial_percent
        self.dataset_load_file: Path = (
            Path(dataset_loadpath) if dataset_loadpath is not None else None
        )
        self.dataset_save_file: Path = (
            Path(dataset_savepath)
            if dataset_savepath is not None
            else Path("barrelsynth_data", "data.pkl")
        )
        self.renderer = pyrender.OffscreenRenderer(imsize[0], imsize[1])

        self.dataset_save_file.parent.mkdir(parents=True, exist_ok=True)

        if self.dataset_load_file is not None:
            with open(self.dataset_load_file, "rb") as f:
                data_dict = pickle.load(f)
                self.axisvecs = data_dict["axis_vectors"]
                self.radii = data_dict["radii"]
                self.burial_offsets = data_dict["burial_offsets"]
                self.pts = data_dict["pts"]
        else:
            self.generate_dataset()

    def generate_dataset(self):
        self.radii = []
        self.burial_offsets = []
        self.scales = []
        self.pts = []

        height = 1.0
        self.axisvecs = self.sample_normals(self.num_poses).cpu().numpy()
        self.radii = (
            np.random.uniform(
                self.hr_ratio_range[0], self.hr_ratio_range[1], self.num_poses
            )
            * height
        )
        self.burial_percentages = np.random.uniform(
            -self.max_burial_percent, self.max_burial_percent, self.num_poses
        )

        self.radii = np.expand_dims(self.radii, axis=-1).repeat(
            axis=-1, repeats=self.cam_views_per_pose
        )
        self.axisvecs = np.expand_dims(self.axisvecs, axis=1).repeat(
            axis=1, repeats=self.cam_views_per_pose
        )  # num_poses x cam_per_view x 3
        self.burial_percentages = np.expand_dims(
            self.burial_percentages, axis=-1
        ).repeat(axis=-1, repeats=self.cam_views_per_pose)
        self.burial_offsets = []
        for i in tqdm(range(self.num_poses)):
            cam_origins = self.sample_camera_views()
            cam_pts = []
            for j, origin in enumerate(cam_origins):
                points, relative_burial = render_occluded_point_cloud(
                    self.axisvecs[i][j],
                    self.camera_fov,
                    self.imsize,
                    self.radii[i][j],
                    height,
                    origin,
                    self.burial_percentages[i][j],
                    self.renderer,
                )
                noise = torch.randn_like(points).cuda() * self.noise_level
                points = points + noise
                cam_pts.append(points.cpu())
                self.burial_offsets.append(relative_burial)
            self.pts += cam_pts
        self.radii = torch.from_numpy(self.radii).float()
        self.burial_offsets = torch.from_numpy(np.array(self.burial_offsets)).float()
        ## Saving things to a pickle
        self.renderer.delete()
        gc.collect()

        with open(self.dataset_save_file, "wb") as f:
            datadict = {}
            datadict["axis_vectors"] = self.axisvecs.reshape(-1, 3)
            datadict["radii"] = self.radii.flatten()
            datadict["burial_offsets"] = self.burial_offsets.flatten()
            datadict["pts"] = self.pts
            pickle.dump(datadict, f)
            print("Saved dataset")

    def pad_point_cloud(self, point_cloud):
        """
        Pads the input point cloud with zeros to ensure it has exactly max_points.
        Args:
            point_cloud (torch.Tensor): The input point cloud tensor of shape (N, C).
            max_points (int): The desired number of points in the output tensor.
        Returns:
            torch.Tensor: The padded point cloud tensor of shape (max_points, C).
        """
        num_points, num_channels = point_cloud.shape
        padded_cloud = torch.zeros(
            (self.max_points, num_channels),
            dtype=point_cloud.dtype,
            device=point_cloud.device,
        )
        length = min(num_points, self.max_points)
        perm_idx = torch.randperm(point_cloud.shape[0])
        pc = point_cloud[perm_idx]
        padded_cloud[:length, :] = pc[:length, :]
        return padded_cloud

    def sample_normals(self, num_samples, device="cuda"):
        """
        Sample random unit vectors uniformly distributed on the upper hemisphere.

        Args:
            num_samples (int): Number of unit vectors to sample.
            device (str): Device to perform the computation ('cpu' or 'cuda').

        Returns:
            torch.Tensor: Tensor of shape (num_samples, 3) containing the sampled unit vectors.
        """
        phi = torch.rand(num_samples, device=device) * 2 * torch.pi
        theta = torch.acos(torch.rand(num_samples, device=device))

        x = torch.sin(theta) * torch.cos(phi)
        z = torch.sin(theta) * torch.sin(phi)
        y = torch.cos(theta)
        return torch.stack((x, y, z), dim=1)

    def sample_camera_views(self):
        """Returns Camera origins (np.array) [num_views,3] around the ocean floor. theta - angle from Y axis, phi angle on XZ plane"""
        theta_range, radius_range, num_views = (
            self.theta_range,
            self.camera_radius_range,
            self.cam_views_per_pose,
        )
        theta = np.random.uniform(theta_range[0], theta_range[1], num_views)
        r = np.random.uniform(radius_range[0], radius_range[1], num_views)
        phi = np.random.uniform(0, 2 * np.pi, num_views)
        cam_origins = np.stack(
            [
                r * np.sin(theta) * np.cos(phi),
                r * np.cos(theta),
                r * np.sin(theta) * np.sin(phi),
            ]
        ).T
        return cam_origins

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.pts)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            sample: (data, label) pair for the given index
        """
        pts = self.pad_point_cloud(self.pts[idx])
        sample = {
            "pts": pts.permute(1, 0),
            "radius_gt": self.radii[idx],
            "axis_vec": self.axisvecs[idx],
            "burial_z": self.burial_offsets[idx],  # TODO: Replace this name to burial_y
        }
        return sample
