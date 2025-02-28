import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

from burybarrel.barrelnet.pointnet_utils import (
    PointNetEncoder,
    feature_transform_reguliarzer,
)
from burybarrel.barrelnet.data import pts2inference_format


class BarrelNet(nn.Module):
    def __init__(self, k=5, normal_channel=True):
        super(BarrelNet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat_normal = PointNetEncoder(
            global_feat=True, use_Tnet=False, channel=channel
        )
        self.feat_radius = PointNetEncoder(
            global_feat=True, use_Tnet=True, channel=channel
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        xn, _, _ = self.feat_normal(x)
        xr, _, _ = self.feat_radius(x)
        # x = xn
        x = (xn + xr) / 2  # in future make it (xn + xr)/2
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        radius = F.sigmoid(x[:, 3])
        zshift = F.tanh(x[:, 4]) * 0.5
        normal = torch.concatenate(
            [F.tanh(x[:, :1]), F.sigmoid(x[:, 1:2]), F.tanh(x[:, 2:3])], dim=1
        )
        normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True)
        return radius, zshift, normal
    
    def predict_np(self, pts, height_radius_ratio, device="cuda", max_points=1000):
        """
        Generates cylinder predictions for numpy input and output.
        
        Args:
            height_radius_ratio: ratio of height over radius
        """
        pts = torch.tensor(pts, device=device).float()
        pts, scale = pts2inference_format(pts, max_points=max_points)
        with torch.no_grad():
            radius_pred, zshift_pred, axis_pred = self(pts)
            radius_pred = radius_pred.float().cpu().numpy()[0]
            zshift_pred = zshift_pred.float().cpu().numpy()[0]
            axis_pred = axis_pred.cpu().numpy()[0]
        axis_pred = axis_pred / np.linalg.norm(axis_pred)
        r = scale * radius_pred
        h = r * height_radius_ratio
        z = zshift_pred * h
        return axis_pred, r, h, z
