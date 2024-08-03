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
import yaml

os.environ["PYOPENGL_PLATFORM"] = "egl"
mi.set_variant("cuda_ad_rgb")

from burybarrel.barrelnet.data import CylinderDataOccluded

# dset = CylinderDataOccluded(num_poses=100)


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def run():
    train_cfg = load_config("configs/config_datagen_train.yaml")
    test_cfg = load_config("configs/config_datagen_test.yaml")
    dset_test = CylinderDataOccluded(**test_cfg)
    dset_train = CylinderDataOccluded(**train_cfg)
