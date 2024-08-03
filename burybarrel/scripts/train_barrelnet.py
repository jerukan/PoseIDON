import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import uuid
import yaml
import roma
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader

from burybarrel.barrelnet.data import (
    generate_cylinder_pts,
    prepare_point_cloud,
    normalize_pc,
    normalize_pc_train,
)
from burybarrel.barrelnet.pointnet_utils import (
    PointNetEncoder,
    feature_transform_reguliarzer,
)
from burybarrel.barrelnet.barrelnet import BarrelNet
from burybarrel.barrelnet.data import CylinderDataOccluded


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def compute_loss(
    sample,
    radius_pred,
    zshift_pred,
    axis_pred,
    use_radius_loss=True,
    use_axis_loss=True,
    use_loss_zshift=True,
):
    """Compute loss on the predictions of pointnet"""
    assert use_axis_loss or use_radius_loss, "Atleast one of the losses should be used"
    loss_axis = (
        1 - F.cosine_similarity(sample["axis_vec"].cuda(), axis_pred, dim=1)
    ).mean()
    loss_radius = F.mse_loss(radius_pred, sample["radius_gt"].cuda())
    loss_zshift = F.mse_loss(zshift_pred, sample["burial_z"].cuda())
    loss = 0.0
    if use_radius_loss:
        loss = loss + loss_radius
    if use_axis_loss:
        loss = loss + loss_axis
    if use_loss_zshift:
        loss = loss + loss_zshift
    return loss, loss_radius, loss_axis, loss_zshift


def train(
    model, train_loader, optimizer, scheduler, writer, config, save_dir="weights/r0"
):
    """
    ## TODO: Too many train config parameters. Pass them as a dictionary through a yaml file or something.
        Train the model for num_epochs
        model : BarrelNet Model instance
    """
    ## Parsing config params
    save_dir = config["save_dir"]
    num_epochs = config["train"]["num_epochs"]
    save_epoch = config["train"]["save_epoch"]
    test_epoch = config["train"]["test_epoch"]
    save_dir = config["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    model.train()  # Set model to training mode
    print("Starting training")
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_loss_radius = 0.0
        running_loss_axis = 0.0
        running_loss_zshift = 0.0
        for i, sample in tqdm(enumerate(train_loader)):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            pts = sample["pts"]
            input, scale = normalize_pc_train(pts.cuda())
            radius_pred, zshift_pred, axis_pred = model(input)
            loss, loss_radius, loss_axis, loss_zshift = compute_loss(
                sample,
                radius_pred * scale,
                zshift_pred,
                axis_pred,
                use_radius_loss=True,
                use_axis_loss=True,
                use_loss_zshift=True,
            )
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_radius += loss_radius.item()
            running_loss_axis += loss_axis.item()
            running_loss_zshift += loss_zshift.item()

        writer.add_scalar(
            "Total_training_loss", running_loss / len(train_loader), epoch
        )
        writer.add_scalar(
            "Radius_training_loss", running_loss_radius / len(train_loader), epoch
        )
        writer.add_scalar(
            "Axis_training_loss", running_loss_axis / len(train_loader), epoch
        )
        writer.add_scalar(
            "Zshift_training_loss", running_loss_zshift / len(train_loader), epoch
        )

        # Step the scheduler at the end of each epoch
        scheduler.step()
        if epoch % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, LR: {scheduler.get_last_lr()[0]}"
            )

        if epoch % save_epoch == 0:
            torch.save(
                model.state_dict(), os.path.join(save_dir, f"pointnet_iter{epoch}.pth")
            )

        if epoch % test_epoch == 0:
            test(model, test_loader, writer, epoch)


def test(model, test_loader, writer, epoch=0):
    model.eval()
    running_acc_axis = 0.0
    running_acc_radius = 0.0
    running_acc_zshift = 0.0

    criterion_cosine = nn.CosineSimilarity(dim=1)
    for sample in tqdm(test_loader):
        with torch.no_grad():
            pts = sample["pts"]
            input, scale = normalize_pc_train(pts.cuda())
            radius_pred, zshift_pred, axis_pred = model(input)
            radius_pred = radius_pred * scale
            acc_axis = (
                1 + criterion_cosine(sample["axis_vec"].cuda(), axis_pred).mean()
            ) / 2
            acc_radius = torch.abs(radius_pred - sample["radius_gt"].cuda())
            acc_zshift = torch.abs(zshift_pred - sample["burial_z"].cuda())

            running_acc_axis += acc_axis.item()
            running_acc_radius += acc_radius.item()
            running_acc_zshift += acc_zshift.item()

    writer.add_scalar(
        "Mean Axis Test Accuracy", running_acc_axis / len(test_loader), epoch
    )
    writer.add_scalar(
        "Mean Radius Test Error", running_acc_radius / len(test_loader), epoch
    )
    writer.add_scalar(
        "Mean ZShift Test Error", running_acc_zshift / len(test_loader), epoch
    )


def run():
    config = load_config("/home/ubuntu/dust3r/configs/config.yaml")
    dirname = str(uuid.uuid4()).replace("-", "")[:6]
    save_dir = os.path.join("logs", dirname, "weights")
    config["train"]["save_dir"] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(f"logs/{dirname}")

    train_data = CylinderDataOccluded(dataset_loadpath=config["data"]["train_dir"])
    test_data = CylinderDataOccluded(dataset_loadpath=config["data"]["test_dir"])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    pointnet = BarrelNet(k=5, normal_channel=False).cuda()
    pointnet.train()
    optimizer = optim.Adam(pointnet.parameters(), lr=config["optimizer"]["lr"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["optimizer"]["step_size"],
        gamma=config["optimizer"]["gamma"],
    )

    with open(os.path.join("logs", dirname, "cfg.yaml"), "w") as file:
        yaml.safe_dump(config, file)
    train(pointnet, train_loader, optimizer, scheduler, writer, config)
