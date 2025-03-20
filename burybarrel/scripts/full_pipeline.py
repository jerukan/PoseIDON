from pathlib import Path

import click
import yaml

from burybarrel.scripts.create_masks import _create_masks
from burybarrel.scripts.run_foundpose import _run_foundpose
from burybarrel.scripts.run_foundpose_fit import _run_foundpose_fit


@click.command()
@click.option(
    "-i",
    "--indir",
    "datadir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "-o",
    "--outdir",
    "resdir",
    required=True,
    type=click.Path(file_okay=False),
)
@click.option(
    "-m",
    "--modeldir",
    "objdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "-d",
    "--device",
    "device",
    type=click.STRING,
)
@click.option(
    "--step-mask",
    "step_mask",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Run SAM mask creation step"
)
@click.option(
    "--step-foundpose",
    "step_foundpose",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Run foundpose template, repre, and inference step"
)
@click.option(
    "--step-fit",
    "step_fit",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Run multiview fitting step"
)
def run_full_pipeline(datadir, resdir, objdir, device=None, step_mask=False, step_foundpose=False, step_fit=False):
    datadir = Path(datadir)
    resdir = Path(resdir)
    objdir = Path(objdir)
    with open(objdir / "model_info.json", "rt") as f:
        model_info = yaml.safe_load(f)
    with open(datadir / "info.json", "rt") as f:
        data_info = yaml.safe_load(f)
    text_prompt = model_info[data_info["object_name"]]["descriptor"]
    imgdir = datadir / "rgb"
    maskdir = resdir / "sam-masks"
    if step_mask:
        _create_masks(imgdir, text_prompt, maskdir, closekernelsize=5, convexhull=True, device=device)
    if step_foundpose:
        _run_foundpose(datadir, resdir, objdir, Path("/home/jeyan/Projects/barrel-playground/otherrepos/foundpose"), pythonbinpath=Path("/scratch/jeyan/conda/envs/foundpose_gpu_311/bin/python"), device=device)
    if step_fit:
        _run_foundpose_fit(datadir, resdir, objdir, use_coarse=True, use_icp=True, seed=0, device=device)
        _run_foundpose_fit(datadir, resdir, objdir, use_coarse=True, use_icp=False, seed=0, device=device)
        _run_foundpose_fit(datadir, resdir, objdir, use_coarse=False, use_icp=True, seed=0, device=device)
        _run_foundpose_fit(datadir, resdir, objdir, use_coarse=False, use_icp=False, seed=0, device=device)
