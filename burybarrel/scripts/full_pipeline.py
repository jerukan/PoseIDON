import concurrent.futures
from pathlib import Path

import click
import yaml

from burybarrel.scripts.create_masks import _create_masks
from burybarrel.scripts.run_foundpose import _run_foundpose
from burybarrel.scripts.run_foundpose_fit import _run_foundpose_fit


@click.command()
@click.option(
    "-n",
    "--name",
    "name",
    required=True,
    type=click.STRING,
)
@click.option(
    "-i",
    "--indir",
    "datadir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    default="/scratch/jeyan/barreldata/divedata/",
    show_default=True,
)
@click.option(
    "-o",
    "--outdir",
    "resdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    default="/scratch/jeyan/barreldata/results/",
    show_default=True,
)
@click.option(
    "-m",
    "--modeldir",
    "objdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    default="/scratch/jeyan/barreldata/models3d/",
    show_default=True,
)
@click.option(
    "-d",
    "--device",
    "device",
    type=click.STRING,
    help="cuda device"
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
def run_full_pipeline(**kwargs):
    _run_full_pipeline(**kwargs)


def _run_full_pipeline(name, datadir, resdir, objdir, device=None, step_mask=False, step_foundpose=False, step_fit=False):
    datadir = Path(datadir) / name
    resdir = Path(resdir) / name
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


def _run_pipelines_gpu(names, datadir, resdir, objdir, device=None, step_mask=False, step_foundpose=False, step_fit=False):
    for name in names:
        _run_full_pipeline(name, datadir, resdir, objdir, device=device, step_mask=step_mask, step_foundpose=step_foundpose, step_fit=step_fit)


@click.command()
@click.option(
    "-n",
    "--name",
    "names",
    required=True,
    type=click.STRING,
    multiple=True,
)
@click.option(
    "-i",
    "--indir",
    "datadir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    default="/scratch/jeyan/barreldata/divedata/",
    show_default=True,
)
@click.option(
    "-o",
    "--outdir",
    "resdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    default="/scratch/jeyan/barreldata/results/",
    show_default=True,
)
@click.option(
    "-m",
    "--modeldir",
    "objdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    default="/scratch/jeyan/barreldata/models3d/",
    show_default=True,
)
@click.option(
    "-d",
    "--device",
    "devices",
    type=click.STRING,
    help="cuda devices to allocate",
    multiple=True,
    default=[f"cuda:{i}" for i in range(1, 8)],
    show_default=True,
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
def run_full_pipelines(names, datadir, resdir, objdir, devices=None, step_mask=False, step_foundpose=False, step_fit=False):
    ndevices = len(devices)
    devicetaskdict = {device: [] for device in devices}
    for i, name in enumerate(names):
        devicetaskdict[devices[i % ndevices]].append(name)
    print("DATASETS TO RUN ON EACH GPU:")
    print(devicetaskdict)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        future_to_res = {
            executor.submit(_run_pipelines_gpu, names, datadir, resdir, objdir, device=device, step_mask=step_mask, step_foundpose=step_foundpose, step_fit=step_fit): device
            for device, names in devicetaskdict.items()
        }
        for future in concurrent.futures.as_completed(future_to_res):
            device = future_to_res[future]
            try:
                future.result()
                print(f"finished {device}")
            except Exception as e:
                print(f"exception in {device}: {e}")
                raise e
