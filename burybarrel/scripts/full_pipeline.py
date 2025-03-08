from pathlib import Path

import click

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
def run_full_pipeline(datadir, resdir, objdir, device=None):
    datadir = Path(datadir)
    resdir = Path(resdir)
    objdir = Path(objdir)
    imgdir = datadir / "rgb"
    maskdir = resdir / "sam-masks"
    _create_masks(imgdir, "underwater debris", maskdir, closekernelsize=5, convexhull=True, device=device)
    _run_foundpose(datadir, resdir, objdir, Path("/home/jeyan/Projects/barrel-playground/otherrepos/foundpose"), pythonbinpath=Path("/scratch/jeyan/conda/envs/foundpose_gpu_311/bin/python"))
    _run_foundpose_fit(datadir, resdir, objdir, use_coarse=True, use_icp=True)
