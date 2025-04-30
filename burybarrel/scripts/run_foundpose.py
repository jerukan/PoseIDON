import os
from pathlib import Path
import subprocess

import click
import torch
import yaml


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
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "-m",
    "--modeldir",
    "objdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--repopath",
    "repopath",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to the foundpose repo base directory",
)
@click.option(
    "--pythonbin",
    "pythonbinpath",
    required=False,
    type=click.Path(exists=True, file_okay=True),
    help="Path to Python binary (activate your virutal environment and do `which python`)"
)
@click.option(
    "-d",
    "--device",
    "device",
    type=click.STRING,
)
def run_foundpose(datadir, resdir, objdir, repopath, pythonbinpath=None, device=None):
    _run_foundpose(datadir, resdir, objdir, repopath, pythonbinpath=pythonbinpath, device=device)

def _run_foundpose(datadir, resdir, objdir, repopath, pythonbinpath=None, device=None):
    """
    Run from a modified foundpose repo from here to generate foundpose output in the
    desired file structure.

    And reduce the clutter of configs in foundpose probably.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    datadir = Path(datadir)
    resdir = Path(resdir)
    objdir = Path(objdir)
    repopath = Path(repopath).absolute()

    foundpose_outdir = resdir / "foundpose-output"
    foundpose_outdir.mkdir(exist_ok=True, parents=True)

    with open(datadir / "info.json", "rt") as f:
        datainfo = yaml.safe_load(f)
    basetemplate_path = repopath / "configs/seabed-template.yaml"
    with open(basetemplate_path, "rt") as f:
        basetemplate = yaml.safe_load(f)
    basetemplate["common_opts"]["object_path"] = str(objdir / datainfo["object_name"])
    basetemplate["common_opts"]["output_path"] = str(foundpose_outdir)
    basetemplate["common_opts"]["cam_json_path"] = str(datadir / "camera.json")
    # barrels are because they're bigger and further away i guess
    if datainfo["object_name"] == "barrelsingle-scaled.ply":
        basetemplate["gen_templates_opts"]["light_intensity"] = 120.0
    basetemplate["common_opts"]["device"] = device
    basetemplate["infer_opts"]["dataset_path"] = str(datadir / "rgb")
    basetemplate["infer_opts"]["mask_path"] = str(resdir / "sam-masks")
    newcfgpath = foundpose_outdir / "config.yaml"
    with open(newcfgpath, "wt") as f:
        yaml.safe_dump(basetemplate, f)

    envvars = {
        "REPO_PATH": str(repopath),
        "PYTHONPATH": f"{repopath}:{repopath / 'external/bop_toolkit'}:{repopath / 'external/dinov2'}",
    }
    env = dict(os.environ, **envvars)
    if pythonbinpath is not None:
        runcmd = [str(pythonbinpath), "scripts/pipeline.py", "--cfg", str(newcfgpath), "--gen-templates", "--gen-repre", "--infer"]
    else:
        runcmd = ["python", "scripts/pipeline.py", "--cfg", str(newcfgpath), "--gen-templates", "--gen-repre", "--infer"]
    subprocess.run(
        runcmd,
        cwd=repopath, env=env, check=True
    )
