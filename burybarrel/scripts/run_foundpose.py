import os
from pathlib import Path
import subprocess
import sys

import click
import yaml

from burybarrel import config, get_logger


logger = get_logger(__name__)


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
    default=config.FOUNDPOSE_PYTHON_BIN_PATH,
    type=click.Path(exists=True, file_okay=True),
    help="Path to Python binary (activate your virutal environment and do `which python`)"
)
@click.option(
    "-d",
    "--device",
    "device",
    type=click.STRING,
)
@click.option(
    "--gen-templates",
    "gen_templates",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Generate render templates",
)
@click.option(
    "--gen-repre",
    "gen_repre",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Generate BoW representations for templates",
)
@click.option(
    "--infer",
    "_infer",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Run inference on a dataset",
)
def run_foundpose(datadir, resdir, objdir, repopath, pythonbinpath=None, device=None, gen_templates=False, gen_repre=False, _infer=False):
    _run_foundpose(datadir, resdir, objdir, repopath, pythonbinpath=pythonbinpath, device=device, gen_templates=gen_templates, gen_repre=gen_repre, _infer=_infer)

def _run_foundpose(datadir, resdir, objdir, repopath, pythonbinpath=None, device=None, gen_templates=False, gen_repre=False, _infer=False):
    """
    Run from a modified foundpose repo from here to generate foundpose output in the
    desired file structure.

    And reduce the clutter of configs in foundpose probably.
    """
    if not (gen_templates or gen_repre or _infer):
        logger.warning("No steps selected to run. Assuming you want to run everything.")
        gen_templates = True
        gen_repre = True
        _infer = True

    if device is None:
        import torch
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
    # set to true later
    # basetemplate["infer_opts"]["vis_results"] = False
    newcfgpath = foundpose_outdir / "config.yaml"
    with open(newcfgpath, "wt") as f:
        yaml.safe_dump(basetemplate, f)

    envvars = {
        "REPO_PATH": str(repopath),
        "PYTHONPATH": f"{repopath}:{repopath / 'external/bop_toolkit'}:{repopath / 'external/dinov2'}",
    }
    env = dict(os.environ, **envvars)
    if pythonbinpath is not None:
        pycmd = str(pythonbinpath)
    else:
        pycmd = "python"
    runcmd = [pycmd, "scripts/pipeline.py", "--cfg", str(newcfgpath)]
    if gen_templates:
        runcmd.append("--gen-templates")
    if gen_repre:
        runcmd.append("--gen-repre")
    if _infer:
        runcmd.append("--infer")
    # runcmd = [pycmd, "scripts/pipeline.py", "--cfg", str(newcfgpath), "--gen-templates", "--gen-repre", "--infer"]
    # runcmd = [pycmd, "scripts/pipeline.py", "--cfg", str(newcfgpath), "--infer"]
    try:
        subprocess.run(
            runcmd,
            cwd=repopath, env=env, check=True, stderr=sys.stderr, stdout=sys.stdout
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error with foundpose:\n{e}")
        raise
