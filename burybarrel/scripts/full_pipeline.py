import concurrent.futures
from pathlib import Path
import time
import traceback

import click
import torch
import yaml

from burybarrel import config, get_logger, add_file_handler, log_dir
from burybarrel.scripts.create_masks import _create_masks
from burybarrel.scripts.run_foundpose import _run_foundpose
from burybarrel.foundpose_fit import load_fit_write


logger = get_logger(__name__)
add_file_handler(logger, log_dir / "fullpipelineruns.log")


cuda_count = torch.cuda.device_count()
if cuda_count > 0:
    default_devices = [f"cuda:{i}" for i in range(cuda_count)]
else:
    default_devices = ["cpu"]


@click.command()
@click.option(
    "-n",
    "--name",
    "names",
    required=True,
    type=click.STRING,
    multiple=True,
    help="dataset names to run. input multiple times for multiple datasets. Use 'all' to run all datasets in the input directory",
)
@click.option(
    "-i",
    "--indir",
    "datadir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    default=config.DEFAULT_DATA_DIR,
    show_default=True,
)
@click.option(
    "-o",
    "--outdir",
    "resdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    default=config.DEFAULT_RESULTS_DIR,
    show_default=True,
)
@click.option(
    "-m",
    "--modeldir",
    "objdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    default=config.DEFAULT_MODEL_DIR,
    show_default=True,
)
@click.option(
    "-d",
    "--device",
    "devices",
    type=click.STRING,
    help="cuda devices to allocate",
    multiple=True,
    default=default_devices,
    show_default=True,
)
@click.option(
    "--step-all",
    "step_all",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Run ALL steps in the pipeline (overrides other step flags)"
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
def run_full_pipelines(names, datadir, resdir, objdir, devices=None, step_all=False, step_mask=False, step_foundpose=False, step_fit=False):
    """
    Run the full pipeline on multiple datasets in parallel using multiple GPUs.

    The datasets are assigned to GPUs in a round-robin fashion.

    TODO: run colmap and openmvs from here too (uh installing openmvs without sudo is hard)
    """
    datadir = Path(datadir)
    resdir = Path(resdir)
    objdir = Path(objdir)
    if step_all:
        step_mask = True
        step_foundpose = True
        step_fit = True
    logger.info(f"RUNNING FULL PIPELINE ON {names} WITH DEVICES {devices}")
    ndevices = len(devices)
    devicetaskdict = {device: [] for device in devices}
    datadir.is_dir()
    if "all" in [n.lower() for n in names]:
        alldatadirs = filter(lambda x: x.is_dir() and (x / "info.json").exists(), datadir.glob("*"))
        names = [x.name for x in alldatadirs]
    # round robin assignment of datasets to devices
    for i, name in enumerate(names):
        devicetaskdict[devices[i % ndevices]].append(name)
    logger.info("DATASETS TO RUN ON EACH GPU:")
    logger.info(devicetaskdict)
    # ProcessPoolExecutor should make more sense, but it just fucking instantly deadlocks
    # from SAM imports and everything else and I can't be bothered to figure it out
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        future_to_res = {
            executor.submit(_run_pipelines_gpu, devnames, datadir, resdir, objdir, device=device, step_mask=step_mask, step_foundpose=step_foundpose, step_fit=step_fit): (device, devnames)
            for device, devnames in devicetaskdict.items()
        }
        for future in concurrent.futures.as_completed(future_to_res):
            device, devnames = future_to_res[future]
            try:
                future.result()
                logger.info(f"finished datasets {device}")
            except Exception as e:
                logger.error(f"exception in {device} for datasets {devnames}: {e}")
    logger.info("FINISHED PROCESSING ALL DATASETS")


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
        mask_t0 = time.time()
        _create_masks(imgdir, text_prompt, maskdir, closekernelsize=5, convexhull=True, device=device)
        mask_t1 = time.time()
        logger.info(f"Mask creation for dataset {name} took {mask_t1 - mask_t0:.2f} seconds")
    if step_foundpose:
        # TODO this needs to be generalized for foundpose parameters lol
        foundpose_t0 = time.time()
        _run_foundpose(datadir, resdir, objdir, "foundpose", pythonbinpath=config.FOUNDPOSE_PYTHON_BIN_PATH, device=device)
        foundpose_t1 = time.time()
        logger.info(f"Foundpose for dataset {name} took {foundpose_t1 - foundpose_t0:.2f} seconds")
    if step_fit:
        fit_t0 = time.time()
        load_fit_write(datadir, resdir, objdir, use_coarse=True, use_icp=True, reconstr_type="colmap", seed=0, device=device)
        fit_t1 = time.time()
        logger.info(f"Multiview fitting for dataset {name} took {fit_t1 - fit_t0:.2f} seconds")
        if (resdir / "fast3r-out").exists():
            load_fit_write(datadir, resdir, objdir, use_coarse=True, use_icp=True, reconstr_type="fast3r", seed=0, device=device)
        if (resdir / "vggt-out").exists():
            load_fit_write(datadir, resdir, objdir, use_coarse=True, use_icp=True, reconstr_type="vggt", seed=0, device=device)


def _run_pipelines_gpu(names, datadir, resdir, objdir, device=None, step_mask=False, step_foundpose=False, step_fit=False):
    """
    Runs a series of datasets iteratively on a single GPU.

    I'm too lazy to figure out how to make each GPU dynamically take the next dataset when it's
    done, so this will have to do.
    """
    for name in names:
        try:
            _run_full_pipeline(name, datadir, resdir, objdir, device=device, step_mask=step_mask, step_foundpose=step_foundpose, step_fit=step_fit)
        except Exception as e:
            logger.error(f"ERROR IN RUNNING {name} with exception: {e}\n{traceback.format_exc()}\nContinuing to next dataset")
