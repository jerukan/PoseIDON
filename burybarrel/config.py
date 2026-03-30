from pathlib import Path


DEFAULT_DATA_DIR = Path("/Volumes/T7/backup/Projects/ms-stuff/barrel-playground/barrels/data/input_data")
DEFAULT_RESULTS_DIR = Path("/Volumes/T7/backup/Projects/ms-stuff/barrel-playground/barrels/results")
DEFAULT_MODEL_DIR = Path("/Volumes/T7/backup/Projects/ms-stuff/barrel-playground/models3d")

# If you have to run colmap reconstruction locally and then run everything else
# on another machine with CUDA, set this to False.
# otherwise, set it to True.
ONE_MACHINE = True
if ONE_MACHINE:
    DEFAULT_DATA_DIR_LOCAL = DEFAULT_DATA_DIR
    DEFAULT_RESULTS_DIR_LOCAL = DEFAULT_RESULTS_DIR
    DEFAULT_MODEL_DIR_LOCAL = DEFAULT_MODEL_DIR
else:
    DEFAULT_DATA_DIR_LOCAL = Path("data/input_data/")
    DEFAULT_RESULTS_DIR_LOCAL = Path("results/")
    DEFAULT_MODEL_DIR_LOCAL = Path("../models3d/")

FOUNDPOSE_PYTHON_BIN_PATH = Path("/Users/yan569/miniconda3/envs/foundpose_cpu")
