# Pose Estimation of Buried Deep-Sea Objects using 3D Vision Deep Learning Models

You want barrels? We got barrels.

## Environment setup

```shell
git clone https://github.com/jerukan/barrels.git
cd barrels
git submodule update --init --recursive
conda env create --name barrels --file environment.yml
conda activate barrels
### if your CUDA setup isn't completely messed up, this can be skipped ###
conda install -c nvidia cuda
export CUDA_HOME=$CONDA_PREFIX
### cuda shenanigans end ###
```

Next, Foundpose dependencies must be set up in a separate environment because it uses
faiss, which is incompatible with numpy 2.x, and I think trying to force 1.x numpy here might
cause problems. Probably.

```shell
cd foundpose
conda env create --name foundpose_gpu_311 --file environment.yml
cd ..
```

Afterwards, go into [burybarrel/config.py](burybarrel/config.py) and change the path to the
FoundPose Python environment.

```python
FOUNDPOSE_PYTHON_BIN_PATH = Path("/path/to/environment/path/foundpose_gpu_311/bin/python")
```

For densifying point clouds and generating 3D reconstruction meshes from
[COLMAP](https://github.com/colmap/colmap), we use [OpenMVS](https://github.com/cdcseacave/openMVS)
since it can do so on a CPU. Getting densification workin on either COLMAP or OpenMVS requires
rebuilding the repository from source, so regardless we'll have to suffer from CMake.

```shell
# TODO
```

### Optional Fast3r and VGGT setup

## Script running

Display list of available commands:

```shell
python -m burybarrel --help
```

Running them:

```shell
python -m burybarrel script-name [ARGS]
```

### Long running scripts in the background

Use tmux, or if you want, use the `nohup` command as follows:

```shell
nohup python -m burybarrel script-name [ARGS] &
```

Output will go to `nohup.out`.

## Data/results file structure organization

- `datasets/`
	- `dataset-name/`
		- `rgb/`
		- `mask/`
		- `gt-overlays/`
		- `camera.json`
		- `gt_obj2cam.json`
		- `info.json` (contains model path, etc)
- `results/`
	- `dataset-name/`
		- `colmap-out/`
			- `cam_poses.json`
			- `sparse.ply`
			- everything else colmap
		- `openmvs-out/`
		- `sam-masks/`
		- `foundpose-output/`
			- foundpose output format stuff
		- `fit-output/`
			- `estimation-name-1/`
				- `fit-overlays`
				- `config.json`
				- `estimated-poses.json`
- `models3d/`
	- `model_info.json` (symmetry info for now)
	- .ply files here
