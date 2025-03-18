# Pose Estimation of Buried Deep-Sea Objects using 3D Vision Deep Learning Models

You want barrels? We got barrels.

## Environment setup

Just run this, I'll deal with a requirements file later. If anything is still missing after
installation just pip install it.

```shell
git clone https://github.com/jerukan/barrels.git
cd barrels
git submodule update --init --recursive
conda create --name barrels python=3.11
conda activate barrels
# if your CUDA setup isn't completely messed up, this can be skipped
conda install -c nvidia cuda
export CUDA_HOME=$CONDA_PREFIX
# install all through pip and call it a day
pip install jupyter plotly dill pyransac3d open3d transforms3d roma mitsuba shapely fake-bpy-module jax rtree mapbox-earcut manifold3d git+https://github.com/luca-medeiros/lang-segment-anything.git git+https://github.com/google-research/visu3d.git git+https://github.com/jerukan/pyrender.git
```

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
