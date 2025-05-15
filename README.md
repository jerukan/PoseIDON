# 6arrel: Pose Estimation of Buried Deep-Sea Objects using 3D Vision Deep Learning Models

You want barrels? We got barrels.

## Environment setup

Environment setup is quite involved and annoying, so brace yourself.

### Main repository setup

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

### FoundPose setup

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
FOUNDPOSE_PYTHON_BIN_PATH = Path("/path/to/conda/environment/foundpose_gpu_311/bin/python")
```

### OpenMVS setup

For densifying point clouds and generating 3D reconstruction meshes from
[COLMAP](https://github.com/colmap/colmap), we use [OpenMVS](https://github.com/cdcseacave/openMVS)
since it can do so on a CPU. Getting densification workin on either COLMAP or OpenMVS requires
rebuilding the repository from source, so regardless we'll have to suffer from CMake.

Note that installing requires root access (I couldn't find a way to install OpenMVS without it). If you're running this repository on a server and don't have root access, you'll have to run the reconstructions locally, and then copy the results onto the server.

#### MacOS

```shell
git clone https://github.com/cdcseacave/openMVS.git
git clone https://github.com/microsoft/vcpkg.git
brew install vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
export VCPKG_ROOT=path/to/barrels/vcpkg
brew install autoconf automake autoconf-archive
cd ../openMVS
mkdir make
cd make
cmake .. -DCMAKE_MAKE_PROGRAM=/usr/bin/make -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build . -j4
# this will install OpenMVS in /usr/local/bin/OpenMVS
# this is optional if you manually set the path to the binaries inside here
cmake --install .
cd ../..
```

#### Linux

```shell
git clone https://github.com/cdcseacave/openMVS.git
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
export VCPKG_ROOT=path/to/barrels/vcpkg
sudo apt-get install autoconf automake
cd ../openMVS
mkdir make
cd make
cmake .. -DCMAKE_MAKE_PROGRAM=/usr/bin/make -DCMAKE_CXX_COMPILER=/usr/bin/clang++
cmake --build . -j4
# this will install OpenMVS in /usr/local/bin/OpenMVS
# this is optional if you manually set the path to the binaries inside here
cmake --install .
cd ../..
```

### Optional Fast3r and VGGT setup

If you want to run the pipeline with recent deep-learning models for 3D reconstruction instead of classical photogrammetry, we have code to test the following models: [Fast3r](https://github.com/facebookresearch/fast3r) and [VGGT](https://github.com/facebookresearch/vggt).

Fast3r setup:

```shell
git clone https://github.com/facebookresearch/fast3r.git
cd fast3r
pip install -r requirements.txt
pip install -e .
cd ..
```

VGGT setup is similar:

```shell
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -r requirements.txt
pip install -e .
cd ..
```

## Running scripts

Display list of available commands:

```shell
python -m burybarrel --help
```

Running them:

```shell
python -m burybarrel script-name [ARGS]
```

### Long running scripts in the background

It's suggested to use [tmux](https://github.com/tmux/tmux/wiki).

Otherwise, you can use the `nohup` command as follows:

```shell
nohup python -m burybarrel script-name [ARGS] &
```

Output will go to `nohup.out`.

## Data/results file structure organization

There are three important directories:
1. input data directory
2. results directory
3. CAD model directory.

The general content of each are listed below.

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

## Running inference

This process is also annoying, so buckle up.

### Data download

May or may not be coming.

### Configure paths

Go to [burybarrel/config.py](burybarrel/config.py) and set the following variables to your own paths:

```python
DEFAULT_DATA_DIR = Path("path/to/input/data/dir")
DEFAULT_RESULTS_DIR = Path("path/to/output/results/dir")
DEFAULT_MODEL_DIR = Path("path/to/CAD/models/dir")

ONE_MACHINE = True
```

Most scripts have options to specify these paths too, so it's not neccessary to set this unless you want to retype the paths every time you run a script.

### Running the model

Provide video information in [configs/footage.yaml](configs/footage.yaml).

```yaml
dataset-name:
  input_path: path/to/video.mp4
  output_dir: path/to/output/results/dir
  start_time: ~
  timezone: US/Pacific
  step: ~
  navpath: ~
  # crop: [0, 120, 1920, 875]
  crop: ~
  maskpaths: [data/dive-data/footage-mask-hud.png]
  fps: 25
  increase_contrast: False
  denoise_depth: True
  object_name: ~
  description: ~
```

Get frames from a video.

```shell
python -m burybarrel get-footage-keyframes -n dataset-name
```

Perform 3D reconstruction.

```shell
python -m burybarrel reconstruct-colmap --sparse --dense -n dataset-name
```

Perform segmentation, FoundPose monocular pose estimates, and multiview pose aggregation.

```shell
python -m burybarrel run-full-pipelines --step-all -n dataset-name
```

### Logging

Runtime logs should be located in the [logs/](logs/) directory.
