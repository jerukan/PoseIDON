# PoseIDON: 6DoF pose estimation with foundation model features for marine sediment burial mapping

Authors: Jerry Yan, Chinmay Talegaonkar, Nicholas Antipa, Eric Terrill, Sophia Merrifield

Published to the ISPRS Journal of Photogrammetry and Remote Sensing (2026).

PoseIDON: **Pose** **I**dentification for **D**epth of **O**bjects via foundation model **N**etworks.

This repository contains the associated code for my thesis and paper of the same name.

The following are links to the [thesis](https://www.proquest.com/dissertations-theses/leveraging-6dof-pose-foundation-models-mapping/docview/3226985070/se-2?accountid=14524), [arXiv](https://arxiv.org/abs/2506.10386), and [paper](https://www.sciencedirect.com/science/article/pii/S0924271626003382).

## Sample data download

Find a sample of the datasets used in the paper at the following [download link](https://drive.google.com/drive/folders/16PTu4ZkgkVIt6lY1XPQzU3qyW280G-qV?usp=drive_link).

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
conda env create --name foundpose_gpu_311 --file environment_gpu.yml
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

- `datasets-folder/`: contains all input image data
	- `dataset-name/`
		- `rgb/`: contains all original RGB images
		- `mask/`: (if available) ground truth mask of the object, with the corresponding image name matching the one in `rgb/`
		- `gt-overlays/`: (if available) ground truth overlay of the object over the original RGB image
		- `camera.json`: camera intrinsics
		- `gt_obj2cam.json`: the ground truth 6DoF translation and rotation relative to the camera for each image
		- `frame-time-nav.csv`: the ROV readings for lat/lon, depth, and orientation for each image
		- `info.json`: miscellaneous information like GT burial depth, lat/lon position, or textual description of scene
- `results-folder/`: contains model outputs
	- `dataset-name/`
		- `colmap-out/`: photogrammetry output from COLMAP
			- `cam_poses.json`: world camera poses predicted by COLMAP
			- `sparse.ply`: sparse point cloud predicted by COLMAP
			- other COLMAP output
		- `openmvs-out/`: dense reconstruction of scene with OpenMVS
			- `scene_dense.ply`: dense point cloud
			- `scene_dense_mesh_refine_texture.obj`: textured mesh reconstruction for visualization purposes
			- other OpenMVS output
		- `sam-masks/`: masks predicted by Grounding DINO + SAM 2
			- `masksinfo.json`: information on bounding boxes, masks, scores, etc predicted for each image
			- the masks will be binary masks with the same name corresponding to the original image
		- `foundpose-output/`: output generated by FoundPose
			- `templates/`: template renders of the CAD model
			- `object_repre/`: KNN model of DINOv2 features from the templates
			- `inference/`: outputs from FoundPose on input RGB images
				- `estimated-poses.json`: estimated 6DoF poses from FoundPose for each image, which may contain multiple hypotheses for each image
				- the image names for the output visualization will not be the same as the original image, but will be in the same order alphabetically
		- `fit-output/`: final output from 3D aggregation of FoundPose poses
			- `estimation-name-1/`: a prediction with specific settings
				- `fit-overlays/`: visualizations of outputs
				- `reconstruction-info.json`: information on depth, scale correction factor, and prediction settings
				- `estimated-poses.json`: 6DoF poses predicted for each image, in the same format as FoundPose inference output
- `models3d/`: contains information on 3D CAD models
	- [`model_info.json`](models3d/model_info.json): symmetry information and descriptors for each CAD model
	- CAD model files are located here, preferably `.ply` format

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

### Evaluation

To generate a spreadsheet of BOP metric results and burial depth errors, run the following:

```shell
python -m burybarrel get-metrics
```

Assuming you have ground truth obviously.

### Logging

Runtime logs should be located in the [logs/](logs/) directory.
