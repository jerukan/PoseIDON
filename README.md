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
conda install -c nvidia cuda
export CUDA_HOME=$CONDA_PREFIX
pip install git+https://github.com/luca-medeiros/lang-segment-anything.git
pip install git+https://github.com/google-research/visu3d.git
pip install git+https://github.com/jerukan/pyrender.git
pip install jupyter plotly dill pyransac3d open3d transforms3d roma mitsuba
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
