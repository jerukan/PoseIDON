import json
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback
import yaml

from burybarrel import get_logger
from burybarrel.utils import denoise_nav_depth
from burybarrel.image import apply_clahe


logger = get_logger(__name__)
DEFAULT_CFG_PATH = Path("configs/footage.yaml")


@click.command()
@click.option("-c", "--config", "cfg_path", default=DEFAULT_CFG_PATH, required=True, type=click.Path(exists=True, dir_okay=False), show_default=True, help="video informaton yaml file")
@click.option("-n", "--name", "names", required=True, type=click.STRING, help="name of data in the yaml config", multiple=True)
def get_footage_keyframes(cfg_path, names):
    with open(cfg_path, "rt") as f:
        cfg_all = yaml.safe_load(f)
    defaults = cfg_all["default"]
    for name in names:
        cfg = cfg_all[name]
        
        cfg_in = {
            **defaults,
            **cfg,
        }
        try:
            _get_footage_keyframes(**cfg_in)
        except Exception as e:
            logger.error(f"ERROR PROCESSING {name}: {e}\n{traceback.format_exc()}")
            continue


@click.command()
@click.option("-i", "--input", "input_path", required=False, type=click.Path(exists=True, dir_okay=False), help="input video path")
@click.option("-o", "--output", "output_dir", required=False, type=click.Path(file_okay=False), help="output dir (default: parent directory of video)")
@click.option("-t", "--time", "start_time", required=False, type=click.STRING, help="start time of the video (ISO 8601 format) (YYYY-MM-DDTHH:MM:SS)")
@click.option("--tz", "timezone", required=False, type=click.STRING, help="timezone from Olson tz database")
@click.option(
    "--fps", "fps", required=True, default=25, show_default=True, type=click.INT, help="Video FPS"
)
@click.option("--step", "step", required=True, type=click.INT, help="Number of frames between each image")
@click.option(
    "--crop", "crop", default=False, type=click.STRING, help="4 integers separated by commas: top left x, top left y, width, height"
)
@click.option(
    "--mask", "maskpath", required=False, type=click.Path(exists=True, dir_okay=False), help="path to mask"
)
@click.option(
    "--contrast",
    "increase_contrast",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="If provided, increases contrast of the images",
)
@click.option("--navpath", "navpath", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--denoisedepth",
    "denoise_depth",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Only used when navpath is provided; naively denoises depth data"
)
def get_footage_keyframes_cmd_old(
    **kwargs
):
    if "crop" in kwargs:
        croparg = kwargs["crop"]
        if croparg is not None:
            if croparg.count(",") != 3:
                raise ValueError("crop argument must have 4 values separated by commas")
            kwargs["crop"] = [int(x) for x in croparg.split(",")]
    _get_footage_keyframes(**kwargs)


def _get_footage_keyframes(
    input_path=None,
    output_dir=None,
    start_time=None,
    timezone=None,
    step=None,
    navpath=None,
    crop=None,
    maskpath=None,
    fps=None,
    increase_contrast=None,
    denoise_depth=None,
    object_name=None,
    description=None,
):
    """
    Retrieves keyframes from a video at specified intervals.

    Order of operations is contrast -> mask -> crop

    Args:
        output_dir: default is parent directory of video
        crop: [top left x, top left y, width, height]
        fps: videos are at 25 fps
    """
    vidpath = Path(input_path)
    if output_dir is None:
        outdir = vidpath.parent / vidpath.stem
    else:
        outdir = Path(output_dir)
    rawdir = outdir / "unprocessed"
    imgdir = outdir / "rgb"
    # top left x, y, width, height
    # bbox = [0, 120, 1920, 875]
    bbox = crop
    
    mask = None
    if maskpath is not None:
        mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)

    imgdir.mkdir(parents=True, exist_ok=True)
    rawdir.mkdir(parents=True, exist_ok=True)
    vid = cv2.VideoCapture(str(vidpath))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    fnames = []

    cnt = 0
    for i in tqdm(range(0, nframes, step), desc="Saving images"):
        vid.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = vid.read()
        if frame is None:
            # video ended before expected number of frames
            print("Video ended before expected number of frames")
            break
        fname = f"frame{str(cnt).zfill(4)}.png"
        cv2.imwrite(str(rawdir / fname), frame)
        if increase_contrast:
            frame = apply_clahe(frame, clipLimit=2.0, tileGridSize=(8, 8))
        if mask is not None:
            frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        if crop:
            cropped = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
        else:
            cropped = frame
        fnames.append(fname)
        cv2.imwrite(str(imgdir / fname), cropped)
        cnt += 1

    infodict = {
        "object_name": "placeholder.ply" if object_name is None else object_name,
        "description": "" if description is None else description,
    }

    if navpath is not None:
        # nav depth data is extremely buggy, values just jump up for no reason
        depth_noise_thresh = 6
        if start_time is None or timezone is None:
            raise ValueError("start_time and timezone must be provided if navpath is provided")
        start_t = pd.Timestamp(start_time, tz=timezone)
        dt = pd.Timedelta(seconds=step / fps)
        curr_t = start_t
        navpath = Path(navpath)
        nav = pd.read_csv(navpath)
        nav["timestamp"] = pd.to_datetime(nav["timestamp"])
        valid_nav = nav
        if denoise_depth:
            valid_nav = denoise_nav_depth(nav, thresh=depth_noise_thresh, iters=2)

        times = []
        nav_times = []
        lats = []
        lons = []
        # meters
        depths = []
        # radians, converted from degree heading measurement relative to north
        yaws = []
        # radians, converted from being relative to vehicle
        pitches = []
        # radians
        rolls = []
        for _ in fnames:
            # nearest prior time interpolation, too lazy to do linear
            navrow = valid_nav[valid_nav["timestamp"] <= curr_t].iloc[-1]
            times.append(curr_t)
            nav_times.append(navrow["timestamp"])
            lats.append(navrow["latitude"])
            lons.append(navrow["longitude"])
            depths.append(navrow["depth"])
            # convert from degrees to radians
            # change yaw/pitch to be relative to camera, not vehicle
            # yaw is not bearing anymore, so counterclockwise instead
            yaws.append((90 - navrow["heading"]) * np.pi / 180)
            pitches.append((navrow["pitch"] + 90) * np.pi / 180)
            rolls.append(navrow["roll"] * np.pi / 180)
            curr_t += dt
        df = pd.DataFrame(
            {
                "filename": fnames,
                "timestamp": times,
                "nav_timestamp": nav_times,
                "latitude": lats,
                "longitude": lons,
                "depth": depths,
                "yaw": yaws,
                "pitch": pitches,
                "roll": rolls,
            }
        )
        df.to_csv(outdir / "frame-time-nav.csv", index=False)
        infodict = {
            **infodict,
            "lat": lats[0],
            "lon": lons[0],
            "depth": depths[0],
        }
    with open(outdir / "info.json", "wt") as f:
        json.dump(infodict, f, indent=4)
