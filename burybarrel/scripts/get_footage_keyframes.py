from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from burybarrel.utils import denoise_nav_depth
from burybarrel.image import apply_clahe


@click.command()
@click.option("-i", "--input", "input_vid", required=True, type=click.Path(exists=True, dir_okay=False), help="input video path")
@click.option("-o", "--output", "output_dir", required=False, type=click.Path(file_okay=False), help="output dir")
@click.option("-t", "--time", "start_time", required=False, type=click.STRING, help="start time of the video")
@click.option("--tz", "timezone", required=False, type=click.STRING, help="timezone")
@click.option(
    "--fps", "fps", required=True, default=25, show_default=True, type=click.INT
)
@click.option("--step", "step", required=True, type=click.INT, help="Number of frames between each image")
@click.option(
    "--crop/--no-crop", "crop", is_flag=True, default=True, show_default=True, type=click.BOOL
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
def get_footage_keyframes(
    input_path,
    step,
    output_dir=None,
    start_time=None,
    timezone=None,
    navpath=None,
    crop=True,
    fps=25,
    increase_contrast=True,
    denoise_depth=True,
):
    """
    Retrieves keyframes from a video at specified intervals.

    Args:
        fps: videos are at 25 fps
    """
    vidpath = Path(input_path)
    if output_dir is None:
        outdir = vidpath.parent / vidpath.stem
    else:
        outdir = Path(output_dir)
    # top left x, y, width, height
    bbox = [0, 120, 1920, 875]
    # nav depth data is extremely buggy, values just jump up for no reason
    depth_noise_thresh = 6

    outdir.mkdir(parents=True, exist_ok=True)
    vid = cv2.VideoCapture(str(vidpath))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    fnames = []

    cnt = 0
    for i in tqdm(range(0, nframes, step)):
        vid.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = vid.read()
        if frame is None:
            # video ended before expected number of frames
            break
        if increase_contrast:
            frame = apply_clahe(frame, clipLimit=2.0, tileGridSize=(8, 8))
        if crop:
            cropped = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
            fname = f"cropped{str(cnt).zfill(4)}.png"
        else:
            cropped = frame
            fname = f"uncropped{str(cnt).zfill(4)}.png"
        fnames.append(fname)
        cv2.imwrite(str(outdir / fname), cropped)
        cnt += 1

    if start_time is None:
        return
    start_t = pd.Timestamp(start_time, tz=timezone)
    dt = pd.Timedelta(seconds=step / fps)
    curr_t = start_t
    if navpath is not None:
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
