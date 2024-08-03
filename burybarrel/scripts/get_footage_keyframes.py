from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from burybarrel.utils import denoise_nav_depth
from burybarrel.image import apply_clahe


def run(
    footagecfg_path,
    footagename,
    step,
    navpath=None,
    crop=True,
    fps=25,
    increase_contrast=True,
    denoise_depth=True,
):
    """
    Args:
        fps: videos are at 25 fps
    """
    with open(footagecfg_path, "rt") as f:
        footagecfg = yaml.safe_load(f)
    cfg = footagecfg[footagename]
    vidpath = Path(cfg["path"])
    start_t = pd.Timestamp(cfg["start_time"], tz=cfg["timezone"])
    outdir = vidpath.parent / vidpath.stem

    # top left x, y, width, height
    bbox = [0, 120, 1920, 875]
    # nav depth data is extremely buggy, values just jump up for no reason
    depth_noise_thresh = 6

    outdir.mkdir(parents=True, exist_ok=True)
    vid = cv2.VideoCapture(str(vidpath))
    dt = pd.Timedelta(seconds=step / fps)
    curr_t = start_t
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    fnames = []

    cnt = 0
    for i in range(0, nframes, step):
        vid.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = vid.read()
        if frame is None:
            # video ended before expected number of frames
            break
        if increase_contrast:
            frame = apply_clahe(frame, clipLimit=2.0, tileGridSize=(8, 8))
        if crop:
            cropped = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
            fname = f"cropped{str(cnt).zfill(4)}.jpg"
        else:
            cropped = frame
            fname = f"uncropped{str(cnt).zfill(4)}.jpg"
        fnames.append(fname)
        cv2.imwrite(str(outdir / fname), cropped)
        cnt += 1

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
